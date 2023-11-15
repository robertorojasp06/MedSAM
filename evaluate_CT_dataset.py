import argparse
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import concurrent.futures
import matplotlib.pyplot as plt
import pickle
import json
from pathlib import Path
from skimage import io, transform
from skimage.measure import label, regionprops
from segment_anything import sam_model_registry
from tqdm import tqdm


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


def dice_coefficient(reference, prediction, are_binary=False):
    if not are_binary:
        reference = reference > 0
        prediction = prediction > 0
    if reference.sum() == 0:
        dice_score = np.nan
    else:
        intersection = np.logical_and(reference, prediction).sum()
        dice_score = (2. * intersection) / (reference.sum() + prediction.sum())
    return dice_score


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


class Evaluator:
    """Run evaluation of MedSAM model on CT dataset."""
    def __init__(self, path_to_cts, path_to_masks, path_to_output, device='cuda:0',
                 dataset_fname_relation='same_name', foreground_label=1) -> None:
        self.path_to_cts = path_to_cts
        self.path_to_masks = path_to_masks
        self.path_to_output = path_to_output
        self.foreground_label = foreground_label
        self.dataset_fname_relation = dataset_fname_relation
        self.device = device
        self._input_fname_extension = '.nii.gz'
        self._original_suffix = "original_size"
        self._preprocessed_suffix = "preprocessed"
        self._embeddings_suffix = "embedding"
        self._gt_masks_suffix = "gt_masks"
        self._output_masks_suffix = "output_masks"
        self._output_overlapped_suffix = "output_overlapped"
        self._epsilon = 1e-6

    def _identify_slices(self):
        """Return a list with dictionaries of each slice containing
        2D connected components."""
        paths_to_masks = [
            item
            for item in Path(self.path_to_masks).glob(f"*{self._input_fname_extension}")
        ]
        slices = []
        slices_count = 0
        for path in tqdm(paths_to_masks):
            mask = np.uint8(sitk.GetArrayFromImage(sitk.ReadImage(path)))
            if np.any(mask == 1):
                if self.dataset_fname_relation == 'nnunet':
                    ct_fname = f"{path.name.split(self._input_fname_extension)[0]}_0000{self._input_fname_extension}"
                else:
                    ct_fname = path.name
                for idx, slice_ in enumerate(mask):
                    if np.any(slice_ == self.foreground_label):
                        slices.append({
                            "slice_id": slices_count,
                            "path_to_cts": self.path_to_cts,
                            "path_to_masks": self.path_to_masks,
                            "path_to_output": self.path_to_output,
                            "dataset_fname_relation": self.dataset_fname_relation,
                            "mask_fname": path.name,
                            "ct_fname": ct_fname,
                            "slice_idx": idx,
                            "slice_rows": slice_.shape[0],
                            "slice_cols": slice_.shape[1]
                        })
                        slices_count += 1
        return slices

    def _preprocess_ct(self, ct):
        """Preprocess slices from a CT volume according to the MedSAM
        pipeline."""
        ct_fname = ct["ct_fname"]
        ct_slice_inds = ct["ct_slice_inds"]
        window = ct["window"]
        if not ct_slice_inds:
            return
        print(f"ct_fname: {ct_fname}, annotated slices: {len(ct_slice_inds)}, processing ...")
        ct_array = sitk.GetArrayFromImage(sitk.ReadImage(Path(self.path_to_cts) / ct_fname))
        path_to_out_preprocessed = Path(self.path_to_output) / self._preprocessed_suffix
        path_to_original_size = Path(self.path_to_output) / self._original_suffix
        path_to_out_preprocessed.mkdir(exist_ok=True)
        path_to_original_size.mkdir(exist_ok=True)
        # CT normalization
        if window:
            lower_bound = window["L"] - window["W"] / 2
            upper_bound = window["L"] + window["W"] / 2
            ct_array_pre = np.clip(ct_array, lower_bound, upper_bound)
            ct_array_pre = (
                (ct_array_pre - np.min(ct_array_pre) + self._epsilon)
                / (np.max(ct_array_pre) - np.min(ct_array_pre) + self._epsilon)
                * 255.0
            )
        else:
            lower_bound, upper_bound = np.percentile(
                ct_array[ct_array > 0], 0.5
            ), np.percentile(ct_array[ct_array > 0], 99.5)
            ct_array_pre = np.clip(ct_array, lower_bound, upper_bound)
            ct_array_pre = (
                (ct_array_pre - np.min(ct_array_pre) + self._epsilon)
                / (np.max(ct_array_pre) - np.min(ct_array_pre) + self._epsilon)
                * 255.0
            )
            ct_array_pre[ct_array == 0] = 0
        ct_array_pre = np.uint8(ct_array_pre)
        # slice preprocessing
        for slice_idx in ct_slice_inds:
            slice_npy = ct_array_pre[slice_idx, :, :]
            slice_3c = np.repeat(slice_npy[:, :, None], 3, axis=-1)
            # image preprocessing
            slice_1024 = transform.resize(
                slice_3c,
                (1024, 1024),
                order=3,
                mode='constant',
                preserve_range=True,
                anti_aliasing=True
            ).astype(np.uint8)
            io.imsave(
                path_to_out_preprocessed / f"{ct_fname.split(self._input_fname_extension)[0]}_slice{slice_idx}.png",
                slice_1024,
                check_contrast=False
            )
            io.imsave(
                path_to_original_size / f"{ct_fname.split(self._input_fname_extension)[0]}_slice{slice_idx}.png",
                slice_3c.astype(np.uint8),
                check_contrast=False
            )

    def _preprocess_slices(self, slices, window, threads):
        """Preprocess the input set of slices according to the
        MedSAM pipeline."""
        unique_cts = [item["ct_fname"] for item in slices]
        unique_cts = list(set(unique_cts))
        ct_to_slices = {item: [] for item in unique_cts}
        for slice_ in slices:
            ct_to_slices[slice_["ct_fname"]].append(slice_)
        cts = []
        for ct_fname, ct_slices in ct_to_slices.items():
            ct = {
                "ct_fname": ct_fname,
                "ct_slice_inds": [item["slice_idx"] for item in ct_slices],
                "window": window
            }
            cts.append(ct)
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            executor.map(self._preprocess_ct, cts)


    def _compute_image_embeddings(self, slices, path_to_checkpoint):
        """Compute image embeddings and save them as torch arrays."""
        path_to_embedding_output = Path(self.path_to_output) / self._embeddings_suffix
        path_to_embedding_output.mkdir(exist_ok=True)
        medsam_model = sam_model_registry["vit_b"](checkpoint=path_to_checkpoint)
        medsam_model = medsam_model.to(self.device)
        medsam_model.eval()
        for slice_ in tqdm(slices):
            path_to_slice = (Path(slice_["path_to_output"]) /
                                self._preprocessed_suffix /
                                f"{slice_['ct_fname'].split(self._input_fname_extension)[0]}_slice{slice_['slice_idx']}.png"
                            )
            slice_1024 = io.imread(path_to_slice)
            # normalize to [0, 1], (H, W, 3)
            slice_1024 = (slice_1024 - slice_1024.min()) / np.clip(
                slice_1024.max() - slice_1024.min(), a_min=1e-8, a_max=None
            )
            # convert the shape to (3, H, W)
            slice_1024_tensor = (
                torch.tensor(slice_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            )
            # Apply image encoder
            with torch.no_grad():
                image_embedding = medsam_model.image_encoder(slice_1024_tensor)  # (1, 256, 64, 64)
            path_to_embedding = (path_to_embedding_output /
                                 f"{path_to_slice.name.split('.png')[0]}.pt"
            ) 
            torch.save(image_embedding, path_to_embedding)

    def _compute_bounding_boxes(self, slices):
        """Return a list of dictionaries for each bounding box
        [min_col, min_row, max_col, max_row]."""
        bboxes = []
        for slice_ in tqdm(slices):
            path_to_mask= (
                Path(slice_['path_to_masks']) /
                slice_['mask_fname']
            )
            mask = sitk.GetArrayFromImage(sitk.ReadImage(path_to_mask))[slice_['slice_idx']]
            labeled = label(mask == self.foreground_label)
            props = regionprops(labeled)
            H, W = slice_['slice_rows'], slice_['slice_cols']
            bboxes_slice = [
                {
                    **slice_,
                    "bbox_original": np.array([[object['bbox'][1], object['bbox'][0], object['bbox'][3], object['bbox'][2]]]),
                    "bbox_1024": np.array([[object['bbox'][1], object['bbox'][0], object['bbox'][3], object['bbox'][2]]]) / np.array([W, H, W, H]) * 1024,
                    "object_coords": object['coords']
                }
                for object in props
            ]
            bboxes += bboxes_slice
        return bboxes

    def _run_inference(self, bboxes, path_to_checkpoint):
        """Run inference on bboxes."""
        path_to_gt_masks = Path(self.path_to_output) / self._gt_masks_suffix
        path_to_output_masks = Path(self.path_to_output) / self._output_masks_suffix
        path_to_output_overlapped = Path(self.path_to_output) / self._output_overlapped_suffix
        path_to_gt_masks.mkdir(exist_ok=True)
        path_to_output_masks.mkdir(exist_ok=True)
        path_to_output_overlapped.mkdir(exist_ok=True)
        medsam_model = sam_model_registry["vit_b"](checkpoint=path_to_checkpoint)
        medsam_model = medsam_model.to(self.device)
        medsam_model.eval()
        performance = []
        for bbox_idx, bbox in tqdm(enumerate(bboxes), total=len(bboxes)):
            path_to_embedding = (
                Path(bbox['path_to_output']) /
                self._embeddings_suffix /
                f"{bbox['ct_fname'].split(self._input_fname_extension)[0]}_slice{bbox['slice_idx']}.pt"
            )
            embedding_tensor = torch.load(path_to_embedding)
            medsam_seg = medsam_inference(
                medsam_model,
                embedding_tensor,
                bbox['bbox_1024'],
                bbox['slice_rows'],
                bbox['slice_cols']
            )
            # measure and save performance
            gt_mask = np.zeros((bbox['slice_rows'], bbox['slice_cols'])).astype(np.uint8)
            gt_mask[bbox['object_coords'][:, 0], bbox['object_coords'][:, 1]] = 1
            dice_score = dice_coefficient(gt_mask, medsam_seg, are_binary=False)
            bbox_fname = f"{path_to_embedding.name.split('.pt')[0]}_bbox{bbox_idx}.png"
            performance_dict = {
                key: value
                for key, value in bbox.items()
                if key not in ('bbox_original', 'bbox_1024', 'object_coords')
            }
            performance_dict.update({
                "bbox_fname": bbox_fname,
                "bbox_original": bbox["bbox_original"].squeeze().tolist(),
                "bbox_1024": bbox["bbox_1024"].squeeze().tolist(),
                "annotated_pixels": int(np.sum(gt_mask > 0)),
                "predicted_pixels": int(np.sum(medsam_seg > 0)),
                "dice_score": dice_score
            })
            performance.append(performance_dict)
            # save outputs
            io.imsave(
                Path(self.path_to_output) / self._gt_masks_suffix / bbox_fname,
                np.uint8(gt_mask * 255.0),
                check_contrast=False
            )
            io.imsave(
                Path(self.path_to_output) / self._output_masks_suffix / bbox_fname,
                np.uint8(medsam_seg * 255.0),
                check_contrast=False
            )
            # plotting
            img_3c = io.imread(Path(bbox['path_to_output']) / self._original_suffix / f"{path_to_embedding.name.split('.pt')[0]}.png")
            _, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(img_3c)
            show_box(bbox['bbox_original'].squeeze(), ax[0])
            ax[0].set_title("Input Image and Bounding Box")
            ax[1].imshow(img_3c)
            show_mask(gt_mask, ax[1])
            show_box(bbox['bbox_original'].squeeze(), ax[1])
            ax[1].set_title("Ground truth")
            ax[2].imshow(img_3c)
            show_mask(medsam_seg, ax[2])
            show_box(bbox['bbox_original'].squeeze(), ax[2])
            ax[2].set_title(f"MedSAM Segmentation (Dice={round(dice_score, 3)})")
            plt.tight_layout()
            plt.savefig(path_to_output_overlapped / f"{path_to_embedding.name.split('.pt')[0]}_bbox{bbox_idx}.png")
            plt.close()
        return performance

    def run_evaluation(self, path_to_checkpoint, window=None, threads=1):
        """Run MedSAM inference on bounding boxes and measure the
        Dice coefficient."""
        assert torch.cuda.is_available(), "You need GPU and CUDA to make the evaluation."
        slices = self._identify_slices()
        self._preprocess_slices(slices, window, threads)
        self._compute_image_embeddings(slices, path_to_checkpoint)
        bboxes = self._compute_bounding_boxes(slices)
        performance = self._run_inference(bboxes, path_to_checkpoint)
        results = {
            "slices": slices,
            "bboxes": bboxes,
            "performance": performance
        }
        return results


if __name__ == "__main__":
    windows = {
        "lung": {"L": -500, "W": 1400},
        "abdomen": {"L": 40, "W": 350},
        "bone": {"L": 400, "W": 1000},
        "air": {"L": -426, "W": 1000},
        "brain": {"L": 50, "W": 100}
    }
    parser = argparse.ArgumentParser(
        description="""Evaluate MedSAM on a CT dataset with annotated
        masks.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_cts',
        type=str,
        help="""Path to the directory with CT studies in NIfTI format
        (.nii.gz)."""
    )
    parser.add_argument(
        'path_to_masks',
        type=str,
        help="""Path to the directory with corresponding masks in NIfTI
        format (.nii.gz)."""
    )
    parser.add_argument(
        'path_to_output',
        type=str,
        help="Path to the directory to save output results."
    )
    parser.add_argument(
        '--path_to_checkpoint',
        type=str,
        default="work_dir/MedSAM/medsam_vit_b.pth",
        help="Path to the checkpoint model."
    )
    parser.add_argument(
        '--window',
        choices=list(windows.keys()),
        default=None,
        help="Window for CT normalization."
    )
    parser.add_argument(
        '--dataset_fname_relation',
        choices=['nnunet', 'same_name'],
        default='nnunet',
        help="""Relation between CT studies and masks filenames.
        'nnunet' means nnUNet pipeline filenames format, and 'same_name'
         means CT and mask NIfTI volumes have the same filename."""
    )
    parser.add_argument(
        '--foreground_label',
        type=int,
        default=1,
        help="""Integer label corresponding to the foreground class in
        the annotated masks."""
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=4,
        help="Maximum number of concurrent threads for CPU tasks."
    )
    parser.add_argument(
        '--inference_only',
        dest='inference_only',
        action='store_true',
        help="""Add this argument to execute only inference. Require
        the previous outputs in the specified output folder."""
    )
    args = parser.parse_args()
    window = windows.get(args.window, None)
    evaluator = Evaluator(
        path_to_cts=args.path_to_cts,
        path_to_masks=args.path_to_masks,
        path_to_output=args.path_to_output,
        dataset_fname_relation=args.dataset_fname_relation,
        foreground_label=args.foreground_label
    )
    if args.inference_only:
        with open(Path(args.path_to_output) / 'slices.pkl', 'rb') as file:
            slices = pickle.load(file)
        with open(Path(args.path_to_output) / 'bboxes.pkl', 'rb') as file:
            bboxes = pickle.load(file)
        performance = evaluator._run_inference(bboxes, args.path_to_checkpoint)
        results = {
            "slices": slices,
            "bboxes": bboxes,
            "performance": performance
        }        
    else:
        results = evaluator.run_evaluation(
            args.path_to_checkpoint,
            window,
            args.threads
        )
    with open(Path(args.path_to_output) / 'slices.pkl', 'wb') as file:
        pickle.dump(results["slices"], file)
    with open(Path(args.path_to_output) / 'bboxes.pkl', 'wb') as file:
        pickle.dump(results["bboxes"], file)
    with open(Path(args.path_to_output) / 'performance.json', 'w') as file:
        json.dump(results["performance"], file, indent=4)
    with open(Path(args.path_to_output) / 'arguments.json', 'w') as file:
        json.dump({**vars(args), "window_L_W": window}, file, indent=4)
