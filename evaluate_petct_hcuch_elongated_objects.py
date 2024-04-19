import matplotlib
matplotlib.use('agg')  # Set the 'agg' backend explicitly before using Matplotlib in threads

import argparse
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import concurrent.futures
import matplotlib.pyplot as plt
import pickle
import json
import pandas as pd
import torch.multiprocessing as mp
import time
from pathlib import Path
from skimage import io, transform
from segment_anything import sam_model_registry
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


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


class BBoxDataset(Dataset):
    def __init__(self, bbox_mapping, device) -> None:
        self.bbox_mapping = bbox_mapping
        self.device = device

    @property
    def bbox_mapping(self):
        return self._bbox_mapping

    @bbox_mapping.setter
    def bbox_mapping(self, value):
        self._bbox_mapping = pd.DataFrame(value)

    def __len__(self):
        return len(self.bbox_mapping)

    def __getitem__(self, index):
        path_to_slice = self.bbox_mapping.loc[index, "preprocessed_slice"]
        path_to_bbox_1024 = self.bbox_mapping.loc[index, "bbox_1024"]
        slice_1024 = io.imread(path_to_slice)
        with open(path_to_bbox_1024, 'rb') as file:
            bbox_1024 = np.load(file)
        # normalize to [0, 1], (H, W, 3)
        slice_1024 = (slice_1024 - slice_1024.min()) / np.clip(
            slice_1024.max() - slice_1024.min(), a_min=1e-8, a_max=None
        )
        # convert the shape to (3, H, W)
        slice_1024_tensor = (
            torch.tensor(slice_1024).float().permute(2, 0, 1).to(self.device)
        )
        # convert bbox to tensor
        bbox_1024_tensor = torch.as_tensor(bbox_1024, dtype=torch.float).to(self.device)
        return (index,
                slice_1024_tensor,
                bbox_1024_tensor,
                self.bbox_mapping.at[index, "slice_rows"],
                self.bbox_mapping.at[index, "slice_cols"],
                self.bbox_mapping.at[index, "annotator"])


class Evaluator:
    """Run evaluation of MedSAM model on elongated 2D objects
    from Gatidis dataset annotated by HCUCH radiologists."""
    def __init__(self, path_to_studies, path_to_objects, path_to_output,
                 path_to_checkpoint, window=None, device='cuda:0',
                 num_workers=4, batch_size=8, threads=8,
                 bbox_scale_factor=1.0, save_overlapped=True) -> None:
        self.path_to_studies = path_to_studies
        self.path_to_objects = path_to_objects
        self.path_to_output = path_to_output
        self.path_to_checkpoint = path_to_checkpoint
        self.window = window
        self.device = device
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.threads = threads
        self.bbox_scale_factor = bbox_scale_factor
        self.save_overlapped = save_overlapped
        self._ct_filename = 'imaging.nii.gz'
        self._original_suffix = "original_size"
        self._preprocessed_suffix = "preprocessed"
        self._embeddings_suffix = "embedding"
        self._gt_masks_suffix = "gt_masks"
        self._bbox_original_suffix = "bboxes_original"
        self._bbox1024_suffix = "bboxes_1024"
        self._output_masks_suffix = "output_masks"
        self._output_overlapped_suffix = "output_overlapped"
        self._epsilon = 1e-6

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, value):
        if value is not None:
            assert "L" in value.keys() and "W" in value.keys()
        self._window = value

    def _preprocess_ct(self, ct):
        """Apply window normalization to a CT volume and resize specified
        slices according to the MedSAM pipeline."""
        ct_annotator = ct["annotator"]
        ct_study = ct["study"]
        ct_slice_inds = ct["ct_slice_inds"]
        if not ct_slice_inds:
            return
        print(f"study: {ct_study}, annotator: {ct_annotator}, annotated slices: {len(ct_slice_inds)}, processing ...")
        ct_array = sitk.GetArrayFromImage(sitk.ReadImage(Path(self.path_to_studies) / ct_annotator / ct_study / self._ct_filename))
        path_to_preprocessed = Path(self.path_to_output) / ct_annotator / self._preprocessed_suffix
        path_to_original_size = Path(self.path_to_output) / ct_annotator / self._original_suffix
        path_to_preprocessed.mkdir(exist_ok=True, parents=True)
        path_to_original_size.mkdir(exist_ok=True, parents=True)
        # CT normalization
        if self.window:
            lower_bound = self.window["L"] - self.window["W"] / 2
            upper_bound = self.window["L"] + self.window["W"] / 2
            ct_array_pre = np.clip(ct_array, lower_bound, upper_bound)
            ct_array_pre = (
                (ct_array_pre - np.min(ct_array_pre) + self._epsilon)
                / (np.max(ct_array_pre) - np.min(ct_array_pre) + self._epsilon)
                * 255.0
            )
        else:
            lower_bound= np.percentile(ct_array[ct_array > 0], 0.5)
            upper_bound = np.percentile(ct_array[ct_array > 0], 99.5)
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
                path_to_preprocessed / f"{ct_study}_slice{slice_idx}.png",
                slice_1024,
                check_contrast=False
            )
            io.imsave(
                path_to_original_size / f"{ct_study}_slice{slice_idx}.png",
                slice_3c.astype(np.uint8),
                check_contrast=False
            )

    def _preprocess_cts_concurrently(self):
        print("Preprocessing CTs ...")
        with open(self.path_to_objects, 'rb') as file:
            objects = pickle.load(file)
        cts = []
        for annotator in objects.keys():
            for study in objects[annotator]:
                slice_inds = [
                    object_['slice_idx']
                    for object_ in objects[annotator][study]
                ]
                cts += [
                    {
                        "study": study,
                        "annotator": annotator,
                        "ct_slice_inds": list(set(slice_inds))
                    }
                ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            executor.map(self._preprocess_ct, cts)

    def _scale_bounding_box(self, bbox_array):
        """Return a scaled version of the input bounding box given
        as a numpy array [min_col, min_row, max_col, max_row]."""
        bbox_array = bbox_array.squeeze()
        center_col = (bbox_array[0] + bbox_array[2]) / 2.0
        center_row = (bbox_array[1] + bbox_array[3]) / 2.0
        width = bbox_array[2] - bbox_array[0]
        height = bbox_array[3] - bbox_array[1]
        new_width = width * self.bbox_scale_factor
        new_height = height * self.bbox_scale_factor
        new_min_col = center_col - new_width / 2.0
        new_min_row = center_row - new_height / 2.0
        new_max_col = center_col + new_width / 2.0
        new_max_row = center_row + new_height / 2.0
        return np.array([[new_min_col, new_min_row, new_max_col, new_max_row]])

    def _save_bounding_box(self, bbox):
        """Save bounding box (original and resized to 1024x1024) as a
        numpy array and append it to self._bboxes."""
        annotator = bbox['annotator']
        study = bbox['study']
        slice_idx = bbox['slice_idx']
        object_id = bbox['object_id']
        bbox_array = np.array([[
            bbox['bbox'][1],
            bbox['bbox'][0],
            bbox['bbox'][3],
            bbox['bbox'][2]
        ]])
        print(f"study: {study}, slice: {slice_idx}, object_id: {object_id}, processing ...")
        H, W = sitk.GetArrayFromImage(sitk.ReadImage(
            Path(self.path_to_studies) /
            annotator /
            study /
            self._ct_filename
        ))[slice_idx].shape
        bbox_array_original = self._scale_bounding_box(bbox_array)
        bbox_array_1024 = self._scale_bounding_box(bbox_array / np.array([W, H, W, H]) * 1024)
        path_to_bbox_original = (
            Path(self.path_to_output) /
            annotator /
            self._bbox_original_suffix /
            bbox['study'] /
            f"{study}_slice{slice_idx}_{int(bbox_array_original.squeeze()[0])}_{int(bbox_array_original.squeeze()[1])}_{int(bbox_array_original.squeeze()[2])}_{int(bbox_array_original.squeeze()[3])}.npy"
        )
        path_to_bbox_1024 = (
            Path(self.path_to_output) /
            annotator /
            self._bbox1024_suffix /
            bbox['study'] /
            f"{study}_slice{slice_idx}_{int(bbox_array_1024.squeeze()[0])}_{int(bbox_array_1024.squeeze()[1])}_{int(bbox_array_1024.squeeze()[2])}_{int(bbox_array_1024.squeeze()[3])}.npy"
        )
        bbox.update({
            "path_to_bbox_original": str(path_to_bbox_original),
            "path_to_bbox_1024": str(path_to_bbox_1024),
            "path_to_original_slice": str(Path(self.path_to_output) / annotator / self._original_suffix / f"{study}_slice{slice_idx}.png"),
            "path_to_preprocessed_slice": str(Path(self.path_to_output) / annotator / self._preprocessed_suffix / f"{study}_slice{slice_idx}.png"),
            "slice_rows": H,
            "slice_cols": W,
            "bbox_original": bbox_array_original,
            "bbox_1024": bbox_array_1024
        })
        for path, key in zip([path_to_bbox_original, path_to_bbox_1024], ['bbox_original', 'bbox_1024']):
            if not path.parent.exists():
                path.parent.mkdir(parents=True)
            with open(path, 'wb') as file:
                np.save(file, bbox[key])
        self._bboxes.append(bbox)

    def _save_bounding_boxes_concurrently(self):
        print("Saving bounding boxes ...")
        with open(self.path_to_objects, 'rb') as file:
            objects = pickle.load(file)
        bboxes = [
            {
                "annotator": annotator,
                **object_
            }
            for annotator, studies in objects.items()
            for objects in studies.values()
            for object_ in objects
        ]
        self._bboxes = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            executor.map(self._save_bounding_box, bboxes)

    @torch.no_grad()
    def _run_inference(self, bbox_dataset):
        """Run inference on bounding boxes."""
        print("Running inference ...")
        dataloader = DataLoader(
            bbox_dataset,
            self.batch_size,
            num_workers=self.num_workers
        )
        medsam_model = sam_model_registry["vit_b"](checkpoint=self.path_to_checkpoint)
        medsam_model = medsam_model.to(self.device)
        medsam_model.eval()
        for index, slice_1024, bbox_1024, slice_rows, slice_cols, annotator in tqdm(dataloader):
            # Apply image encoder
            with torch.no_grad():
                embedding_tensor = medsam_model.image_encoder(slice_1024)  # (1, 256, 64, 64)
 
            # Apply bbox (prompt) encoder and mask decoder
            sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
                points=None,
                boxes=bbox_1024,
                masks=None,
            )
            low_res_logits, _ = medsam_model.mask_decoder(
                image_embeddings=embedding_tensor,  # (B, 256, 64, 64)
                image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )

            low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

            # save individual output masks in original resolution
            for low_res_sample, fname_idx, H, W, annotator_ in zip(low_res_pred, index, slice_rows, slice_cols, annotator):
                low_res_sample = low_res_sample.unsqueeze(0)
                fname_idx = fname_idx.item()
                H = H.item()
                W = W.item()
                low_res_sample = F.interpolate(
                    low_res_sample,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )  # (1, 1, gt.shape)
                low_res_sample = low_res_sample.squeeze().cpu().numpy()  # (256, 256)
                medsam_seg = (low_res_sample > 0.5).astype(np.uint8)
                bbox_fname_sample = Path(bbox_dataset.bbox_mapping.loc[fname_idx, "bbox_original"]).name
                path_to_output_mask = Path(self.path_to_output) / annotator_ / self._output_masks_suffix / f"{bbox_fname_sample.split('.npy')[0]}.png"
                if not path_to_output_mask.parent.exists():
                    path_to_output_mask.parent.mkdir(parents=True)
                io.imsave(
                    path_to_output_mask,
                    np.uint8(medsam_seg * 255.0),
                    check_contrast=False
                )

    def _compute_performance(self, bbox):
        """Compute performance on a single bounding box."""
        print(f"bbox: {Path(bbox['path_to_bbox_original']).name}, processing ...")
        # read predicted mask in original size
        path_to_output_mask = (
            Path(self.path_to_output) /
            bbox['annotator'] /
            self._output_masks_suffix /
            f"{Path(bbox['path_to_bbox_original']).name.split('.npy')[0]}.png"
        )
        medsam_seg = np.uint8(io.imread(path_to_output_mask).astype('bool'))
        # measure and save performance
        gt_mask = np.zeros((bbox['slice_rows'], bbox['slice_cols'])).astype(np.uint8)
        gt_mask[bbox['coordinates'][:, 0], bbox['coordinates'][:, 1]] = 1
        dice_score = dice_coefficient(gt_mask, medsam_seg, are_binary=False)
        bbox_fname = Path(bbox["path_to_bbox_original"]).name
        performance = {
            key: value
            for key, value in bbox.items()
            if key not in ('bbox_original', 'bbox_1024', 'coordinates')
        }
        performance.update({
            "bbox_original_fname": bbox_fname,
            "bbox_original": [int(item) for item in bbox["bbox_original"].squeeze().tolist()],
            "bbox_1024": [int(item) for item in bbox["bbox_1024"].squeeze().tolist()],
            "annotated_pixels": int(np.sum(gt_mask > 0)),
            "predicted_pixels": int(np.sum(medsam_seg > 0)),
            "dice_score": dice_score
        })
        self._performance.append(performance)
        # save gt
        bbox_fname_png = f"{bbox_fname.split('.npy')[0]}.png"
        path_to_gt_mask = Path(self.path_to_output) / bbox['annotator'] / self._gt_masks_suffix / bbox_fname_png
        if not path_to_gt_mask.parent.exists():
            path_to_gt_mask.parent.mkdir(parents=True)
        io.imsave(
            path_to_gt_mask,
            np.uint8(gt_mask * 255.0),
            check_contrast=False
        )
        # save plotting with overlapped output
        if self.save_overlapped:
            img_3c = io.imread(bbox["path_to_original_slice"])
            _, ax = plt.subplots(1, 3, figsize=(22.5, 7.5))
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
            path_to_output_overlapped = Path(self.path_to_output) / bbox["annotator"] / self._output_overlapped_suffix / bbox_fname_png
            if not path_to_output_overlapped.parent.exists():
                path_to_output_overlapped.parent.mkdir(parents=True)
            plt.savefig(path_to_output_overlapped)
            plt.close()

    def _compute_performance_concurrently(self):
        """Compute performance from predicted masks."""
        print("Computing performance ...")
        self._performance = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
            executor.map(self._compute_performance, self._bboxes)
        return self._performance

    def run_evaluation(self):
        """Run MedSAM inference on bounding boxes and measure the
        Dice coefficient."""
        assert torch.cuda.is_available(), "You need GPU and CUDA to make the evaluation."
        self._preprocess_cts_concurrently()
        self._save_bounding_boxes_concurrently()
        bbox_mapping = [
            {
                "annotator": item["annotator"],
                "study": item["study"],
                "slice_idx": item["slice_idx"],
                "slice_rows": item["slice_rows"],
                "slice_cols": item["slice_cols"],
                "bbox_1024": item["path_to_bbox_1024"],
                "row_min_1024": item["bbox_1024"].squeeze()[0],
                "col_min_1024": item["bbox_1024"].squeeze()[1],
                "row_max_1024": item["bbox_1024"].squeeze()[2],
                "col_max_1024": item["bbox_1024"].squeeze()[3],
                "preprocessed_slice": item["path_to_preprocessed_slice"],
                "bbox_original": item["path_to_bbox_original"],
                "row_min_original": item["bbox_original"].squeeze()[0],
                "col_min_original": item["bbox_original"].squeeze()[1],
                "row_max_original": item["bbox_original"].squeeze()[2],
                "col_max_original": item["bbox_original"].squeeze()[3],
            }
            for item in self._bboxes
        ]
        bbox_dataset = BBoxDataset(bbox_mapping, self.device)
        self._run_inference(bbox_dataset)
        performance = self._compute_performance_concurrently()
        results = {
            "bboxes": self._bboxes,
            "performance": performance
        }
        return results


if __name__ == "__main__":
    mp.set_start_method('spawn')
    windows = {
        "lung": {"L": -500, "W": 1400},
        "abdomen": {"L": 40, "W": 350},
        "bone": {"L": 400, "W": 1000},
        "air": {"L": -426, "W": 1000},
        "brain": {"L": 50, "W": 100},
        "mediastinum": {"L": 50, "W": 350}
    }
    parser = argparse.ArgumentParser(
        description="""Evaluate MedSAM on elongated 2d objects annotated
        by HCUCH radiologists on Gatidis dataset.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_studies',
        type=str,
        help="""Path to the directory with the CT studies in NIfTI format
        (.nii.gz). Expected format: annotators -> studies -> imaging.nii.gz"""
    )
    parser.add_argument(
        'path_to_objects',
        type=str,
        help="""Path to the pickle file containing the 2d objects as
        a dictionary. Expected format: 'annotators': {'studies': [objects]}."""
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
        help="""Window for CT normalization. If None, values are clipped
        to percentiles 0.5 and 99.5, and then mapped to the range 0-255."""
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=8,
        help="Maximum number of concurrent threads for CPU tasks."
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help="Max workers for GPU processing."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help="Batch size for inference using GPU."
    )
    parser.add_argument(
        '--bbox_scale_factor',
        type=float,
        default=1.0,
        help="""Factor to scale the size (in both dimensions) of
        bounding boxes computed from the connected components
        in the annotated masks."""
    )
    parser.add_argument(
        '--dont_save_overlapped',
        action='store_true',
        help="""Add this flag to avoid saving output images
        with masks and boxes overlapped in the image."""
    )
    args = parser.parse_args()
    window = windows.get(args.window, None)
    evaluator = Evaluator(
        path_to_studies=args.path_to_studies,
        path_to_objects=args.path_to_objects,
        path_to_output=args.path_to_output,
        path_to_checkpoint=args.path_to_checkpoint,
        threads=args.threads,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        bbox_scale_factor=args.bbox_scale_factor,
        save_overlapped = not args.dont_save_overlapped,
        window=window
    )
    start_time = time.time()
    results = evaluator.run_evaluation()
    end_time = time.time()
    execution_time = end_time - start_time
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int(execution_time % 60)
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    print(f"execution time (HH:MM:SS): {formatted_time}")
    with open(Path(args.path_to_output) / 'bboxes.pkl', 'wb') as file:
        pickle.dump(results["bboxes"], file)
    json_format_performance = [
        {
            key: int(value) if isinstance(value, np.int64) else value
            for key, value in object_.items()
        }
        for object_ in results["performance"]
    ]
    with open(Path(args.path_to_output) / 'performance.json', 'w') as file:
        json.dump(
            {
                "bboxes": json_format_performance,
                "execution time (HH:MM:SS)": formatted_time
            },
            file,
            indent=4
        )
    with open(Path(args.path_to_output) / 'arguments.json', 'w') as file:
        json.dump({**vars(args), "window_L_W": window}, file, indent=4)
