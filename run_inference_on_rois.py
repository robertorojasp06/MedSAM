import argparse
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from skimage import transform

from segment_anything import sam_model_registry


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


def normalize_ct(ct_array, window=None, epsilon = 1e-6):
    if window:
        lower_bound = window["L"] - window["W"] / 2
        upper_bound = window["L"] + window["W"] / 2
        ct_array_pre = np.clip(ct_array, lower_bound, upper_bound)
        ct_array_pre = (
            (ct_array_pre - np.min(ct_array_pre) + epsilon)
            / (np.max(ct_array_pre) - np.min(ct_array_pre) + epsilon)
            * 255.0
        )
    else:
        lower_bound= np.percentile(ct_array[ct_array > 0], 0.5)
        upper_bound = np.percentile(ct_array[ct_array > 0], 99.5)
        ct_array_pre = np.clip(ct_array, lower_bound, upper_bound)
        ct_array_pre = (
            (ct_array_pre - np.min(ct_array_pre) + epsilon)
            / (np.max(ct_array_pre) - np.min(ct_array_pre) + epsilon)
            * 255.0
        )
        ct_array_pre[ct_array == 0] = 0
    return np.uint8(ct_array_pre)


def draw_boxes(ct_array, rois, path_to_output):
    path_to_output = Path(path_to_output) / "overlapped-bboxes"
    path_to_output.mkdir(exist_ok=True)
    for roi_counter, roi in enumerate(tqdm(rois), start=1):
        for slice_idx in tqdm(range(min(roi["slices"]), max(roi["slices"]) + 1), leave=False):
            _, ax = plt.subplots()
            slice_npy = np.flip(ct_array[slice_idx, :, :], axis=0)
            slice_3c = np.repeat(slice_npy[:, :, None], 3, axis=-1)
            ax.imshow(slice_3c)
            show_box(roi["bbox"], ax)
            path_to_overlapped = Path(path_to_output) / f"roi_{roi_counter}_bbox_{roi['bbox'][0]}_{roi['bbox'][1]}_{roi['bbox'][2]}_{roi['bbox'][3]}_slice_{slice_idx}.png"
            plt.tight_layout()
            plt.savefig(path_to_overlapped)
            plt.close()


def format_slice_tensor(slice_1024, device):
    # normalize to [0, 1], (H, W, 3)
    slice_1024 = (slice_1024 - slice_1024.min()) / np.clip(
        slice_1024.max() - slice_1024.min(), a_min=1e-8, a_max=None
    )
    # convert the shape to (3, H, W)
    slice_1024_tensor = (
        torch.tensor(slice_1024).float().permute(2, 0, 1).to(device)
    )
    # add batch dimension
    slice_1024_tensor = slice_1024_tensor.unsqueeze(0)
    return slice_1024_tensor


def format_bbox_tensor(bbox_1024, device):
    # convert bbox to tensor
    bbox_1024_tensor = torch.as_tensor(bbox_1024, dtype=torch.float).to(device)
    # add batch dimension
    bbox_1024_tensor = bbox_1024_tensor.unsqueeze(0)
    return bbox_1024_tensor


@torch.no_grad()
def run_inference(ct_array, rois, path_to_checkpoint, path_to_output,
                  device=torch.device('cpu')):
    path_to_output_overlapped_original = Path(path_to_output) / 'output-overlapped-original'
    path_to_output_overlapped_original.mkdir(exist_ok=True)
    medsam_model = sam_model_registry["vit_b"](checkpoint=path_to_checkpoint)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    ct_mask = np.zeros(ct_array.shape).astype(np.uint8)
    for roi_counter, roi in enumerate(tqdm(rois), start=1):
        for slice_idx in tqdm(range(min(roi["slices"]), max(roi["slices"]) + 1), leave=False):
            # preprocess slice
            slice_npy = np.flip(ct_array[slice_idx, :, :], axis=0)
            slice_3c = np.repeat(slice_npy[:, :, None], 3, axis=-1)
            slice_1024 = transform.resize(
                slice_3c,
                (1024, 1024),
                order=3,
                mode='constant',
                preserve_range=True,
                anti_aliasing=True
            ).astype(np.uint8)
            # preprocess bounding box
            bbox_original = np.array([[roi['bbox'][0], roi['bbox'][1], roi['bbox'][2], roi['bbox'][3]]])
            bbox_1024 = np.array([[roi['bbox'][0], roi['bbox'][1], roi['bbox'][2], roi['bbox'][3]]]) / np.array([roi['W'], roi['H'], roi['W'], roi['H']]) * 1024
            # get embedding for slice (image encoder)
            slice_1024_tensor = format_slice_tensor(slice_1024, device)
            embedding_tensor = medsam_model.image_encoder(slice_1024_tensor)  # (1, 256, 64, 64)
            # get embedding for bounding box (prompt encoder)
            bbox_1024_tensor = format_bbox_tensor(bbox_1024, device)
            sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
                points=None,
                boxes=bbox_1024_tensor,
                masks=None,
            )
            # apply decoder
            low_res_logits, _ = medsam_model.mask_decoder(
                image_embeddings=embedding_tensor,  # (B, 256, 64, 64)
                image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )
            low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
            # get output mask in original resolution
            low_res_pred = F.interpolate(
               low_res_pred,
               size=(roi['H'], roi['W']),
               mode="bilinear",
               align_corners=False,
            ) # (1, 1, gt.shape)
            medsam_seg = (low_res_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) # (H, W)
            # save output mask overlapped (original resolution)
            path_to_output_original = (
                path_to_output_overlapped_original /
                f"roi_{roi_counter}_bbox_{roi['bbox'][0]}_{roi['bbox'][1]}_{roi['bbox'][2]}_{roi['bbox'][3]}_slice_{slice_idx}.png"
            )
            _, ax = plt.subplots(1, 2, figsize=(15, 7.5))
            ax[0].imshow(slice_3c)
            show_box(bbox_original.squeeze(), ax[0])
            ax[0].set_title("Input Image and Bounding Box")
            ax[1].imshow(slice_3c)
            show_mask(medsam_seg, ax[1])
            show_box(bbox_original.squeeze(), ax[1])
            ax[1].set_title("MedSAM Segmentation")
            plt.tight_layout()
            plt.savefig(path_to_output_original)
            plt.close()
            # add output mask to ct mask
            foreground_coords = np.where(medsam_seg == 1)
            ct_mask[slice_idx][foreground_coords] = roi_counter
    return ct_mask


if __name__ == "__main__":
    windows = {
        "lung": {"L": -500, "W": 1400},
        "abdomen": {"L": 40, "W": 350},
        "bone": {"L": 400, "W": 1000},
        "air": {"L": -426, "W": 1000},
        "brain": {"L": 50, "W": 100}
    }
    parser = argparse.ArgumentParser(
        description="""Run MedSAM inference on a set of 3D ROIs for
        a particular CT volume.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_ct',
        type=str,
        help="Path to the CT volume in NIfTI format."
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
        help="Path to the model checkpoint."
    )
    parser.add_argument(
        '--path_to_rois',
        type=str,
        default='run_inference_on_rois.json',
        help="""Path to the JSON file containing a list with dictionaries,
        each dictionary corresponding to a 3D ROI, containing the following
        keys: 'series_orthanc_uuid', 'bbox', 'slices', 'H', 'W'."""
    )
    parser.add_argument(
        '--window',
        choices=list(windows.keys()),
        default=None,
        help="""Window for CT normalization. If None, values are clipped
        to percentiles 0.5 and 99.5, and then mapped to the range 0-255."""
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu'],
        default='gpu',
        help="Device to be used for inference."
    )
    args = parser.parse_args()
    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
    ct_image = sitk.ReadImage(args.path_to_ct)
    ct_array = sitk.GetArrayFromImage(ct_image)
    with open(args.path_to_rois, 'r') as file:
        rois = json.load(file)
    # normalize ct
    ct_array_pre = normalize_ct(ct_array, windows[args.window])
    # run inference
    ct_mask = run_inference(
        ct_array_pre,
        rois,
        args.path_to_checkpoint,
        args.path_to_output,
        device
    )
    # save output ct mask
    ct_mask = np.flip(ct_mask, axis=1)
    ct_mask_image = sitk.GetImageFromArray(ct_mask)
    ct_mask_image.CopyInformation(ct_image)
    sitk.WriteImage(
        ct_mask_image,
        Path(args.path_to_output) / f"{Path(args.path_to_ct).name.split('.nii.gz')[0]} mask.nii.gz"
    )
