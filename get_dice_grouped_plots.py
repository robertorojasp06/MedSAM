import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io
from tqdm import tqdm
from matplotlib_scalebar.scalebar import ScaleBar


DICE_LIMITS = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
MASK_COLOR = np.array([145 / 255, 200 / 255, 250 / 255, 0.6])


def show_mask(mask, ax, color=np.array([251 / 255, 252 / 255, 30 / 255, 0.6])):
    if color is False or color is None:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


def extract_centered_patch(image, bbox, patch_size=(128, 128)):
    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate center of the bounding box
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    # Calculate half patch size
    half_patch_width = patch_size[0] // 2
    half_patch_height = patch_size[1] // 2
    
    # Calculate patch boundaries
    patch_x_min = max(0, center_x - half_patch_width)
    patch_x_max = min(image.shape[1], center_x + half_patch_width)
    patch_y_min = max(0, center_y - half_patch_height)
    patch_y_max = min(image.shape[0], center_y + half_patch_height)
    
    # Calculate margin offsets
    margin_left = max(0, half_patch_width - center_x)
    margin_right = max(0, center_x + half_patch_width - image.shape[1])
    margin_top = max(0, half_patch_height - center_y)
    margin_bottom = max(0, center_y + half_patch_height - image.shape[0])
    
    # Extract the patch, considering margins
    patch = image[patch_y_min:patch_y_max, patch_x_min:patch_x_max, ...]
    if image.ndim == 3:
        patch = np.pad(
            patch,
            ((margin_top, margin_bottom), (margin_left, margin_right), (0, 0)),
            mode='constant'
        )
    else:
        patch = np.pad(
            patch,
            ((margin_top, margin_bottom), (margin_left, margin_right)),
            mode='constant'
        )    
    return patch


def main(path_to_performance, path_to_slices,
         path_to_output_masks, path_to_gt_masks,
         path_to_labels, path_to_output, series_list,
         scalebar):
    with open(path_to_performance, 'r') as file:
        performance = json.load(file)
    performance_df = pd.DataFrame(performance['bboxes'])
    for idx in range(len(DICE_LIMITS) - 1):
        print(f"boxes with dice score in the interval [{DICE_LIMITS[idx]}, {DICE_LIMITS[idx + 1]})")
        filtered_df = performance_df[
            (performance_df['dice_score'] >= DICE_LIMITS[idx]) &
            (performance_df['dice_score'] < DICE_LIMITS[idx + 1])
        ]
        path_to_output_slices = (
            Path(path_to_output) /
            f"{DICE_LIMITS[idx]}_to_{DICE_LIMITS[idx + 1]}" /
            "slices"
        )
        path_to_output_painted_slices = (
            Path(path_to_output) /
            f"{DICE_LIMITS[idx]}_to_{DICE_LIMITS[idx + 1]}" /
            "painted-slices"
        )
        path_to_output_image_mask_patches = (
            Path(path_to_output) /
            f"{DICE_LIMITS[idx]}_to_{DICE_LIMITS[idx + 1]}" /
            "image-mask-patches"
        )
        path_to_output_image_patches = (
            Path(path_to_output) /
            f"{DICE_LIMITS[idx]}_to_{DICE_LIMITS[idx + 1]}" /
            "image-patches"
        )
        path_to_output_mask_patches = (
            Path(path_to_output) /
            f"{DICE_LIMITS[idx]}_to_{DICE_LIMITS[idx + 1]}" /
            "mask-patches"
        )
        for path in (
            path_to_output_slices,
            path_to_output_painted_slices,
            path_to_output_image_mask_patches,
            path_to_output_image_patches,
            path_to_output_mask_patches):
            path.mkdir(parents=True, exist_ok=True)
        for _, row in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0]):
            path_to_slice = (
                Path(path_to_slices) /
                f"{row['ct_fname'].split('.nii.gz')[0]}_slice{row['slice_idx']}.png"
            )
            path_to_output_mask = (
                Path(path_to_output_masks) /
                f"{row['ct_fname'].split('.nii.gz')[0]}_sliceidx{row['slice_idx']}_foregroundlabel{row['foreground_label']}_{row['bbox_original'][0]}_{row['bbox_original'][1]}_{row['bbox_original'][2]}_{row['bbox_original'][3]}.png"
            )
            path_to_gt_mask = (
                Path(path_to_gt_masks) /
                path_to_output_mask.name
            )
            path_to_json = (
                Path(path_to_labels) /
                f"{row['ct_fname'].split('.nii.gz')[0]}.json"
            )
            slice_array = io.imread(path_to_slice)
            output_mask = io.imread(path_to_output_mask)
            gt_mask = io.imread(path_to_gt_mask)
            patch_image = extract_centered_patch(
                slice_array,
                np.array(row['bbox_original'])
            )
            patch_output_mask = extract_centered_patch(
                output_mask,
                np.array(row['bbox_original'])
            )
            patch_gt_mask = extract_centered_patch(
                gt_mask,
                np.array(row['bbox_original'])
            )
            with open(path_to_json, 'r') as file:
                labels = json.load(file)
            path_to_output_slice = (
                Path(path_to_output_slices) /
                f"{Path(path_to_slice).name}"
            )
            path_to_output_painted_slice = (
                Path(path_to_output_painted_slices) /
                f"{Path(path_to_output_mask).name}"
            )
            path_to_output_image_mask_patch = (
                Path(path_to_output_image_mask_patches) /
                f"{Path(path_to_output_mask).name}"
            )
            path_to_output_image_patch = (
                Path(path_to_output_image_patches) /
                f"{Path(path_to_output_mask).name}"
            )
            path_to_output_mask_patch = (
                Path(path_to_output_mask_patches) /
                f"{Path(path_to_output_mask).name}"
            )
            series_uuid = Path(path_to_slice).name.split('_')[0]
            row_size = [
                series_["row_spacing"]
                for series_ in series_list
                if series_["uuid"] == series_uuid

            ]
            # Original slice
            _, ax = plt.subplots()
            ax.imshow(slice_array)
            ax.set_title("slice", fontsize=16, fontweight='bold')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(path_to_output_slice, transparent=True)
            plt.close()
            # Slice with overlapped masks (gt and prediction)
            _, ax = plt.subplots()
            ax.imshow(slice_array)
            show_mask((gt_mask > 0) * 1, ax)
            show_mask((output_mask > 0) * 1, ax, color=MASK_COLOR)
            ax.set_title(f"{labels.get(str(row['foreground_label']))} - dice score: {np.round(row['dice_score'], 3)}")
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(path_to_output_painted_slice)
            plt.close()
            # Patch and Patch with overlapped masks
            _, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(patch_image)
            ax[1].imshow(patch_image)
            show_mask((patch_gt_mask > 0) * 1, ax[1])
            show_mask((patch_output_mask > 0) * 1, ax[1], color=MASK_COLOR)
            ax[0].set_title("image")
            ax[1].set_title(f"{labels.get(str(row['foreground_label']))} - dice score: {np.round(row['dice_score'], 3)}")
            ax[0].axis('off')
            ax[1].axis('off')
            scalebar = ScaleBar(
                row_size[0],
                'mm',
                location='lower right',
                length_fraction=0.2
            )
            ax[0].add_artist(scalebar)
            plt.tight_layout()
            plt.savefig(path_to_output_image_mask_patch)
            plt.close()
            # Patch
            _, ax = plt.subplots()
            ax.imshow(patch_image)
            ax.set_title("image", fontsize=16, fontweight='bold')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(path_to_output_image_patch, transparent=True)
            plt.close()
            # Patch with overlapped masks
            _, ax = plt.subplots()
            ax.imshow(patch_image)
            show_mask((patch_gt_mask > 0) * 1, ax)
            show_mask((patch_output_mask > 0) * 1, ax, color=MASK_COLOR)
            ax.set_title(
                f"{labels.get(str(row['foreground_label']))} - dice score: {np.round(row['dice_score'], 3)}",
                fontsize=16,
                fontweight='bold'
            )
            ax.axis('off')
            scalebar = ScaleBar(
                row_size[0],
                'mm',
                location='lower right',
                length_fraction=0.2
            )
            ax.add_artist(scalebar)
            plt.tight_layout()
            plt.savefig(path_to_output_mask_patch, transparent=True)
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Plot and save ROIs of outputs obtained by using
        the 'evaluate_CT_dataset.py', grouped according to the Dice score.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_results',
        type=str,
        help="""Path to the directory containing the outputs obtained from the
        evaluate_CT_dataset.py script. It must have the 'performance.json' file
        and the following subbirectories:'original_size', 'output_masks',
        'gt_masks'."""
    )
    parser.add_argument(
        'path_to_labels',
        type=str,
        help="""Path to the directory containing the JSON files
        with the label mapping for each CT volume/mask. This folder
        is obtained by using the 'get_numpy_from_nifti.py' script."""
    )
    parser.add_argument(
        '--path_to_output',
        type=str,
        default=None,
        help="""Path to the directory to save the resulting plots.
        If None, 'path_to_results' is used."""
    )
    parser.add_argument(
        '--path_to_series',
        type=str,
        default=None,
        help="""Path to the series.json file containing the metadata for
        each series. This file is used to retrieve the pixel size for
        scalebars. It assume the filenames in 'original_size' slices to
        contain the "uuid" of the series before the first underscore.
        If None, no scalebar is added."""
    )
    parser.add_argument(
        '--scalebar',
        type=float,
        default=5.0,
        help="Length of scalebar in centimeters."
    )
    args = parser.parse_args()
    path_to_performance = Path(args.path_to_results) / "performance.json"
    path_to_slices = Path(args.path_to_results) / "original_size"
    path_to_output_masks = Path(args.path_to_results) / "output_masks"
    path_to_gt_masks = Path(args.path_to_results) / "gt_masks"
    if not args.path_to_output:
        path_to_output = Path(args.path_to_results) / "dice-grouped-segmentations"
    else:
        path_to_output = Path(args.path_to_output) / "dice-grouped-segmentations"
    if args.path_to_series:
        with open(args.path_to_series, 'r') as file:
            series = json.load(file)
    else:
        series = None
    main(
        path_to_performance,
        path_to_slices,
        path_to_output_masks,
        path_to_gt_masks,
        args.path_to_labels,
        path_to_output,
        series,
        args.scalebar
    )
