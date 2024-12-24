import argparse
import numpy as np
import pandas as pd
import json
import SimpleITK as sitk
import concurrent.futures
from pathlib import Path
from skimage.measure import label as label_objects
from skimage.measure import regionprops
from skimage import io
from tqdm import tqdm
from pandas.errors import EmptyDataError


def get_3d_objects(path_to_performance, path_to_output_masks, path_to_series,
                   path_to_gt_masks, path_to_semantic_output,
                   path_to_instance_output, path_to_instance_gt, foreground_label=1):
    """Construct output volume masks from 2d results obtained using
    the 'evaluate_CT_dataset.py' script. Output volume masks are saved
    in nifti format.
    
    Parameters
    ----------
    path_to_performance : str
        Path to the 'performance.json' file.
    path_to_output_masks : str
        Path to the folder containing the slices with 2d output masks.
    path_to_series : str
        Path to the 'series.json' file containing the series metadata.
    path_to_gt_masks : str
        Path to the folder containing the ground truth annotated masks
        in nifti format.
    path_to_semantic_output : str
        Path to the directory to save output semantic masks (background vs foreground).
    path_to_instance_output : str
        Path to the directory to save output instance masks (one foreground label
        for each 3d object).
    path_to_instance_gt : str
        Path to the directory to save ground truth instance masks.
    foreground_label : int, optional
        Integer label for foreground objects in the output volume masks.
    """
    with open(path_to_series, 'r') as file:
        series_df = pd.DataFrame(json.load(file))
    with open(path_to_performance, 'r') as file:
        performance_df = pd.DataFrame(json.load(file)['bboxes'])
    unique_series = performance_df['study'].unique().tolist()
    for series_uuid in tqdm(unique_series):
        print(f"series name: {series_uuid}")
        series_performance_df = performance_df[performance_df['study'] == series_uuid]
        mask_shape = (
            series_df.loc[series_df['uuid'] == series_uuid, 'slices'].values[0],
            series_df.loc[series_df['uuid'] == series_uuid, 'rows'].values[0],
            series_df.loc[series_df['uuid'] == series_uuid, 'columns'].values[0]
        )
        output_mask = np.zeros(mask_shape).astype('bool')
        for _, row in series_performance_df.iterrows():
            path_to_slice = Path(path_to_output_masks) / f"{row['bbox_original_fname'].split('.npy')[0]}.png"
            slice_mask = io.imread(path_to_slice)
            slice_labeled = label_objects(slice_mask)
            objects = regionprops(slice_labeled)
            for object_ in objects:
                rows_coords = object_.coords[:, 0]
                cols_coords = object_.coords[:, 1]
                output_mask[
                    np.repeat([row['slice_idx']], len(rows_coords)),
                    rows_coords,
                    cols_coords
                ] = True
        gt_image_sitk = sitk.ReadImage(Path(path_to_gt_masks) / f"{row['study']}.nii.gz")
        mask_sitk = sitk.GetImageFromArray((output_mask * foreground_label).astype('uint8'))
        mask_sitk.CopyInformation(gt_image_sitk)
        sitk.WriteImage(
            mask_sitk,
            Path(path_to_semantic_output) / f"{series_uuid}.nii.gz"
        )
        instance_mask = label_objects(output_mask)
        instance_mask_image = sitk.GetImageFromArray(instance_mask)
        instance_mask_image.CopyInformation(gt_image_sitk)
        sitk.WriteImage(
            instance_mask_image,
            Path(path_to_instance_output) / f"{series_uuid}.nii.gz"
        )
        gt_mask = sitk.GetArrayFromImage(gt_image_sitk)
        gt_instance_mask = np.int32(label_objects(gt_mask))
        gt_instance_image = sitk.GetImageFromArray(gt_instance_mask)
        gt_instance_image.CopyInformation(gt_image_sitk)
        sitk.WriteImage(
            gt_instance_image,
            Path(path_to_instance_gt) / f"{series_uuid}.nii.gz"
        )


class Evaluator:
    def __init__(self, path_to_output, min_gt_overlap=0.5) -> None:
        self.path_to_output = path_to_output
        self.min_reference_overlap = min_gt_overlap

    def _compute_overlapping_ratio(self, prediction_object, gt_object, mask_shape):
        """Return |Prediction AND GT| / |GT|."""
        prediction_mask = np.zeros(mask_shape).astype('bool')
        prediction_mask[
            prediction_object.coords[:, 0],
            prediction_object.coords[:, 1],
            prediction_object.coords[:, 2],
        ] = True
        gt_mask = np.zeros(mask_shape).astype('bool')
        gt_mask[
            gt_object.coords[:, 0],
            gt_object.coords[:, 1],
            gt_object.coords[:, 2],
        ] = True
        overlapping_ratio = np.sum(np.logical_and(prediction_mask, gt_mask)) / np.sum(gt_mask)
        return overlapping_ratio

    def _compute_dice(self, prediction_object, gt_object, mask_shape):
        prediction_mask = np.zeros(mask_shape).astype('bool')
        prediction_mask[
            prediction_object.coords[:, 0],
            prediction_object.coords[:, 1],
            prediction_object.coords[:, 2],
        ] = True
        gt_mask = np.zeros(mask_shape).astype('bool')
        gt_mask[
            gt_object.coords[:, 0],
            gt_object.coords[:, 1],
            gt_object.coords[:, 2],
        ] = True
        dice = 2.0 * np.sum(np.logical_and(prediction_mask, gt_mask)) / (np.sum(prediction_mask) + np.sum(gt_mask))
        return dice

    def compute_single_3d_instance_performance(self, series):
        """Compute segmentation performance for a single series.

        Parameters
        ----------
        series : dict
            Dictionary with the following keys:
            - path_to_prediction
            - path_to_gt
            - path_to_gt_labels
        """
        print(f"filename: {Path(series['path_to_gt']).name}")
        gt_objects_info = []
        predicted_objects_info = []
        matches_info = []
        false_positive_info = []
        false_negative_info = []
        predicted_mask = sitk.GetArrayFromImage(sitk.ReadImage(series["path_to_prediction"]))
        gt_image = sitk.ReadImage(series["path_to_gt"])
        gt_mask = sitk.GetArrayFromImage(gt_image)
        spacing = (
            gt_image.GetSpacing()[-1],    # slice size
            gt_image.GetSpacing()[1],     # row size
            gt_image.GetSpacing()[0]      # column size
        )
        instance_predicted_mask = label_objects(predicted_mask)
        instance_gt_mask = label_objects(gt_mask)
        predicted_objects = regionprops(instance_predicted_mask, spacing=spacing)
        gt_objects = regionprops(instance_gt_mask, spacing=spacing)
        match_objects = []
        false_negative_objects = []
        available_predicted_objects = predicted_objects.copy()
        available_predicted_labels = [object_.label for object_ in available_predicted_objects]
        with open(series["path_to_gt_labels"], 'r') as file:
            gt_labels = json.load(file)
        for object_ in predicted_objects:
            predicted_objects_info.append({
                "series": Path(series["path_to_gt"]).name.split('.nii.gz')[0],
                "instance_label_value": object_.label,
                "size_voxels": object_.num_pixels,
                "volume_mm3": object_.area,
                "volume_ml": object_.area * 1e-3
            })
        for gt_object in gt_objects:
            # Find label description
            foreground_values = list(np.unique(gt_mask[
                gt_object.coords[:, 0],
                gt_object.coords[:, 1],
                gt_object.coords[:, 2]]
            ))
            if len(foreground_values) > 1:
                raise ValueError(f"object with more than 1 description label in {series['path_to_gt']}")
            label_description = gt_labels.get(str(foreground_values[0]))
            gt_objects_info.append({
                "series": Path(series["path_to_gt"]).name.split('.nii.gz')[0],
                "instance_label_value": gt_object.label,
                "instance_label_description": label_description,
                "size_voxels": gt_object.num_pixels,
                "volume_mm3": gt_object.area,
                "volume_ml": gt_object.area * 1e-3
            })
            # Find match
            intersection_values = list(np.unique(instance_predicted_mask[
                gt_object.coords[:, 0],
                gt_object.coords[:, 1],
                gt_object.coords[:, 2]]
            ))
            intersection_objects = [
                {
                    "object": predicted_object,
                    "overlapping_ratio": self._compute_overlapping_ratio(predicted_object, gt_object, gt_mask.shape)
                }
                for predicted_object in available_predicted_objects
                if predicted_object.label in intersection_values
            ]
            intersection_objects = sorted(
                intersection_objects,
                key=lambda x: x["overlapping_ratio"],
                reverse=True
            )
            if len(intersection_objects) > 0 and intersection_objects[0]["overlapping_ratio"] >= self.min_reference_overlap:
                match_objects.append({
                    "gt": gt_object,
                    "predicted": intersection_objects[0]["object"]
                })
                available_predicted_labels.remove(intersection_objects[0]["object"].label)
                available_predicted_objects = [
                    object_
                    for object_ in available_predicted_objects
                    if object_.label in available_predicted_labels
                ]
                matches_info.append({
                    "series": Path(series["path_to_gt"]).name.split('.nii.gz')[0],
                    "gt_instance_label_value": gt_object.label,
                    "gt_instance_label_description": label_description,
                    "gt_size_voxels": gt_object.num_pixels,
                    "gt_volume_mm3": gt_object.area,
                    "gt_volume_ml": gt_object.area * 1e-3,
                    "predicted_instance_label_value": intersection_objects[0]["object"].label,
                    "predicted_size_voxels": intersection_objects[0]["object"].num_pixels,
                    "predicted_volume_mm3": intersection_objects[0]["object"].area,
                    "predicted_volume_ml": intersection_objects[0]["object"].area * 1e-3,
                    "overlapping_ratio": intersection_objects[0]["overlapping_ratio"],
                    "dice": self._compute_dice(intersection_objects[0]["object"], gt_object, gt_mask.shape)
                })
            else:
                false_negative_objects.append(gt_object)
                false_negative_info.append({
                    "series": Path(series["path_to_gt"]).name.split('.nii.gz')[0],
                    "instance_label_value": gt_object.label,
                    "instance_label_description": label_description,
                    "size_voxels": gt_object.num_pixels,
                    "volume_mm3": gt_object.area,
                    "volume_ml": gt_object.area * 1e-3
                })
        for object_ in available_predicted_objects:
            intersection_values = list(np.unique(instance_gt_mask[
                object_.coords[:, 0],
                object_.coords[:, 1],
                object_.coords[:, 2]]
            ))
            intersection_objects = [
                {
                    "object": gt_object,
                    "overlapping_ratio": self._compute_overlapping_ratio(object_, gt_object, gt_mask.shape)
                }
                for gt_object in gt_objects
                if gt_object.label in intersection_values
            ]
            intersection_objects = sorted(
                intersection_objects,
                key=lambda x: x["overlapping_ratio"],
                reverse=True
            )
            overlapping_ratio = intersection_objects[0]["overlapping_ratio"] if intersection_objects else 0
            false_positive_info.append({
                "series": Path(series["path_to_gt"]).name.split('.nii.gz')[0],
                "instance_label_value": object_.label,
                "size_voxels": object_.num_pixels,
                "volume_mm3": object_.area,
                "volume_ml": object_.area * 1e-3,
                "overlapping_ratio": overlapping_ratio
            })
        # Save results
        pd.DataFrame(predicted_objects_info).to_csv(
            Path(self.path_to_output) / f"{Path(series['path_to_gt']).name.split('.nii.gz')[0]}_predicted_objects.csv",
            index=False
        )
        pd.DataFrame(gt_objects_info).to_csv(
            Path(self.path_to_output) / f"{Path(series['path_to_gt']).name.split('.nii.gz')[0]}_gt_objects.csv",
            index=False
        )
        pd.DataFrame(matches_info).to_csv(
            Path(self.path_to_output) / f"{Path(series['path_to_gt']).name.split('.nii.gz')[0]}_matches.csv",
            index=False
        )
        pd.DataFrame(false_positive_info).to_csv(
            Path(self.path_to_output) / f"{Path(series['path_to_gt']).name.split('.nii.gz')[0]}_false_positives.csv",
            index=False
        )
        pd.DataFrame(false_negative_info).to_csv(
            Path(self.path_to_output) / f"{Path(series['path_to_gt']).name.split('.nii.gz')[0]}_false_negatives.csv",
            index=False
        )


def aggregate_performance_results(path_to_results, path_to_output):
    prefixes = {
        "gt_objects": "gt_objects.csv",
        "predicted_objects": "predicted_objects.csv",
        "matches": "matches.csv",
        "false_negatives": "false_negatives.csv",
        "false_positives": "false_positives.csv"
    }
    for item, prefix in prefixes.items():
        output_df = pd.DataFrame()
        for path in Path(path_to_results).glob(f"*{prefix}"):
            try:
                df = pd.read_csv(path)
            except EmptyDataError:
                df = pd.DataFrame()
            output_df = pd.concat([output_df, df])
        output_df.to_csv(
            Path(path_to_output) / f"{item}.csv",
            index=False
        )


def compute_3d_performance(path_to_predicted_masks, path_to_gt_masks,
                           path_to_gt_labels, path_to_output, max_workers=4):
    # compute instance performance
    series = [
        {
            "path_to_gt": path,
            "path_to_prediction": Path(path_to_predicted_masks) / path.name,
            "path_to_gt_labels": Path(path_to_gt_labels) / f"{path.name.split('.nii.gz')[0]}.json"
        }
        for path in Path(path_to_gt_masks).glob('*.nii.gz')
    ]
    path_to_output_performance = Path(path_to_output) / 'performance'
    path_to_output_performance.mkdir(exist_ok=True)
    evaluator = Evaluator(
        path_to_output=path_to_output_performance,
        min_gt_overlap=0.5
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        _ = list(tqdm(executor.map(evaluator.compute_single_3d_instance_performance, series), total=len(series)))
    aggregate_performance_results(path_to_output_performance, path_to_output)
    # compute semantic performance
    semantic_performance = []
    for path_to_gt in tqdm(list(Path(path_to_gt_masks).glob('*.nii.gz'))):
        path_to_prediction = Path(path_to_predicted_masks) / path_to_gt.name
        prediction_mask = sitk.GetArrayFromImage(sitk.ReadImage(path_to_prediction)).astype('bool')
        gt_image = sitk.ReadImage(path_to_gt)
        gt_mask = sitk.GetArrayFromImage(gt_image).astype('bool')
        spacing = (
            gt_image.GetSpacing()[-1],    # slice size
            gt_image.GetSpacing()[1],     # row size
            gt_image.GetSpacing()[0]      # column size
        )
        dice = 2.0 * np.sum(np.logical_and(prediction_mask, gt_mask)) / (np.sum(prediction_mask) + np.sum(gt_mask))
        semantic_performance.append({
            "series": path_to_gt.name.split('.nii.gz')[0],
            "gt_voxels": np.sum(gt_mask),
            "gt_volume_mm3": np.sum(gt_mask) * spacing[0] * spacing[1] * spacing[2],
            "gt_volume_ml": np.sum(gt_mask) * spacing[0] * spacing[1] * spacing[2] * 1e-3,
            "prediction_voxels": np.sum(prediction_mask),
            "prediction_volume_mm3": np.sum(prediction_mask) * spacing[0] * spacing[1] * spacing[2],
            "prediction_volume_ml": np.sum(prediction_mask) * spacing[0] * spacing[1] * spacing[2] * 1e-3,
            "dice": dice
        })
    pd.DataFrame(semantic_performance).to_csv(
        Path(path_to_output) / "semantic_performance.csv",
        index=False
    )


def main():
    parser = argparse.ArgumentParser(
        description="""Get 3d objects from 2d objects and compute
        3d performance.

        2d objects are obtained using the script 'evaluate_CT_dataset.py'.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_performance_results',
        type=str,
        help="""Path to the performance.json file."""
    )
    parser.add_argument(
        'path_to_output_masks',
        type=str,
        help="""Path to the directory containing the output 2d
        slice masks in '.png' format."""
    )
    parser.add_argument(
        'path_to_gt_masks',
        type=str,
        help="""Path to the directory containing the ground truth
        volume masks in '.nii.gz' (nifti) format."""
    )
    parser.add_argument(
        'path_to_gt_labels',
        type=str,
        help="""Path to the directory containing the JSON files with
        labels of the gt volume masks."""
    )
    parser.add_argument(
        'path_to_series',
        type=str,
        help="Path to the series.json file containing metadata of series."
    )
    parser.add_argument(
        'path_to_output',
        type=str,
        help="""Path to the directory to save output results."""
    )
    args = parser.parse_args()
    path_to_semantic_masks = Path(args.path_to_output) / "output-semantic-masks"
    path_to_semantic_masks.mkdir(exist_ok=True)
    path_to_instance_masks = Path(args.path_to_output) / "output-instance-masks"
    path_to_instance_masks.mkdir(exist_ok=True)
    path_to_instance_gt_masks = Path(args.path_to_output) / "gt-instance-masks"
    path_to_instance_gt_masks.mkdir(exist_ok=True)
    get_3d_objects(
        args.path_to_performance_results,
        args.path_to_output_masks,
        args.path_to_series,
        args.path_to_gt_masks,
        path_to_semantic_masks,
        path_to_instance_masks,
        path_to_instance_gt_masks
    )
    compute_3d_performance(
        path_to_instance_masks,
        args.path_to_gt_masks,
        args.path_to_gt_labels,
        args.path_to_output
    )


if __name__ == "__main__":
    main()
