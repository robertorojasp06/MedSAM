import json
import numpy as np
import SimpleITK as sitk
import argparse
import pandas as pd
import concurrent.futures
from skimage.measure import label as label_objects
from skimage.measure import regionprops
from pathlib import Path
from tqdm import tqdm


class Extractor:
    def __init__(self):
        self.max_workers = 4
        self.extension = '.nii.gz'
        self.verbose = True

    def _extract_objects_from_slice(self, slice_idx):
        label_image = label_objects(self._mask_volume[slice_idx])
        objects = regionprops(
            label_image,
            spacing=(self._spacing["row"], self._spacing["column"])
        )
        objects_data = []
        for object_ in objects:
            foreground_voxels = self._mask_volume[
                slice_idx,
                object_.coords[:, 0],
                object_.coords[:, 1]
            ]
            foreground_values = [
                value
                for value in np.unique(foreground_voxels)
                if value > 0
            ]
            lesion_label = self._labels.get(str(foreground_values[0]), None)
            objects_data.append({
                "filename": self._filename,
                "slice_idx": slice_idx,
                "lesion_instance_foreground_value": foreground_values[0],
                "lesion_type": lesion_label.split(',')[0].strip(),
                "lesion_location": lesion_label.split(',')[1].strip(),
                "pixels_count": object_.num_pixels,
                "area_mm2": object_.area,
                "centroid_row": object_.centroid[0],
                "centroid_col": object_.centroid[1],
                "bbox_min_row": object_.bbox[0],
                "bbox_min_col": object_.bbox[1],
                "bbox_max_row": object_.bbox[2],
                "bbox_max_col": object_.bbox[3],
                "axis_major_length_mm": object_.axis_major_length,
                "axis_minor_length_mm": object_.axis_minor_length
            })
            if len(foreground_values) > 1:
                print(f"More than one label type for object: {objects_data[-1]}")
        return objects_data

    def extract_objects_from_volume(self, volume):
        """'volume' is a dictionary containing the following keys:
            "path_to_mask": str,
            "path_to_label": str
        """
        path_to_mask = volume["path_to_mask"]
        path_to_label = volume["path_to_label"]
        objects = []
        image = sitk.ReadImage(str(path_to_mask))
        spacing = image.GetSpacing()
        mask = sitk.GetArrayFromImage(image)
        with open(path_to_label, 'r') as file:
            labels = json.load(file)
        # Get objects for each slice
        self._mask_volume = mask
        self._filename = Path(path_to_mask).parts[-1]
        self._labels = labels
        self._spacing = {
            "row": spacing[1],
            "column": spacing[0]
        }
        positive_slices = [
            idx
            for idx, slice_ in enumerate(mask)
            if np.any(slice_.astype('bool'))
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            slice_objects = executor.map(self._extract_objects_from_slice, positive_slices)
        objects = [
            object_
            for items in slice_objects
            for object_ in items
        ]
        return objects

    def extract_objects(self, path_to_masks, path_to_labels):
        paths_to_masks = sorted(list(Path(path_to_masks).glob(f"*{self.extension}")))
        paths_to_labels = sorted(list(Path(path_to_labels).glob(f"*.json")))
        final_objects = []
        for path_to_mask, path_to_label in tqdm(zip(paths_to_masks, paths_to_labels), total=len(paths_to_masks)):
            if self.verbose:
                tqdm.write(f"filename: {path_to_mask.parts[-1]}")
            volume_objects = self.extract_objects_from_volume({
                "path_to_mask": path_to_mask,
                "path_to_label": path_to_label
            })
            final_objects.extend(volume_objects)
        return final_objects


def main():
    parser = argparse.ArgumentParser(
        description="""Extract data from 2d objects identified
        as connected components from annotated volumes.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_masks',
        type=str,
        help="""Path to the directory containing the annotated
        volumes saved as nifti files."""
    )
    parser.add_argument(
        'path_to_labels',
        type=str,
        help="""Path to the directory containing the JSON
        files with the labels for annotated masks. Must share
        the same filename as the files of the annotated masks."""
    )
    parser.add_argument(
        '--path_to_output',
        type=str,
        default=Path.cwd(),
        help="Path to the output directory."
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=8,
        help="Max threads for multiprocessing."
    )
    args = parser.parse_args()
    extractor = Extractor()
    extractor.max_workers = args.max_workers
    objects = extractor.extract_objects(
        args.path_to_masks,
        args.path_to_labels
    )
    pd.DataFrame(objects).to_csv(
        Path(args.path_to_output) / "objects_2d.csv",
        index=False
    )


if __name__ == "__main__":
    main()