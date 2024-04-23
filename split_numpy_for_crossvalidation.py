import json
import shutil
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm


class Splitter:
    def __init__(self, path_to_numpy, series_metadata) -> None:
        self.path_to_npy = path_to_numpy
        self.series_metadata = series_metadata

    @property
    def series_metadata_df(self):
        return pd.DataFrame(self.series_metadata)

    def _get_samples(self):
        """Returns the metadata as a dataframe containing only
        the samples inside the input numpy folder. Assumes filenames
        follow the convention '{slice_idx}_{series_uuid}.npy'."""
        npy_samples_df = pd.DataFrame([
            {
                "filename": path.name,
                "series_uuid": path.stem.split('_')[-1],
                "slice_idx": path.stem.split('_')[0],
                "patient_code": self.series_metadata_df.loc[self.series_metadata_df['uuid'] == path.stem.split('_')[-1], 'patient_code'].values[0]
            }
            for path in (Path(self.path_to_npy) / 'imgs').glob('*.npy')
        ])
        return npy_samples_df

    def split_leave_one_out(self):
        npy_samples_df = self._get_samples().reset_index(drop=True)
        loo_crossvalidator = LeaveOneGroupOut()
        print(f"splits: {loo_crossvalidator.get_n_splits(groups=npy_samples_df['patient_code'].to_list())}")
        inds = loo_crossvalidator.split(
            npy_samples_df,
            groups=npy_samples_df['patient_code'].to_list()
        )
        splitted_npy_samples_df = pd.DataFrame()
        for idx, (training_inds, _) in enumerate(inds):
            iteration_df = npy_samples_df.copy()
            iteration_df['cv_iteration'] = idx
            iteration_df['subset'] = npy_samples_df.apply(
                lambda row: 'train' if row.name in training_inds else 'validation',
                axis=1
            )
            splitted_npy_samples_df = pd.concat([splitted_npy_samples_df, iteration_df])
        return splitted_npy_samples_df

def copy_image_and_mask(row, path_to_npy, path_to_output):
    path_to_output_img = (
        Path(path_to_output) /
        f"iteration-{row['cv_iteration']}" /
        row["subset"] /
        "imgs" /
        row["filename"]
    )
    path_to_output_mask = (
        Path(path_to_output) /
        f"iteration-{row['cv_iteration']}" /
        row["subset"] /
        "gts" /
        row["filename"]
    )
    Path(path_to_output_img).parent.mkdir(exist_ok=True, parents=True)
    Path(path_to_output_mask).parent.mkdir(exist_ok=True, parents=True)
    shutil.copy(
        Path(path_to_npy) / 'imgs' / row["filename"],
        path_to_output_img
    )
    shutil.copy(
        Path(path_to_npy) / 'gts' / row["filename"],
        path_to_output_mask
    )


def main():
    parser = argparse.ArgumentParser(
        description="""Split a dataset with images and annotated masks using Leave One
        Patient Out cross validation.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_npy',
        type=str,
        help="""Path to the directory containing 2 subfolders: 'imgs'
        and 'gts'. Slice images are contained in the 'imgs' folder, and
        annotated masks are contained in the 'gts' folder."""
    )
    parser.add_argument(
        'path_to_output',
        type=str,
        help="Path to the directory to save output data."
    )
    parser.add_argument(
        'path_to_series',
        type=str,
        help="""Path to the series.json file containing metadata about the
        series."""
    )
    args = parser.parse_args()
    with open(args.path_to_series, 'r') as file:
        series = json.load(file)
    splitter = Splitter(args.path_to_npy, series)
    splits_df = splitter.split_leave_one_out()
    tqdm.pandas()
    splits_df.progress_apply(
        lambda row: copy_image_and_mask(row, args.path_to_npy, args.path_to_output),
        axis=1
    )
    splits_df.to_csv(
        Path(args.path_to_output) / 'splits.csv',
        index=False
    )


if __name__ == "__main__":
    main()
