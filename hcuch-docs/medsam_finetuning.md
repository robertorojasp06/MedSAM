# Finetune MedSAM using local HCUCH data

To do this, we first need the local data saved as NIfTI files. We should have 3 folders:

- **imgs:** NIfTI files with the CT volume data.
- **gts:** NIfTI files with the segmentation masks.
- **labels:** JSON files containing the mapping between voxel values and label names (lesion type, localization) in the segmentation masks.

Corresponding images, masks and labels are supposed to have the same filename.

The finetuning process involves 3 steps:

1. Get numpy data from original NIfTI volumes.
2. Split data for cross validation.
3. Run the finetuning script.

## Get numpy data from original NIfTI volumes.
First, activate the environment:

```conda activate medsam```

Then, run the script separately for the train and test data:

```python get_numpy_from_nifti.py path_to_data path_to_output path_to_windows_mapping --sanity-check```

## Split data for cross validation
This is done only for the train data, over the numpy files generated in the previous step. Run:

```python split_numpy_for_crossvalidation.py path_to_numpy path_to_output path_to_series```

## Run the finetuning script

```python finetune.py -i path_to_train_npy -v path_to_val_npy -work_dir path_to_output_directory -num_epochs epochs --val_every n_samples```

Use the `--help` option to see other parameters to be set. 





