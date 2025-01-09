# Evaluate MedSAM on test set from HCUCH

This is done typically after finetuning MedSAM model with masks annotated by HITL procedure.

The evaluation process involves 3 steps:

1. Create output directories.
2. Convert format of checkpoint file.
3. Run the evaluation script.

## Create output directories

This step involves creating an output directory in the `results` directory. The foldername should be the same as the foldername containing the corresponding trained models.

You can create all directories at once running the following lines in the terminal:

```
find path/to/folder/with/models/folders/ -mindepth 1 -maxdepth 1 -type d | while read src; do folder_name=$(basename "$src") mkdir -p "path/to/output/folder/$folder_name" done
```

replacing `path/to/folder/with/models/folders/` and `path/to/output/folder` with the corresponding paths.

## Convert format of checkpoint file

Convert the original checkpoint file to the SAM checkpoint format. This is required for inference.

First, activate the environment:

```conda activate medsam```

Then, run the python script:

```python utils/ckpt_convert.py path/to/original/checkpoint```

replacing `path/to/original/checkpoint` for the corresponding checkpoint file (usually in `.pth` format).

You can convert all files at once running the following in the terminal:

```find path/to/folder/with/models/folders/ -mindepth 1 -maxdepth 1 -type d | while read src; do folder_name=$(basename "$src") python utils/ckpt_convert.py "path/to/output/folder/$folder_name/medsam_model_ft_best.pth" done```

replacing `path/to/folder/with/models/folders/` and `path/to/output/folder` with the corresponding paths.

## Run the evaluation script

Run the script `evaluate_CT_dataset.py`. The following block is an example. Please adapt the input parameters to your needs:

```python evaluate_CT_dataset.py /media/rrojas/data2/FONDEF_ID23I10337/data/hcuch-fondef/medsam-finetuning/iteration-3/test/nifti/imgs/ /media/rrojas/data2/FONDEF_ID23I10337/data/hcuch-fondef/medsam-finetuning/iteration-3/test/nifti/gts/ results/MedSAM-HITL-iteration-3/MedSAM-ViT-B-20241231-0300/ --path_to_checkpoint work_dir/MedSAM-HITL-iteration-3/MedSAM-ViT-B-20241231-0300/medsam_model_ft_best_converted.pth --window /media/rrojas/data2/FONDEF_ID23I10337/data/hcuch-fondef/medsam-finetuning/iteration-3/windows_mapping.json```  

