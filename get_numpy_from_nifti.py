import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import cc3d
import argparse
import json
from skimage import transform
from pathlib import Path
from tqdm import tqdm


WINDOWS = {
	        "lung": {"L": -500, "W": 1400},
	        "abdomen": {"L": 40, "W": 350},
	        "bone": {"L": 400, "W": 1000},
	        "air": {"L": -426, "W": 1000},
	        "brain": {"L": 50, "W": 100},
			"mediastinum": {"L": 50, "W": 350}
	     }


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255 / 255, 0 / 255, 0 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(np.uint8(mask_image))


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


def main(path_to_root, path_to_output, windows_mapping,
		 save_plots=False):
	path_to_imgs = Path(path_to_root) / 'imgs'
	path_to_gts = Path(path_to_root) / 'gts'
	path_to_output_imgs = Path(path_to_output) / 'imgs'
	path_to_output_gts = Path(path_to_output) / 'gts'
	path_to_output_plots = Path(path_to_output) / 'sanity-check'

	for path in (path_to_output_imgs, path_to_output_gts):
		path.mkdir(parents=True, exist_ok=True)
	if save_plots:
		path_to_output_plots.mkdir(parents=True, exist_ok=True)

	for path_to_img in tqdm(sorted(list(path_to_imgs.glob('*.nii.gz')))):
		print(f"CT filename: {path_to_img.name}")
		ct_array = sitk.GetArrayFromImage(sitk.ReadImage(path_to_img))
		mask_array = sitk.GetArrayFromImage(sitk.ReadImage(path_to_gts / path_to_img.name))
		# Make masks binary
		if 'MF_' in path_to_img.name:
			mask_array[mask_array>0] = 1
		else:
			if 'PETCT_13b40a817b' in path_to_img.name:
				mask_array[mask_array==1] = 0
				mask_array[mask_array>1] = 1
			if 'PETCT_15a205ffcc' in path_to_img.name:
				mask_array[mask_array==1] = 0
				mask_array[mask_array>1] = 1
			else:
				mask_array[mask_array>0] = 1
		# Check mask has annotations
		if (mask_array == 0).all():
			continue
		# Normalize CT
		window_name = windows_mapping.get(path_to_img.name)
		ct_array_pre = normalize_ct(ct_array, WINDOWS.get(window_name))
		# Get annotated slices
		annotated_slices = [
			slice_idx
			for slice_idx in range(mask_array.shape[0])
			if not (mask_array[slice_idx] == 0).all()
		]
		for slice_idx in tqdm(annotated_slices):
			mask_slice = cc3d.dust(
				mask_array[slice_idx],
				threshold=5,
				connectivity=8,
				in_place=True
			)
			if (mask_slice == 0).all():
				continue
			ct_slice = ct_array_pre[slice_idx]
			ct_slice_1024 = np.uint8(transform.resize(
				ct_slice,
				(1024, 1024),
				order=3,
				mode='constant',
				preserve_range=True,
				anti_aliasing=True
			))
			ct_slice_1024_3c = np.repeat(ct_slice_1024[:, :, None], 3, axis=-1)
			mask_slice_1024 = np.uint8(transform.resize(
				mask_slice,
				(1024, 1024),
				order=0,
				mode='constant',
				preserve_range=True,
				anti_aliasing=True
			))
			name = path_to_img.name.replace('.nii.gz', '')
			name2save_img = Path(path_to_output_imgs) / f'{slice_idx}_{name}.npy'
			name2save_seg = Path(path_to_output_gts) / f'{slice_idx}_{name}.npy'
			np.save(name2save_img, ct_slice_1024_3c)
			np.save(name2save_seg, mask_slice_1024)

			if save_plots:
				_, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
				ax[0].imshow(ct_slice_1024_3c)
				ax[1].imshow(ct_slice_1024_3c)
				show_mask(mask_slice_1024 * 255, ax[1])
				ax[0].set_title('image')
				ax[1].set_title('mask')
				plt.tight_layout()
				plt.savefig(path_to_output_plots / f"{slice_idx}_{name}.png")
				plt.close()


def get_windows_mapping(window_arg, path_to_cts):
	if window_arg not in WINDOWS:
		with open(window_arg, 'r') as file:
			mapping = json.load(file)
	else:
		mapping = {
			path.name: window_arg
			for path in Path(path_to_cts).glob('*.nii.gz')
		}
	return mapping


def check_windows_mapping(mapping, path_to_cts):
	# Check wrong windows
	wrong_windows = [
		f"filename '{filename}' with wrong window '{window}'."
		for filename, window in mapping.items()
		if window not in WINDOWS
	]
	if wrong_windows:
		raise ValueError('\n'.join(wrong_windows))
	# Check all CTs have their corresponding window
	unassigned_cts = [
		f"filename '{path.name}' does not have a window assigned."
		for path in Path(path_to_cts).glob('*.nii.gz')
		if path.name not in mapping.keys()
	]
	if unassigned_cts:
		raise ValueError('\n'.join(unassigned_cts))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Create numpy arrays for slices of annotated CT volumes",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument(
		'path_to_data',
		type=str,
		help = """Path to to the directory containing the CT images and
		masks saved as compressed nifti files (.nii.gz). CT images are saved
		in the 'imgs' folder, and CT masks in the 'gts' folder. Corresponding
		images and masks share the filename."""
	)
	parser.add_argument(
		'path_to_output',
		type=str,
		help="Path to the directory to save output numpy files."
	)
	parser.add_argument(
		'window',
		type=str,
		help=f"""Window for CT normalization: {list(WINDOWS.keys())}.
		This window is applied on all CTs. Alternatively, you can provide
		the path to a JSON file with a dictionary containing the
		mapping between filenames and windows."""
	)
	parser.add_argument(
		'--sanity-check',
		dest='sanity_check',
		action='store_true',
		help="""Add this flag to save plots of resulting images and masks
		for sanity check."""
	)
	args = parser.parse_args()
	windows_mapping = get_windows_mapping(
		args.window,
		Path(args.path_to_data) / 'imgs'
	)
	check_windows_mapping(
		windows_mapping,
		Path(args.path_to_data) / 'imgs'
	)
	main(
		args.path_to_data,
		args.path_to_output,
		windows_mapping,
		args.sanity_check
	)
