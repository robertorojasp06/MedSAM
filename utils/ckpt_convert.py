# -*- coding: utf-8 -*-
import torch
import argparse
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Convert medsam model checkpoint to sam checkpoint
        format for convenient inference.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_medsam_ckpt',
        type=str,
        help="""Path to the input MedSAM model checkpoint. Output
        model is saved next to the input model checkpoint and with the
        filename suffix '_converted'."""
    )
    parser.add_argument(
        '--path_to_sam_ckpt',
        type=str,
        default=Path.cwd() / 'work_dir' / 'SAM' / 'sam_vit_b_01ec64.pth',
        help="Path to the SAM model checkpoint."
    )
    parser.add_argument(
        '--multi_gpu',
        dest='multi_gpu_ckpt',
        action='store_true',
        help="Add this flag if the model is trained with multi-gpu."
    )
    args = parser.parse_args()
    sam_ckpt = torch.load(args.path_to_sam_ckpt)
    medsam_ckpt = torch.load(args.path_to_medsam_ckpt)
    sam_keys = sam_ckpt.keys()
    for key in sam_keys:
        if not args.multi_gpu_ckpt:
            sam_ckpt[key] = medsam_ckpt["model"][key]
        else:
            sam_ckpt[key] = medsam_ckpt["model"]["module." + key]
    path_to_output = (
        Path(args.path_to_medsam_ckpt).parent /
        f"{Path(args.path_to_medsam_ckpt).stem}_converted.pth"
    )
    torch.save(sam_ckpt, path_to_output)
