import numpy as np
import matplotlib.pyplot as plt
import os
import json

join = os.path.join
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
import pandas as pd
from pathlib import Path
from skimage.measure import label as label_objects
from skimage.measure import regionprops

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


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


def plot_validation_performance(validation_performance, path_to_output_file):
    performance_df = pd.DataFrame(validation_performance)
    mean_performance = performance_df[["epoch", "mean_dice_score"]].groupby("epoch").mean().reset_index().sort_values(by="epoch")
    plt.plot(
        mean_performance["epoch"],
        mean_performance["mean_dice_score"],
        'b--*'
    )
    plt.title("Validation performance")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Dice")
    plt.savefig(path_to_output_file)
    plt.close()


class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=5,
                 min_object_size_pixels=1):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift
        self.min_object_size_pixels = min_object_size_pixels

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        if np.max(img_1024) > 1.0:
            img_1024 = img_1024 / 255
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(
            gt == random.choice(label_ids.tolist())
        )  # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        # get bounding box for a random 2d object
        props = regionprops(label_objects(gt2D))
        props = [
            object_
            for object_ in props
            if object_.num_pixels >= self.min_object_size_pixels
        ]
        if len(props) == 0:
            raise ValueError(f"mask in filename {self.gt_path_files[index]} only has small objects (less than {self.min_object_size_pixels} pixels in size)")
        random_object = np.random.choice(props)
        x_min, x_max = random_object['bbox'][1], random_object['bbox'][3]
        y_min, y_max = random_object['bbox'][0], random_object['bbox'][2]
        gt2D = np.uint8(np.zeros(gt2D.shape))
        gt2D[random_object.coords[:, 0], random_object.coords[:, 1]] = 1
        # add random perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bbox = np.array([x_min, y_min, x_max, y_max])
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bbox).float(),
            img_name,
        )


class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


def main():
    # set up parser
    parser = argparse.ArgumentParser(
        description="""Train the image encoder and mask decoder (freeze
        prompt image encoder.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--tr_npy_path",
        type=str,
        default="data/npy/CT_Abd",
        help="Path to training npy files; two subfolders: gts and imgs"
    )
    parser.add_argument(
        "-v",
        "--val_npy_path",
        type=str,
        default=None,
        help="Path to validation npy files; two subfolders: gts and imgs"
    )
    parser.add_argument(
        "-task_name",
        type=str,
        default="MedSAM-ViT-B",
        help="Task name. Useful for identification."
    )
    parser.add_argument(
        "-model_type",
        type=str,
        default="vit_b",
        help="""MedSAM model type. See MedSAM architecture and repository
        for more details."""
    )
    parser.add_argument(
        "-checkpoint",
        type=str,
        default="work_dir/MedSAM/medsam_vit_b.pth",
        help="Path to the checkpoint pytorch model."
    )
    parser.add_argument(
        "-work_dir",
        type=str,
        default="./work_dir",
        help="""Path to the directory to save output results. A new
        folder is created inside."""
    )
    parser.add_argument(
        "-num_epochs",
        type=int,
        default=1000,
        help="Epochs to train the model."
    )
    parser.add_argument(
        "-batch_size",
        type=int,
        default=2,
        help="Batch size for training."
    )
    parser.add_argument(
        "-val_batch_size",
        type=int,
        default=1,
        help="Batch size for validation."
    )
    parser.add_argument(
        "-num_workers",
        type=int,
        default=0,
        help="Number of workers."
    )
    parser.add_argument(
        "-weight_decay",
        type=float,
        default=0.01,
        help="""Weight decay. Regularization technique to penalize
        large model weights."""
    )
    parser.add_argument(
        "-lr",
        type=float,
        default=0.0001,
        metavar="LR",
        help="Learning rate (absolute lr)."
    )
    parser.add_argument(
        "-use_wandb",
        type=bool,
        default=False,
        help="Use wandb to monitor training."
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=False,
        help="Add this flag to use amp."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="""Path to the checkpoint to resume training.
        The checkpoint contains the epoch to start from, and the
        model and optimizer state dicts."""
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="""Device for processing. Use 'cpu' to use CPU,
        but using GPU is highly recommended."""
    )
    parser.add_argument(
        "--val_every",
        type=int,
        default=10,
        help="Validate every specified number of epochs."
    )
    parser.add_argument(
        "--plot_val",
        action="store_true",
        help="""Add this flag to plot segmentations obtained for the
        validation set."""
    )
    parser.add_argument(
        '--min_object_size',
        type=int,
        default=1,
        help="""Minimum size in pixels of 2d objects to be
        considered for training and evaluation."""
    )
    args = parser.parse_args()
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    path_to_output_folder = join(args.work_dir, args.task_name + "-" + run_id)
    os.makedirs(path_to_output_folder, exist_ok=True)

    # sanity test of dataset class
    tr_dataloader = DataLoader(
        NpyDataset(
            args.tr_npy_path,
            min_object_size_pixels=args.min_object_size
        ),
        batch_size=8,
        shuffle=True
    )
    for step, (image, gt, bboxes, names_temp) in enumerate(tr_dataloader):
        print(image.shape, gt.shape, bboxes.shape)
        # show the example
        _, axs = plt.subplots(1, 2, figsize=(25, 25))
        idx = random.randint(0, 7)
        axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
        show_mask(gt[idx].cpu().numpy(), axs[0])
        show_box(bboxes[idx].numpy(), axs[0])
        axs[0].axis("off")
        # set title
        axs[0].set_title(names_temp[idx])
        idx = random.randint(0, 7)
        axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
        show_mask(gt[idx].cpu().numpy(), axs[1])
        show_box(bboxes[idx].numpy(), axs[1])
        axs[1].axis("off")
        # set title
        axs[1].set_title(names_temp[idx])
        # plt.show()
        plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.savefig(
            Path(path_to_output_folder) / "input_data_sanitycheck.png",
            bbox_inches="tight",
            dpi=300
        )
        plt.close()
        break

    if args.use_wandb:
        import wandb

        wandb.login()
        wandb.init(
            project=args.task_name,
            config={
                "lr": args.lr,
                "batch_size": args.batch_size,
                "data_path": args.tr_npy_path,
                "model_type": args.model_type,
            },
        )

    # set up model for training
    device = torch.device(args.device)
    shutil.copyfile(
        __file__, join(path_to_output_folder, run_id + "_" + os.path.basename(__file__))
    )

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    medsam_model.train()

    encoder_decoder_params = list(medsam_model.image_encoder.parameters()) + list(medsam_model.mask_decoder.parameters())
    optimizer = torch.optim.Adam(encoder_decoder_params, lr=args.lr, weight_decay=0)

    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # train
    num_epochs = args.num_epochs
    losses = []
    best_loss = 1e10
    best_val_performance = -1e10
    datasets = {
        "train": NpyDataset(
            args.tr_npy_path,
            min_object_size_pixels=args.min_object_size
        )
    }
    print("Number of training samples: ", len(datasets["train"]))
    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    }
    if args.val_npy_path:
        datasets.update({
            "val": NpyDataset(
                args.val_npy_path,
                min_object_size_pixels=args.min_object_size
            )
        })
        dataloaders.update({
            "val": DataLoader(
                datasets["val"],
                batch_size=args.val_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        })
        validation_performance = []
    start_epoch = 0
    validation_epoch = start_epoch + args.val_every - 1
    if args.resume is not None:
        if os.path.isfile(args.resume):
            # Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(dataloaders["train"])):
            boxes_np = boxes.numpy()
            sam_trans = ResizeLongestSide(medsam_model.image_encoder.img_size)
            box = sam_trans.apply_boxes(boxes_np, (gt2D.shape[-2], gt2D.shape[-1]))
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            image, gt2D = image.to(device), gt2D.to(device)

            image_embeddings = medsam_model.image_encoder(image)

            ## do not compute gradients for image encoder and prompt encoder
            #with torch.no_grad():
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)

            # get prompt embeddings
            sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None,
                )
            # predicted masks
            mask_predictions, _ = medsam_model.mask_decoder(
                image_embeddings=image_embeddings, # (B, 256, 64, 64)
                image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=False,
              )
            mask_predictions_1024 = F.interpolate(
                mask_predictions,
                size=(image.shape[2], image.shape[3]),
                mode="bilinear",
                align_corners=False,
            )

            loss = seg_loss(mask_predictions_1024, gt2D)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        is_validation_epoch = True if epoch == validation_epoch else False

        if args.val_npy_path and is_validation_epoch:
            medsam_model.eval()
            epoch_val_performance = 0
            if args.plot_val:
                path_to_val_folder = Path(path_to_output_folder) / f"validation-epoch-{validation_epoch}"
                path_to_val_folder.mkdir()
            with torch.no_grad():
                for batch_idx, (image, gt2D, boxes, _) in enumerate(tqdm(dataloaders["val"])):
                    boxes_np = boxes.numpy()
                    sam_trans = ResizeLongestSide(medsam_model.image_encoder.img_size)
                    box = sam_trans.apply_boxes(boxes_np, (gt2D.shape[-2], gt2D.shape[-1]))
                    box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                    image, gt2D = image.to(device), gt2D.to(device)

                    image_embeddings = medsam_model.image_encoder(image)

                    if len(box_torch.shape) == 2:
                        box_torch = box_torch[:, None, :] # (B, 1, 4)

                    # get prompt embeddings
                    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
                        points=None,
                        boxes=box_torch,
                        masks=None,
                    )
                    # predicted masks
                    mask_predictions, _ = medsam_model.mask_decoder(
                        image_embeddings=image_embeddings, # (B, 256, 64, 64)
                        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                        multimask_output=False,
                    )

                    mask_predictions_1024 = F.interpolate(
                        mask_predictions,
                        size=(image.shape[2], image.shape[3]),
                        mode="bilinear",
                        align_corners=False
                    )

                    loss = seg_loss(mask_predictions_1024, gt2D)
                    mask_probs_1024 = torch.sigmoid(mask_predictions_1024)
                    mask_binary_1024 = (mask_probs_1024 > 0.5).long()
                    dice_scores = monai.metrics.meandice.compute_dice(
                        y_pred=mask_binary_1024,
                        y=gt2D,
                        include_background=False
                    )
                    dice_batch_mean = dice_scores.mean().item()
                    print(f"validation dice score (batch mean): {dice_batch_mean}")
                    validation_performance.append({
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "loss": loss.item(),
                        "mean_dice_score": dice_batch_mean
                    })
                    epoch_val_performance += dice_batch_mean
                    # Plot output segmentation for a random sample from the batch
                    if args.plot_val:
                        _, ax = plt.subplots(1, 3, figsize=(15, 5))
                        random_idx = np.random.choice(range(image.shape[0]))
                        ax[0].imshow(image[random_idx].cpu().permute(1, 2, 0).numpy())
                        ax[0].set_title("Input Image")
                        ax[1].imshow(image[random_idx].cpu().permute(1, 2, 0).numpy())
                        show_mask(gt2D[random_idx].cpu().numpy(), ax[1])
                        show_box(boxes[random_idx].numpy(), ax[1])
                        ax[1].set_title("Ground truth")
                        ax[2].imshow(image[random_idx].cpu().permute(1, 2, 0).numpy())
                        show_mask(mask_binary_1024[random_idx].cpu().numpy(), ax[2])
                        show_box(boxes[random_idx].numpy(), ax[2])
                        ax[2].set_title(f"MedSAM Segmentation (Dice={round(dice_scores[random_idx].item(), 3)})")
                        plt.tight_layout()
                        plt.savefig(Path(path_to_val_folder) / f"epoch_{epoch}_batch_{batch_idx}_sample_{random_idx}_val.png")
                        plt.close()
            with open(Path(path_to_output_folder) / "validation_performance.json", 'w') as file:
                json.dump(validation_performance, file, indent=4)
            epoch_val_performance /= (batch_idx + 1)
            validation_epoch += args.val_every
            medsam_model.train()

        epoch_loss /= step
        losses.append(epoch_loss)
        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        # save the latest model
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(
            checkpoint,
            join(path_to_output_folder, "medsam_model_ft_latest.pth")
        )
        # save the best model
        if args.val_npy_path and is_validation_epoch:
            if epoch_val_performance > best_val_performance:
                best_val_performance = epoch_val_performance
                checkpoint = {
                    "model": medsam_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_performance": best_val_performance
                }
                torch.save(
                    checkpoint,
                    join(path_to_output_folder, "medsam_model_ft_best.pth")
                )
        elif not args.val_npy_path:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                checkpoint = {
                    "model": medsam_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": best_loss
                }
                torch.save(
                    checkpoint,
                    join(path_to_output_folder, "medsam_model_ft_best.pth")
                )
        # plot training loss
        plt.plot(losses)
        plt.title("Training loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(path_to_output_folder, args.task_name + "train_loss.png"))
        plt.close()

        # plot validation performance
        if args.val_npy_path and validation_performance:
            plot_validation_performance(
                validation_performance,
                join(path_to_output_folder, args.task_name + "val_performance.png")
            )

    with open(Path(path_to_output_folder) / "arguments.json", 'w') as file:
        json.dump(vars(args), file, indent=4)


if __name__ == "__main__":
    main()
