"""Sequential category training for VarNet.

Trains the VarNet model on one anatomical category at a time using
`CustomCategoryDataModule`. Each category is trained for a fixed number
of epochs (default: 5) before moving to the next category, continuing
from the same model weights.

This does NOT modify the underlying dataloader; it only changes the
training schedule.
"""

from argparse import ArgumentParser
from pathlib import Path
import os
import pathlib
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.pl_modules import VarNetModule

from dataloaders.custom_category_data_module_clean import CustomCategoryDataModule

import wandb
import numpy as np
from fastmri.pl_modules import mri_module

# Patch log_image (same as original) so images work for both TB and W&B.

def _patched_log_image(self, name: str, image):
    img = image.detach().cpu().numpy()
    if hasattr(self.logger.experiment, "add_image"):
        self.logger.experiment.add_image(name, img, global_step=self.global_step)
    elif isinstance(self.logger.experiment, wandb.wandb_sdk.wandb_run.Run):
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        self.logger.log_image(key=name, images=[img], caption=[f"Step {self.global_step}"])

mri_module.MriModule.log_image = _patched_log_image
mri_module.MriModule.num_log_images = 8


def build_args():
    parser = ArgumentParser()

    # Mode (only training here, but keep option for test if desired)
    parser.add_argument("--mode", default="train", choices=("train", "test"), type=str)

    # Sequential training config
    parser.add_argument(
        "--epochs_per_category", default=5, type=int,
        help="Number of epochs to train per category before moving to the next"
    )
    parser.add_argument(
        "--resume_from_checkpoint", default=None, type=Path,
        help="Optional checkpoint to resume from for the FIRST category"
    )

    # Data module specific args (adds data_path, category_mapping_path, category_order, etc.)
    parser = CustomCategoryDataModule.add_data_specific_args(parser)

    # Mask / transform params
    parser.add_argument(
        "--mask_type", choices=("random", "equispaced_fraction"), default="equispaced_fraction", type=str,
        help="Type of k-space mask"
    )
    parser.add_argument(
        "--center_fractions", nargs="+", default=[0.08], type=float,
        help="Center fractions for mask"
    )
    parser.add_argument(
        "--accelerations", nargs="+", default=[4], type=int,
        help="Acceleration factors for mask"
    )

    # Model hyperparameters (VarNet)
    parser.add_argument("--num_cascades", default=12, type=int)
    parser.add_argument("--pools", default=4, type=int)
    parser.add_argument("--chans", default=18, type=int)
    parser.add_argument("--sens_pools", default=4, type=int)
    parser.add_argument("--sens_chans", default=8, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_step_size", default=40, type=int)
    parser.add_argument("--lr_gamma", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)

    # Trainer/global settings
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--deterministic", default=True, type=bool)
    parser.add_argument("--strategy", default="ddp", type=str, help="Distributed strategy (e.g., ddp, ddp_cpu, auto)")
    parser.add_argument("--gpus", default=2, type=int, help="Number of GPUs to use")
    parser.add_argument("--gradient_clip_val", default=1.0, type=float)
    parser.add_argument(
        "--default_root_dir", default=str(Path.cwd() / "runs"), type=str,
        help="Base root dir for logging and checkpoints"
    )

    args = parser.parse_args()

    # Root directory stamping with cascades/acc & timestamp
    args.default_root_dir = args.default_root_dir + f"_casc{args.num_cascades}_acc{args.accelerations[0]}"
    base_root_dir = pathlib.Path(args.default_root_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = base_root_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    args.default_root_dir = run_root

    return args


def make_transforms(args):
    mask = create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations)
    train_t = VarNetDataTransform(mask_func=mask, use_seed=False)
    val_t = VarNetDataTransform(mask_func=mask)
    test_t = VarNetDataTransform()
    return train_t, val_t, test_t


def make_model(args):
    model = VarNetModule(
        num_cascades=args.num_cascades,
        pools=args.pools,
        chans=args.chans,
        sens_pools=args.sens_pools,
        sens_chans=args.sens_chans,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
    )
    return model


def train_sequential(args):
    pl.seed_everything(args.seed, workers=True)

    if not args.category_order or len(args.category_order) == 0:
        raise ValueError("Provide --category_order with at least one category.")

    train_transform, val_transform, test_transform = make_transforms(args)
    model = make_model(args)

    # Optionally resume from a checkpoint for the first category only
    if args.resume_from_checkpoint is not None and Path(args.resume_from_checkpoint).is_file():
        ckpt_path = str(args.resume_from_checkpoint)
    else:
        ckpt_path = None

    # WandB main run (aggregate)
    main_logger = WandbLogger(project="mskmri-varnet", name=f"sequential_{Path(args.default_root_dir).name}")

    # Iterate categories sequentially
    for idx, category in enumerate(args.category_order):
        cat_root = Path(args.default_root_dir) / f"cat_{idx+1}_{category.upper()}"
        cat_root.mkdir(parents=True, exist_ok=True)

        # DataModule restricted to single category
        data_module = CustomCategoryDataModule(
            data_path=args.data_path,
            category_mapping_path=args.category_mapping_path,
            category_order=[category],  # single category
            challenge=args.challenge,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            test_split=args.test_split,
            test_path=args.test_path,
            sample_rate=args.sample_rate,
            val_sample_rate=args.val_sample_rate,
            test_sample_rate=args.test_sample_rate,
            volume_sample_rate=args.volume_sample_rate,
            val_volume_sample_rate=args.val_volume_sample_rate,
            test_volume_sample_rate=args.test_volume_sample_rate,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed_sampler=(args.strategy in ("ddp", "ddp_cpu")),
        )

        # Dedicated logger per category for clarity
        cat_logger = WandbLogger(project="mskmri-varnet", name=f"{category}_segment")

        checkpoint_dir = cat_root / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath=checkpoint_dir,
                save_top_k=-1,
                verbose=True,
                every_n_epochs=1,
                save_on_train_epoch_end=True,
            )
        ]

        trainer = pl.Trainer(
            max_epochs=args.epochs_per_category,
            accelerator="gpu" if args.gpus > 0 else "cpu",
            devices=args.gpus if args.gpus > 0 else None,
            strategy=args.strategy,
            deterministic=args.deterministic,
            gradient_clip_val=args.gradient_clip_val,
            default_root_dir=str(cat_root),
            logger=[main_logger, cat_logger],
            callbacks=callbacks,
        )

        print(f"=== Training category {category} ({idx+1}/{len(args.category_order)}) for {args.epochs_per_category} epochs ===")
        trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path if idx == 0 else None)
        ckpt_path = None  # Only apply resume for first segment

    print("Sequential category training complete.")


def main():
    args = build_args()
    if args.mode != "train":
        raise ValueError("This script is intended for sequential training only (mode=train).")
    train_sequential(args)


if __name__ == "__main__":
    main()
