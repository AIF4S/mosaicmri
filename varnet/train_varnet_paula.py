"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pathlib
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.pl_modules import FastMriDataModule, VarNetModule
from pytorch_lightning.loggers import WandbLogger
import wandb
import numpy as np
from fastmri.pl_modules import mri_module
from datetime import datetime
from dataloaders.custom_mix_data_module import CustomMixDataModule  # Import our custom module
from pathlib import Path

from dataloaders.custom_category_data_module_clean import CustomCategoryDataModule
from dataloaders.custom_single_category_mix_data_module import SingleCategoryMixDataModule
from dataloaders.custom_balanced_category_data_module import BalancedCategoryDataModule


def patched_log_image(self, name: str, image):
    """Patched log_image that works with both TB and W&B."""
    img = image.detach().cpu().numpy()
    if hasattr(self.logger.experiment, "add_image"):
        # TensorBoard
        self.logger.experiment.add_image(name, img, global_step=self.global_step)
    elif isinstance(self.logger.experiment, wandb.wandb_sdk.wandb_run.Run):
        # W&B
        img = image.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] in (1, 3):  # (C, H, W) -> (H, W, C)
            img = np.transpose(img, (1, 2, 0))

        # use WandbLogger API (works reliably)
        self.logger.log_image(
            key=name,
            images=[img],
            caption=[f"Step {self.global_step}"]
        )
        print(f"Logged image {name} to W&B")

mri_module.MriModule.log_image = patched_log_image
mri_module.MriModule.num_log_images = 4

class IntraEpochCheckpoint(pl.Callback):
    """Save multiple checkpoints during each training epoch at evenly spaced batch indices.

    This avoids only saving at epoch end and instead saves `num_per_epoch` times within the epoch.
    """
    def __init__(self, dirpath: pathlib.Path, num_per_epoch: int = 10):
        super().__init__()
        self.dirpath = pathlib.Path(dirpath)
        self.num_per_epoch = max(0, int(num_per_epoch))
        self._targets = []  # batch indices at which to save within current epoch
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.num_per_epoch <= 0:
            self._targets = []
            return
        # Determine number of training batches for this epoch
        num_batches = trainer.num_training_batches
        if isinstance(num_batches, (list, tuple)):
            # In case of multiple dataloaders, take total
            num_batches = sum(int(x) for x in num_batches)
        num_batches = int(num_batches) if num_batches is not None else 0
        if num_batches <= 1:
            self._targets = []
            return
        # Evenly spaced indices excluding 0 and last batch
        # positions are computed with (k / (N+1)) * num_batches
        targets = []
        for k in range(1, self.num_per_epoch + 1):
            t = (k / (self.num_per_epoch + 1)) * num_batches
            idx = max(0, min(num_batches - 1, int(round(t)) - 1))
            targets.append(idx)
        # Deduplicate and sort
        self._targets = sorted(set(targets))

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int) -> None:
        if not self._targets:
            return
        current_epoch = trainer.current_epoch
        if batch_idx in self._targets:
            # Build filename
            global_step = trainer.global_step
            fname = f"epoch{current_epoch:03d}_intra_step{global_step}.ckpt"
            save_path = self.dirpath / fname
            # Use trainer to save checkpoint to include all state
            trainer.save_checkpoint(str(save_path))
            # Remove this target so we don't save twice for the same batch
            try:
                self._targets.remove(batch_idx)
            except ValueError:
                pass

def cli_main(args):
    pl.seed_everything(args.seed)

    run_name = str(args.default_root_dir).split("/")[-2]
    wandb_logger = None
    if not args.debug:
        wandb_logger = WandbLogger(
            project="mskmri-varnet",
            name=run_name
        )

    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    # use random masks for train transform, fixed masks for val transform
    train_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
    val_transform = VarNetDataTransform(mask_func=mask)
    test_transform = VarNetDataTransform()

    # Both can't be not None at the same timw
    if args.volume_sample_rate is not None and args.volume_sample_rate <= 1:
        args.sample_rate, args.val_sample_rate, args.test_sample_rate = None, None, None

    if args.sample_rate is not None and args.sample_rate <= 1:
        args.volume_sample_rate, args.val_volume_sample_rate, args.test_volume_sample_rate = None, None, None

    if args.data_loader == "default":
        data_module = FastMriDataModule(
            data_path=args.data_path,
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
            distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
        )
    elif args.data_loader == "custom_mix":
        data_module = CustomMixDataModule(
            data_path1=args.data_path1,  # First dataset path
            data_path2=args.data_path2,  # Second dataset path
            challenge=args.challenge,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            test_split=args.test_split,
            test_path1=args.test_path1,
            test_path2=args.test_path2,
            sample_rate=args.sample_rate, 
            val_sample_rate=args.val_sample_rate, 
            test_sample_rate=args.test_sample_rate, 
            volume_sample_rate=args.volume_sample_rate,
            val_volume_sample_rate=args.val_volume_sample_rate,
            test_volume_sample_rate=args.test_volume_sample_rate,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
        )

    elif args.data_loader == "categories":
        data_module = CustomCategoryDataModule(
            data_path=args.data_path,  # First dataset path
            category_mapping_path=args.category_mapping_path,
            category_order=args.category_order,
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
            distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
        )
    elif args.data_loader == "balanced_categories":
        # Expect args.volume_rates to be a list matching category_order
        if args.volume_rates is None:
            raise ValueError("--volume_rates must be provided for balanced_categories")
        data_module = BalancedCategoryDataModule(
            data_path=args.data_path,
            category_mapping_path=args.category_mapping_path,
            category_order=args.category_order,
            volume_rates=args.volume_rates,
            challenge=args.challenge,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_dataset_cache_file=(args.use_dataset_cache_file if hasattr(args, 'use_dataset_cache_file') else True),
            distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
        )
    elif args.data_loader == "knees_only":
        data_module = SingleCategoryMixDataModule(
            data_path1=args.data_path1,  # First dataset path 
            data_path2=args.data_path2,  # Second dataset path
            category_mapping_path=args.category_mapping_path,
            category_name="knee",
            challenge=args.challenge,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            val_source=getattr(args, "val_source", "mixed"),
            test_split=args.test_split,
            test_path1=args.test_path1,
            test_path2=args.test_path2,
            sample_rate=args.sample_rate, 
            val_sample_rate=args.val_sample_rate, 
            test_sample_rate=args.test_sample_rate, 
            volume_sample_rate=args.volume_sample_rate,
            val_volume_sample_rate=args.val_volume_sample_rate,
            test_volume_sample_rate=args.test_volume_sample_rate,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
        )

    # ------------
    # model
    # ------------
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
    

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(
        args,
        gradient_clip_val=1.0,   # new: helps prevent exploding grads → NaNs
        detect_anomaly=False,    # set to True if you want autograd to pinpoint bad ops
        callbacks=args.callbacks,
    logger=wandb_logger if wandb_logger is not None else True,
    )

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    parser = ArgumentParser()

    # basic args
    backend = "ddp"
    num_gpus = 2 if backend == "ddp" else 1

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )
    parser.add_argument(
        "--data_loader",
        default="default",
        choices=("default", "custom_mix", "categories", "knees_only", "balanced_categories"),
        type=str,
        help="Data loader type",
    )
    parser.add_argument(
        "--data_path1", default=None, type=Path)
    parser.add_argument(
        "--data_path2", default=None, type=Path)
    parser.add_argument(
        "--test_path1", default=None, type=Path)
    parser.add_argument(
        "--test_path2", default=None, type=Path)
    
    parser.add_argument(
        "--category_mapping_path", default=None, type=Path,
        help="Path to JSON file mapping filenames to categories"
    )
    parser.add_argument(
        "--category_order", 
        nargs="+",
        default=[],
        type=str,
        help="List of categories to include, in order"
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced_fraction"),
        default="equispaced_fraction",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, use small subset of data for faster debugging",
    )
    # data config
    parser = FastMriDataModule.add_data_specific_args(parser)

    # SingleCategoryMixDataModule-specific
    parser.add_argument(
        "--val_source",
        choices=("mixed", "data1", "data2"),
        default="mixed",
        type=str,
        help="Validation set source for knees_only dataloader: mixed concat, dataset1-only, or dataset2-only.",
    )
    parser.add_argument(
        "--volume_rates",
        nargs="+",
        type=float,
        default=None,
        help="Per-category volume sampling rates (one float per category in --category_order).",
    )
    parser.set_defaults(
        mask_type="equispaced_fraction",
        challenge="multicoil",
        batch_size=4,
        test_path=None,
    )
    parser.add_argument(
        "--intra_epoch_checkpoints",
        type=int,
        default=0,
        help="If > 0, save this many evenly spaced checkpoints during each training epoch (intra-epoch).",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed"
    )
    # module config
    parser = VarNetModule.add_model_specific_args(parser)
    parser.set_defaults(
        num_cascades=12,  # number of unrolled iterations
        pools=4,  # number of pooling layers for U-Net
        chans=18,  # number of top-level channels for U-Net
        sens_pools=4,  # number of pooling layers for sense est. U-Net
        sens_chans=8,  # number of top-level channels for sense est. U-Net
        lr=1e-4,  # Adam learning rate
        lr_step_size=100,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight regularization strength
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        strategy=backend,  # what distributed version to use
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
        max_epochs=50,  # max number of epochs
    )

    args = parser.parse_args()

    args.default_root_dir = args.default_root_dir + f"_casc{args.num_cascades}_acc{args.accelerations[0]}"
    base_root_dir = (pathlib.Path(args.default_root_dir) if args.default_root_dir else pathlib.Path.cwd())
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. 20251119_103012
    run_root = base_root_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    args.default_root_dir = run_root

    checkpoint_dir = args.default_root_dir / "checkpoints" 
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.default_root_dir / "checkpoints",
            save_top_k=-1,                      # <--- KEEP ALL CHECKPOINTS
            verbose=True,
            monitor=None,
            mode="min",
            every_n_epochs=1,                   # <--- SAVE ONCE PER EPOCH
            save_on_train_epoch_end=True,       # ensure saving at epoch end
        )
    ]
    # Log learning rate to logger (e.g., W&B) every step
    args.callbacks.append(LearningRateMonitor(logging_interval="step"))

    # Optionally add intra-epoch checkpointing
    if getattr(args, "intra_epoch_checkpoints", 0) and args.intra_epoch_checkpoints > 0:
        args.callbacks.append(
            IntraEpochCheckpoint(
                dirpath=checkpoint_dir,
                num_per_epoch=args.intra_epoch_checkpoints,
            )
        )

    # set default checkpoint if one exists in our checkpoint directory
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(base_root_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            print(f"Found existing checkpoint(s), resuming from latest: {ckpt_list[-1]}")
            args.resume_from_checkpoint = str(ckpt_list[-1])

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()
