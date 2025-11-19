"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pathlib
from argparse import ArgumentParser

import pytorch_lightning as pl

from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.pl_modules import FastMriDataModule, VarNetModule
from pytorch_lightning.loggers import WandbLogger
import wandb
import numpy as np
from fastmri.pl_modules import mri_module

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

def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # WandB logger
    wandb_logger = WandbLogger(
        project="mskmri-varnet",   # your wandb project
        name=f"varnet_casc{args.num_cascades}_bs{args.batch_size}",  # run name
    )

    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    # use random masks for train transform, fixed masks for val transform
    train_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
    val_transform = VarNetDataTransform(mask_func=mask)
    test_transform = VarNetDataTransform()
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
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
        logger=wandb_logger,
        gradient_clip_val=1.0,   # new: helps prevent exploding grads → NaNs
        detect_anomaly=False,    # set to True if you want autograd to pinpoint bad ops
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
    # data config
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        mask_type="equispaced_fraction",
        challenge="multicoil",
        batch_size=4,
        test_path=None,
    )
    # module config
    parser = VarNetModule.add_model_specific_args(parser)
    parser.set_defaults(
        num_cascades=8,  # number of unrolled iterations
        pools=4,  # number of pooling layers for U-Net
        chans=18,  # number of top-level channels for U-Net
        sens_pools=4,  # number of pooling layers for sense est. U-Net
        sens_chans=8,  # number of top-level channels for sense est. U-Net
        lr=1e-4,  # Adam learning rate
        lr_step_size=40,  # epoch at which to decrease learning rate
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

    # configure checkpointing in checkpoint_dir
    args.data_path = pathlib.Path(args.data_path) if args.data_path else None
    args.default_root_dir = pathlib.Path(args.default_root_dir) if args.default_root_dir else None
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.default_root_dir / "checkpoints",
            save_top_k=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]

    # set default checkpoint if one exists in our checkpoint directory
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
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
