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

import torch

from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.data import transforms as T
from fastmri.pl_modules import FastMriDataModule, VarNetModule
from pytorch_lightning.loggers import WandbLogger
import wandb
import numpy as np
from fastmri.pl_modules import mri_module
from datetime import datetime
from dataloaders.custom_mix_data_module import CustomMixDataModule  # Import our custom module
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class VarNetModuleWithDebugVis(VarNetModule):
    """VarNetModule with optional first-batches visualization.

    When enabled (by setting attributes on the instance), this saves two PNGs per
    batch:
      - mask visualization
      - masked k-space (log-magnitude) visualization
    """

    def _maybe_save_debug_vis(self, batch, batch_idx: int, model_output=None) -> None:
        if not getattr(self, "debug_vis_enabled", False):
            return
        if getattr(self, "debug_vis_done", 0) >= int(getattr(self, "debug_vis_max_batches", 0)):
            return

        # In DDP, avoid multiple ranks writing the same files.
        if getattr(self, "debug_vis_rank_zero_only", True):
            if hasattr(self, "global_rank") and int(self.global_rank) != 0:
                return

        out_dir = Path(getattr(self, "debug_vis_dir", Path(".")))
        out_dir.mkdir(parents=True, exist_ok=True)

        masked_kspace = batch.masked_kspace.detach().cpu()

        # masked_kspace: (C, H, W, 2) complex-last OR (C, H, W) complex.
        if masked_kspace.shape[-1] == 2:
            ks_cplx = torch.view_as_complex(masked_kspace)
        else:
            ks_cplx = masked_kspace

        # Masked k-space log-magnitude (display first coil only to keep it simple).
        ks0 = ks_cplx[0]
        ks_log = torch.log(ks0.abs() + 1e-9).numpy()

        # Zero-filled reconstruction from masked k-space (RSS over coils).
        img_coils = torch.fft.ifftshift(
            torch.fft.ifft2(
                torch.fft.fftshift(ks_cplx, dim=(-2, -1)),
                norm="ortho",
            ),
            dim=(-2, -1),
        )
        img_rss = torch.sqrt((img_coils.abs() ** 2).sum(dim=0))
        img_rss_np = img_rss.numpy()

        # Target + model output (optional). We center-crop for display (and to match target/output).
        target_t = None
        if hasattr(batch, "target") and isinstance(batch.target, torch.Tensor):
            target_t = batch.target.detach().cpu()

        output_t = None
        if isinstance(model_output, torch.Tensor):
            output_t = model_output.detach().cpu()

        # Ensure both are shape-compatible and center-cropped like evaluation pipelines.
        crop_size = getattr(batch, "crop_size", None)
        if isinstance(crop_size, torch.Tensor):
            crop_size = tuple(int(x) for x in crop_size.flatten()[:2].tolist())
        elif isinstance(crop_size, (list, tuple)) and len(crop_size) >= 2:
            crop_size = (int(crop_size[0]), int(crop_size[1]))
        else:
            crop_size = None

        # Crop model output first (if we can), then crop both to smallest common size.
        if output_t is not None:
            try:
                # Typical VarNet output is (B, H, W). Some variants may be (B, 1, H, W).
                if crop_size is not None:
                    output_t = T.center_crop(output_t, crop_size)
            except Exception:
                pass

        if target_t is not None and output_t is not None:
            try:
                target_t, output_t = T.center_crop_to_smallest(target_t, output_t)
            except Exception:
                pass

        # Convert to numpy for plotting (show first in batch if needed).
        target_np = None
        if isinstance(target_t, torch.Tensor):
            targ = target_t
            if targ.ndim >= 3:
                targ = targ[0]
            target_np = targ.squeeze().numpy()

        output_np = None
        if isinstance(output_t, torch.Tensor):
            out = output_t
            if out.ndim == 4 and out.shape[1] == 1:
                out = out[:, 0]
            if out.ndim >= 3:
                out = out[0]
            # If complex-as-two-channels, take magnitude
            if out.ndim == 3 and out.shape[-1] == 2:
                out = torch.view_as_complex(out)
                out = out.abs()
            output_np = out.squeeze().numpy()


        
        slice_num = int(getattr(batch, 'slice_num', [0])[0])
        base = f"train_batch{slice_num:06d}"

        ncols = 4 if output_np is not None else 3
        fig, axs = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
        if ncols == 1:
            axs = [axs]

        axs[0].imshow(ks_log[0], cmap="gray")
        axs[0].set_title("Masked k-space (log-mag)\ncoil0")
        axs[0].axis("off")

        axs[1].imshow(img_rss_np[0], cmap="gray")
        axs[1].set_title("Zero-filled (from masked)\nRSS")
        axs[1].axis("off")

        if output_np is not None and getattr(output_np, "ndim", 0) == 2:
            axs[2].imshow(output_np, cmap="gray", zorder=0)
            axs[2].set_title("Model output")
        else:
            axs[2].text(0.5, 0.5, "No output", ha="center", va="center")
            axs[2].set_title("Model output")
        # Force a visible white grid overlay.
        # Use minor ticks for dense grid lines and draw them above the image.
        axs[2].set_axis_on()
        axs[2].set_frame_on(False)
        axs[2].tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        if output_np is not None and getattr(output_np, "ndim", 0) == 2:
            h, w = output_np.shape
            step = max(16, int(min(h, w) // 16))  # ~16 grid cells across
            axs[2].set_xticks(np.arange(0, w + 1, step), minor=True)
            axs[2].set_yticks(np.arange(0, h + 1, step), minor=True)
        axs[2].grid(True, which="minor", color="white", linestyle="-", linewidth=1.2, alpha=0.95)
        axs[2].set_axisbelow(False)

        t_idx = 3 if ncols == 4 else 2
        if ncols == 4:
            if target_np is not None and target_np.ndim == 2:
                axs[t_idx].imshow(target_np, cmap="gray", zorder=0)
                axs[t_idx].set_title("Target")
            else:
                axs[t_idx].text(0.5, 0.5, "No target", ha="center", va="center")
                axs[t_idx].set_title("Target")
            axs[t_idx].set_axis_on()
            axs[t_idx].set_frame_on(False)
            axs[t_idx].tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
            if target_np is not None and getattr(target_np, "ndim", 0) == 2:
                h, w = target_np.shape
                step = max(16, int(min(h, w) // 16))
                axs[t_idx].set_xticks(np.arange(0, w + 1, step), minor=True)
                axs[t_idx].set_yticks(np.arange(0, h + 1, step), minor=True)
            axs[t_idx].grid(True, which="minor", color="white", linestyle="-", linewidth=1.2, alpha=0.95)
            axs[t_idx].set_axisbelow(False)

        fig.suptitle(f"{getattr(batch, 'fname', [''])[0]} | slice {slice_num}", fontsize=10)
        plt.tight_layout()
        plt.savefig(out_dir / f"{base}_debug_vis.png", dpi=150)
        plt.close(fig)

        self.debug_vis_done = int(getattr(self, "debug_vis_done", 0)) + 1

    def training_step(self, batch, batch_idx):
        # For debug vis, try to capture the model output for the current batch.
        # We do this under no_grad so it doesn't interfere with training graphs.
        model_out = None
        if getattr(self, "debug_vis_enabled", False):
            try:
                with torch.no_grad():
                    # VarNetModule.forward typically takes masked_kspace, mask, and num_low_frequencies.
                    # Our batch object should provide these attributes.
                    mk = batch.masked_kspace
                    m = batch.mask
                    nlf = getattr(batch, "num_low_frequencies", None)
                    if nlf is None:
                        model_out = self(mk, m)
                    else:
                        model_out = self(mk, m, nlf)
            except Exception:
                # If anything about the forward signature or batch fields differs, just skip output vis.
                model_out = None

        self._maybe_save_debug_vis(batch, batch_idx, model_output=model_out)
        return super().training_step(batch, batch_idx)

from dataloaders.custom_category_data_module_clean import CustomCategoryDataModule
from dataloaders.custom_single_category_mix_data_module import SingleCategoryMixDataModule
from dataloaders.custom_balanced_category_data_module import BalancedCategoryDataModule
from dataloaders.custom_json_train_knee_val_data_module import JsonTrainKneeValDataModule
from dataloaders.rebuttal_march_data_modules import (
    FullNoAnkleNoAnkleValDataModule,
    FullNoHipNoHipValDataModule,
    FullNoWristHandNoWristHandValDataModule,
    FullNoShoulderNoShoulderValDataModule,
    FullRandomContrastFamilyShoulderValDataModule,
    FullNoKneeNoKneeValDataModule,
    FullRandomKneeValDataModule,
    FullRandomShoulderValDataModule,
    FullRandomSingleContrastKneeValDataModule,
    ShoulderContrastFamilyShoulderValDataModule,
    ShoulderOnlyShoulderValDataModule,
    KneeOnlyKneeValDataModule,
    KneeOnlySingleContrastKneeValDataModule,
)


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


class ValidationMetricsPrinter(pl.Callback):
    """Print PSNR and SSIM at the end of validation.

    This callback looks for metric keys containing 'psnr' or 'ssim' (case-insensitive)
    in trainer.callback_metrics and prints them on rank 0.
    """
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Only print from rank 0 if distributed
        try:
            import torch
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                if torch.distributed.get_rank() != 0:
                    return
        except Exception:
            pass

        metrics = trainer.callback_metrics
        # metrics can be a mapping from metric name to tensor/number
        found = {}
        for k, v in metrics.items():
            name = str(k).lower()
            if "psnr" in name:
                try:
                    val = float(v.detach().cpu().item()) if hasattr(v, "detach") else float(v)
                except Exception:
                    try:
                        val = float(v)
                    except Exception:
                        continue
                found["PSNR"] = val
            if "ssim" in name:
                try:
                    val = float(v.detach().cpu().item()) if hasattr(v, "detach") else float(v)
                except Exception:
                    try:
                        val = float(v)
                    except Exception:
                        continue
                found["SSIM"] = val

        if found:
            psnr = found.get("PSNR", None)
            ssim = found.get("SSIM", None)
            psnr_str = f"{psnr:.2f} dB" if psnr is not None else "N/A"
            ssim_str = f"{ssim:.4f}" if ssim is not None else "N/A"
            print(f"[val summary] PSNR: {psnr_str} | SSIM: {ssim_str}")

def cli_main(args):
    pl.seed_everything(args.seed)

    run_name = str(args.default_root_dir).split("/")[-2]
    wandb_logger = None
    if not args.debug:
        wandb_logger = WandbLogger(
            project="mskmri-rebuttal",
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

    elif args.data_loader == "json_train_knee_val":
        # Train on explicit JSON slice selection; validate on one MSK category.
        # Note: we intentionally treat *all* JSON slices as training set.
        data_module = JsonTrainKneeValDataModule(
            data_path_train_root=args.data_path_train_root,
            selector_json_path=args.selector_json_path,
            data_path_val_root=args.data_path_val_root,
            category_mapping_path=args.category_mapping_path,
            challenge=args.challenge,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            require_json_split=args.require_json_split,
            val_category=args.json_val_category,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
        )
    elif args.data_loader == "knee_only_knee_val":
        data_module = KneeOnlyKneeValDataModule(
            data_path=args.data_path,
            category_mapping_path=args.category_mapping_path,
            challenge=args.challenge,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            random_seed=args.rebuttal_random_seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
            use_dataset_cache_file=args.use_dataset_cache_file,
        )
    elif args.data_loader == "full_random_knee_val":
        data_module = FullRandomKneeValDataModule(
            data_path=args.data_path,
            category_mapping_path=args.category_mapping_path,
            challenge=args.challenge,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            train_slice_limit=args.train_slice_limit,
            random_seed=args.rebuttal_random_seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
            use_dataset_cache_file=args.use_dataset_cache_file,
        )
    elif args.data_loader == "full_no_knee_no_knee_val":
        data_module = FullNoKneeNoKneeValDataModule(
            data_path=args.data_path,
            category_mapping_path=args.category_mapping_path,
            challenge=args.challenge,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            random_seed=args.rebuttal_random_seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
            use_dataset_cache_file=args.use_dataset_cache_file,
        )
    elif args.data_loader == "full_no_ankle_no_ankle_val":
        data_module = FullNoAnkleNoAnkleValDataModule(
            data_path=args.data_path,
            category_mapping_path=args.category_mapping_path,
            challenge=args.challenge,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            random_seed=args.rebuttal_random_seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
            use_dataset_cache_file=args.use_dataset_cache_file,
        )
    elif args.data_loader == "full_no_shoulder_no_shoulder_val":
        data_module = FullNoShoulderNoShoulderValDataModule(
            data_path=args.data_path,
            category_mapping_path=args.category_mapping_path,
            challenge=args.challenge,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            random_seed=args.rebuttal_random_seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
            use_dataset_cache_file=args.use_dataset_cache_file,
        )
    elif args.data_loader == "full_no_hip_no_hip_val":
        data_module = FullNoHipNoHipValDataModule(
            data_path=args.data_path,
            category_mapping_path=args.category_mapping_path,
            challenge=args.challenge,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            random_seed=args.rebuttal_random_seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
            use_dataset_cache_file=args.use_dataset_cache_file,
        )
    elif args.data_loader == "full_no_wrist_hand_no_wrist_hand_val":
        data_module = FullNoWristHandNoWristHandValDataModule(
            data_path=args.data_path,
            category_mapping_path=args.category_mapping_path,
            challenge=args.challenge,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            random_seed=args.rebuttal_random_seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
            use_dataset_cache_file=args.use_dataset_cache_file,
        )
    elif args.data_loader == "knee_single_contrast_knee_val":
        data_module = KneeOnlySingleContrastKneeValDataModule(
            data_path=args.data_path,
            category_mapping_path=args.category_mapping_path,
            challenge=args.challenge,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            contrast_name=args.contrast_name,
            random_seed=args.rebuttal_random_seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
            use_dataset_cache_file=args.use_dataset_cache_file,
        )
    elif args.data_loader == "full_random_single_contrast_knee_val":
        data_module = FullRandomSingleContrastKneeValDataModule(
            data_path=args.data_path,
            category_mapping_path=args.category_mapping_path,
            challenge=args.challenge,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            train_slice_limit=args.train_slice_limit,
            contrast_name=args.contrast_name,
            match_contrast_ratio=args.match_contrast_ratio,
            random_seed=args.rebuttal_random_seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
            use_dataset_cache_file=args.use_dataset_cache_file,
        )
    elif args.data_loader == "shoulder_only_shoulder_val":
        data_module = ShoulderOnlyShoulderValDataModule(
            data_path=args.data_path,
            category_mapping_path=args.category_mapping_path,
            challenge=args.challenge,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            train_slice_limit=args.train_slice_limit,
            random_seed=args.rebuttal_random_seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
            use_dataset_cache_file=args.use_dataset_cache_file,
        )
    elif args.data_loader == "full_random_shoulder_val":
        data_module = FullRandomShoulderValDataModule(
            data_path=args.data_path,
            category_mapping_path=args.category_mapping_path,
            challenge=args.challenge,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            train_slice_limit=args.train_slice_limit,
            random_seed=args.rebuttal_random_seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
            use_dataset_cache_file=args.use_dataset_cache_file,
        )
    elif args.data_loader == "shoulder_contrast_family_shoulder_val":
        data_module = ShoulderContrastFamilyShoulderValDataModule(
            data_path=args.data_path,
            category_mapping_path=args.category_mapping_path,
            challenge=args.challenge,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            train_slice_limit=args.train_slice_limit,
            contrast_name=args.contrast_name,
            random_seed=args.rebuttal_random_seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
            use_dataset_cache_file=args.use_dataset_cache_file,
        )
    elif args.data_loader == "full_random_contrast_family_shoulder_val":
        data_module = FullRandomContrastFamilyShoulderValDataModule(
            data_path=args.data_path,
            category_mapping_path=args.category_mapping_path,
            challenge=args.challenge,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            train_slice_limit=args.train_slice_limit,
            contrast_name=args.contrast_name,
            match_contrast_ratio=args.match_contrast_ratio,
            random_seed=args.rebuttal_random_seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
            use_dataset_cache_file=args.use_dataset_cache_file,
        )

    # ------------
    # model
    # ------------
    model = VarNetModuleWithDebugVis(
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

    # Optional: save a few debug visuals from the first training batches.
    if getattr(args, "debug_vis", False):
        debug_dir = Path(args.default_root_dir) / "debug_vis"
        model.debug_vis_enabled = True
        model.debug_vis_dir = debug_dir
        model.debug_vis_max_batches = int(getattr(args, "debug_vis_batches", 2))
        model.debug_vis_rank_zero_only = bool(getattr(args, "debug_vis_rank_zero_only", True))
        model.debug_vis_done = 0
    

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
        choices=(
            "default",
            "custom_mix",
            "categories",
            "knees_only",
            "balanced_categories",
            "json_train_knee_val",
            "knee_only_knee_val",
            "full_random_knee_val",
            "full_no_knee_no_knee_val",
            "full_no_ankle_no_ankle_val",
            "full_no_shoulder_no_shoulder_val",
            "full_no_hip_no_hip_val",
            "full_no_wrist_hand_no_wrist_hand_val",
            "knee_single_contrast_knee_val",
            "full_random_single_contrast_knee_val",
            "shoulder_only_shoulder_val",
            "full_random_shoulder_val",
            "shoulder_contrast_family_shoulder_val",
            "full_random_contrast_family_shoulder_val",
        ),
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
        "--selector_json_path",
        default=None,
        type=Path,
        help="Path to dataset JSON containing selected files/slices (used by json_train_knee_val)",
    )

    # JsonTrainKneeValDataModule-specific
    parser.add_argument(
        "--data_path_train_root",
        default=None,
        type=Path,
        help="Root used to resolve JSON-selected training files (used by json_train_knee_val)",
    )
    parser.add_argument(
        "--data_path_val_root",
        default=None,
        type=Path,
        help="Validation root directory (e.g. .../varnet_msk_dataset2/multicoil_val) (used by json_train_knee_val)",
    )
    parser.add_argument(
        "--require_json_split",
        action="store_true",
        help="If set, dataset1 only uses JSON entries whose split matches each partition (train/val/test).",
    )
    parser.add_argument(
        "--json_val_category",
        default="KNEE",
        type=str,
        help="Validation category for json_train_knee_val loader (e.g. KNEE, SHOULDER).",
    )

    # Rebuttal March dataloader-specific
    parser.add_argument(
        "--train_slice_limit",
        default=10000,
        type=int,
        help="Slice limit for full-random rebuttal dataloaders.",
    )
    parser.add_argument(
        "--contrast_name",
        default=None,
        type=str,
        help="Contrast name for single-contrast rebuttal dataloaders (e.g., PD_FS, T1, T2_FS).",
    )
    parser.add_argument(
        "--rebuttal_random_seed",
        default=24,
        type=int,
        help="Random seed used for rebuttal random slice selection.",
    )
    parser.add_argument(
        "--match_contrast_ratio",
        action="store_true",
        help="When using mixed-anatomy contrast-filtered loader with slice cap, match contrast slice ratio (e.g., PD vs PD_FS).",
    )
    
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

    # Optional training debug visualization.
    parser.add_argument(
        "--debug_vis",
        action="store_true",
        help="If set, saves mask + masked k-space plots for the first few training batches.",
    )
    parser.add_argument(
        "--debug_vis_batches",
        type=int,
        default=2,
        help="How many initial training batches to visualize when --debug_vis is set.",
    )
    parser.add_argument(
        "--debug_vis_rank_zero_only",
        action="store_true",
        help="Only save debug visuals on global rank 0 (recommended for DDP).",
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

    # Add validation metrics printer to callbacks so PSNR/SSIM are printed after val
    args.callbacks.append(ValidationMetricsPrinter())

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
