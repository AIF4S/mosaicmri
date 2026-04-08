#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

CUDA_VISIBLE_DEVICES=1 python run_end2end_baseline_direct.py \
  --model u-net \
  --train_h5_dir /data/datasets/msk_mri_h5/varnet_msk_dataset2/multicoil_train \
  --eval_h5_dir /data/datasets/msk_mri_h5/varnet_msk_dataset2/multicoil_val \
  --category_mapping_path /data/datasets/msk_mri_h5/file_category_mapping.json \
  --val_anatomy KNEE \
  --train_slice_limit 10000 \
  --selection_seed 24 \
  --output_dir /data/checkpoints/msk_mri_dataset/logs/logs/rebuttal_march \
  --run_name r2_unet_mixed_anatomy_10k_knee_val \
  --mask_type random \
  --center_fractions 0.04 \
  --accelerations 8 \
  --unet_chans 128 \
  --unet_num_pools 4 \
  --lr 0.001 \
  --batch_size 4 \
  --num_workers 16 \
  --num_epochs 50 \
  --num_checkpoints 1 \
  --model_seed 24 \
  --world_size 1 \
  --val_every_n_epochs 1 \
  --enable_wandb \
  --wandb_project mskmri-rebuttal \
  --wandb_name r2_unet_mixed_anatomy_10k_knee_val \
  --wandb_log_images \
  --wandb_num_images 4 \
  --skip_eval

