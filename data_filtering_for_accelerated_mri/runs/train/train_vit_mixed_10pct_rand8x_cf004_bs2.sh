#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


CUDA_VISIBLE_DEVICES=0 python run_end2end_baseline_direct.py \
  --model vit \
  --train_h5_dir /data/datasets/msk_mri_h5/varnet_msk_dataset2/multicoil_train \
  --eval_h5_dir /data/datasets/msk_mri_h5/varnet_msk_dataset2/multicoil_val \
  --output_dir /data/checkpoints/msk_mri_dataset/logs/logs/rebuttal_march \
  --run_name msk_vit_large_mixed_10pct_random_cf004_acc8_bs2 \
  --train_fraction 0.10 \
  --val_fraction 0.10 \
  --subset_seed 42 \
  --mask_type random \
  --center_fractions 0.04 \
  --accelerations 8 \
  --vit_avrg_img_size 320 \
  --vit_patch_size 10 10 \
  --vit_in_chans 1 \
  --vit_embed_dim 48 \
  --vit_depth 10 \
  --vit_num_heads 16 \
  --lr 0.0005 \
  --batch_size 2 \
  --num_workers 16 \
  --num_epochs 50 \
  --num_checkpoints 1 \
  --model_seed 42 \
  --world_size 1 \
  --val_every_n_epochs 1 \
  --enable_wandb \
  --wandb_project mskmri-rebuttal \
  --wandb_name msk_vit_large_mixed_10pct_random_cf004_acc8_bs2 \
  --wandb_log_images \
  --wandb_num_images 4 \
  --skip_eval
