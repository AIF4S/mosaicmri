#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU_LIST="4,5,6,7"
PER_GPU_BATCH=4
WORLD_SIZE="$(awk -F',' '{print NF}' <<< "$GPU_LIST")"
GLOBAL_BATCH="$((PER_GPU_BATCH * WORLD_SIZE))"

CUDA_VISIBLE_DEVICES="$GPU_LIST" python run_end2end_baseline_direct.py \
  --model vit \
  --train_h5_dir /data/datasets/msk_mri_h5/varnet_msk_dataset2/multicoil_train \
  --eval_h5_dir /data/datasets/msk_mri_h5/varnet_msk_dataset2/multicoil_val \
  --category_mapping_path /data/datasets/msk_mri_h5/file_category_mapping.json \
  --val_anatomy KNEE \
  --train_slice_limit 10000 \
  --selection_seed 24 \
  --output_dir /data/checkpoints/msk_mri_dataset/logs/logs/rebuttal_march \
  --run_name r2_vit_mixed_anatomy_10k_knee_val \
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
  --batch_size "$GLOBAL_BATCH" \
  --num_workers 16 \
  --num_epochs 50 \
  --num_checkpoints 1 \
  --model_seed 24 \
  --world_size "$WORLD_SIZE" \
  --val_every_n_epochs 1 \
  --enable_wandb \
  --wandb_project mskmri-rebuttal \
  --wandb_name r2_vit_mixed_anatomy_10k_knee_val \
  --wandb_log_images \
  --wandb_num_images 4 \
  --skip_eval
