#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

RUN_NAME="msk_unet_large_mixed_10pct_random_cf004_acc8"
CKPT_DIR="/data/checkpoints/msk_mri_dataset/logs/logs/rebuttal_march/${RUN_NAME}/train/checkpoints"
EPOCH="65"

if [[ "$EPOCH" == "latest" ]]; then
  CKPT_FILE="$(ls -1v "${CKPT_DIR}"/epoch_*.pt | tail -n 1)"
else
  CKPT_FILE="${CKPT_DIR}/epoch_${EPOCH}.pt"
fi

if [[ ! -f "$CKPT_FILE" ]]; then
  echo "Checkpoint not found: $CKPT_FILE" >&2
  exit 1
fi

echo "Using checkpoint: $CKPT_FILE"

CUDA_VISIBLE_DEVICES=6 python run_end2end_baseline_inference_rss_only_h5.py \
  --model u-net \
  --checkpoint_file "$CKPT_FILE" \
  --data_path /data/datasets/msk_mri_h5/varnet_msk_dataset2/multicoil_test_full \
  --output_path "${REPO_DIR}/output/${RUN_NAME}/inference_full_test_metrics_png" \
  --device cuda \
  --num_workers 16 \
  --num_png_samples 24 \
  --unet_chans 128 \
  --unet_num_pools 4
