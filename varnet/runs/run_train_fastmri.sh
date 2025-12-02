#!/bin/bash
# ================================
# run_varnet.sh
# ================================

# Run training
CUDA_VISIBLE_DEVICES=7 python train_varnet_paula.py \
  --mode train \
  --data_path ../../../../../data/datasets/fastMRI/knee \
  --default_root_dir "./logs/varnet_fastmri" \
  --mask_type equispaced_fraction \
  --center_fractions 0.08 \
  --accelerations 4 \
  --num_cascades 8 \
  --accelerator gpu \
  --batch_size 1 \
  --num_workers 16 \
  --max_epochs 50 \
  --gpus 1 \
  --strategy ddp \
  --lr 0.0001 \
  --lr_step_size 40 \
  --lr_gamma 0.1 \
  --weight_decay 0.0 \
  --volume_sample_rate 0.3 \
  --val_volume_sample_rate 0.5 \
  --test_volume_sample_rate 0.5 \
  --debug \
  # --limit_train_batches 100 \
  # --limit_val_batches 200 \ # <- for debugging