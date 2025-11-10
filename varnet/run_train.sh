#!/bin/bash
# ================================
# run_varnet.sh
# ================================

# Run training
python MRI/varnet/train_varnet_paula.py \
  --mode train \
  --data_path ../../../../../data/datasets/fastMRI/knee/multicoil_val \
  --default_root_dir "./logs/varnet" \
  --mask_type equispaced_fraction \
  --center_fractions 0.08 \
  --accelerations 4 \
  --accelerator gpu \
  --batch_size 1 \
  --num_workers 16 \
  --max_epochs 50 \
  --gpus 8 \
  --strategy ddp \
  --lr 0.001 \
  --lr_step_size 40 \
  --lr_gamma 0.1 \
  --weight_decay 0.0 \
  # --limit_train_batches 100 \
  # --limit_val_batches 200 \ # <- for debugging