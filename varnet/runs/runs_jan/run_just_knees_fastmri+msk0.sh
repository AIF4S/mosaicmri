#!/bin/bash
# ================================
# run_varnet.sh
# ================================

# Run training

CUDA_VISIBLE_DEVICES=0 python train_varnet_paula.py \
  --mode train \
  --data_path1 ../../../../../data/datasets/msk_mri_h5/varnet_msk_dataset2 \
  --data_path2 ../../../../../data/datasets/fastMRI/knee \
  --data_loader knees_only \
  --default_root_dir "../../../../../data/checkpoints/msk_mri_dataset/logs/logs/january_exps/varnet_knees_only_fastmri+msk" \
  --mask_type equispaced_fraction \
  --category_mapping_path ../../../../../data/datasets/msk_mri_h5/file_category_mapping.json \
  --center_fractions 0.08 \
  --accelerations 8 \
  --num_cascades 8 \
  --accelerator gpu \
  --batch_size 1 \
  --num_workers 16 \
  --max_epochs 60 \
  --gpus 1 \
  --strategy ddp \
  --lr 0.0001 \
  --lr_step_size 60 \
  --lr_gamma 0.1 \
  --weight_decay 0.0 \
  --val_source "data2" \
  # --limit_train_batches 100 \
  # --limit_val_batches 500 \