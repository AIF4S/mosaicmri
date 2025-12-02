#!/bin/bash
# ================================
# run_varnet.sh
# ================================

# Run training
CUDA_VISIBLE_DEVICES=2 python train_varnet_per_category.py \
  --mode train \
  --data_path ../../../../../data/datasets/msk_mri_h5/varnet_msk_dataset2 \
  --default_root_dir "./logs/varnet_msk_easy_to_hard" \
  --category_mapping_path ../../../../../data/datasets/msk_mri_h5/file_category_mapping.json \
  --category_order PELVIS_PUBALGIA HIP SHOULDER LEG_TIBFIB SCAPULA_RIBS_FEMUR KNEE SPINE ELBOW ANKLE FOOT WRIST_HAND_THUMB \
  --mask_type equispaced_fraction \
  --center_fractions 0.08 \
  --challenge multicoil \
  --accelerations 4 \
  --num_cascades 8 \
  --batch_size 1 \
  --num_workers 16 \
  --gpus 1 \
  --strategy ddp \
  --lr 0.0001 \
  --lr_step_size 40 \
  --lr_gamma 0.1 \
  --weight_decay 0.0 \
  --epochs_per_category 8 \