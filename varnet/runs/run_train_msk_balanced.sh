#!/bin/bash
# ================================
# run_varnet.sh
# ================================

CUDA_VISIBLE_DEVICES=1 python train_varnet_paula.py \
  --mode train \
  --data_loader balanced_categories \
  --data_path ../../../../../data/datasets/msk_mri_h5/varnet_msk_dataset2 \
  --category_mapping_path ../../../../../data/datasets/msk_mri_h5/file_category_mapping.json \
  --category_order scapula_ribs_femur pelvis_pubalgia hip shoulder leg_tibfib knee spine elbow ankle foot wrist_hand_thumb \
  --volume_rates 1.0 0.7 0.14 0.095 0.8 0.06077 0.033 0.7 0.15827 0.35 0.5 \
  --mask_type equispaced_fraction \
  --default_root_dir "./logs/varnet_balanced_categories" \
  --center_fractions 0.08 \
  --accelerations 4 \
  --num_cascades 8 \
  --accelerator gpu \
  --batch_size 1 \
  --num_workers 16 \
  --max_epochs 15 \
  --gpus 1 \
  --strategy ddp \
  --lr 0.0001 \
  --lr_step_size 40 \
  --lr_gamma 0.1 \
  --weight_decay 0.0 \
