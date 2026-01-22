#!/bin/bash
# ==============================================
# Train on JSON-selected slices (all -> TRAIN)
# Validate on MSK multicoil_val, KNEE ONLY
# ==============================================

CUDA_VISIBLE_DEVICES=5 python train_varnet_paula.py \
  --mode train \
  --data_loader json_train_knee_val \
  --selector_json_path "/home/berk/MRI/data_filtering_for_accelerated_mri/datasets_custom/train/dreamsim_ensemble_emb_p128_msk_mri_knn_1_msk_mri_knee_val_filter.json" \
  --data_path_train_root ../../../../../data/datasets/msk_mri_h5/varnet_msk_dataset2/multicoil_train \
  --data_path_val_root ../../../../../data/datasets/msk_mri_h5/varnet_msk_dataset2/multicoil_val \
  --category_mapping_path ../../../../../data/datasets/msk_mri_h5/file_category_mapping.json \
  --default_root_dir "../../../../../data/checkpoints/msk_mri_dataset/logs/logs/january_exps/varnet_json_dreamsim_train_slices_knee_val_msk" \
  --mask_type random \
  --center_fractions 0.04 \
  --accelerations 8 \
  --num_cascades 8 \
  --accelerator gpu \
  --batch_size 1 \
  --num_workers 16 \
  --max_epochs 60 \
  --gpus 1 \
  --strategy ddp \
  --lr 0.0001 \
  --lr_step_size 100 \
  --lr_gamma 0.1 \
  --weight_decay 0.0 \
  # --limit_train_batches 100 \
  # --limit_val_batches 500 \
