#!/bin/bash
# ================================
# run_varnet.sh
# ================================

# Run training
CUDA_VISIBLE_DEVICES=4 python train_varnet_paula.py \
  --data_path ../../../../../data/datasets/msk_mri_h5/varnet_msk_dataset2 \
  --default_root_dir "../../../../../data/checkpoints/msk_mri_dataset/logs/logs/january_exps/varnet_mskmrpro_50%_volumes" \
  --mask_type random \
  --center_fractions 0.04 \
  --accelerations 8 \
  --num_cascades 8 \
  --accelerator gpu \
  --batch_size 1 \
  --num_workers 16 \
  --max_epochs 50 \
  --gpus 1 \
  --strategy ddp \
  --lr 0.0001 \
  --lr_step_size 100 \
  --lr_gamma 0.1 \
  --weight_decay 0.0 \
  --volume_sample_rate 0.5 \
  --val_volume_sample_rate 0.5 \
  --test_volume_sample_rate 0.5 \