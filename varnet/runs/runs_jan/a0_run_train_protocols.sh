#!/bin/bash
# ================================
# run_varnet.sh
# ================================

# All anatomy categories
parts=(
  # T2

  # T1
  # T1_FS

  # PD
  # PD_FS

  T2_FS
  STIR
)

# Common args for all trainings
common_vars=(
  --data_path ../../../../../data/datasets/msk_mri_h5/varnet_msk_dataset2
  --data_loader categories
  --category_mapping_path ../../../../../data/datasets/msk_mri_h5/file_protocol_mapping.json
  --center_fractions 0.04
  --accelerations 8
  --num_cascades 8
  --mask_type random
  --accelerator gpu
  --batch_size 1
  --num_workers 16
  --max_epochs 50
  --gpus 1
  --strategy ddp
  --lr 0.0001
  --lr_step_size 100
  --lr_gamma 0.1
  --weight_decay 0.0
)

i=0
for part in "${parts[@]}"; do
  echo ">>> Training anatomy: ${part}"
  CUDA_VISIBLE_DEVICES=7 python train_varnet_paula.py \
    "${common_vars[@]}" \
    --default_root_dir "../../../../../data/checkpoints/msk_mri_dataset/logs/logs/january_exps/protocols_full/varnet_msk_protocols_${part}" \
    --category_order "${part}" \
    
  i=$((i + 1))
done
