#!/bin/bash
# ================================
# run_varnet.sh
# ================================

# All anatomy categories
parts=(
  T1
  T1_FS
  T2
  T2_FS
  PD
  PD_FS
  STIR
)

# Common args for all trainings
common_vars=(
  --mode train
  --data_path ../../../../../data/datasets/msk_mri_h5/varnet_msk_dataset2
  --data_loader categories
  --category_mapping_path ../../../../../data/datasets/msk_mri_h5/file_protocol_mapping.json
  --mask_type equispaced_fraction
  --center_fractions 0.08
  --accelerations 8
  --num_cascades 8
  --accelerator gpu
  --batch_size 1
  --num_workers 16
  --max_epochs 15
  --gpus 1
  --strategy ddp
  --lr 0.0001
  --lr_step_size 40
  --lr_gamma 0.1
  --weight_decay 0.0
)
VOLUME_RATES=(0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5)
# Loop over anatomies
i=0
for part in "${parts[@]}"; do
  echo ">>> Training anatomy: ${part}"
  CUDA_VISIBLE_DEVICES=4 python train_varnet_paula.py \
    "${common_vars[@]}" \
    --default_root_dir ""../../../../../data/checkpoints/msk_mri_dataset/logs/logs/anatomies_full/varnet_msk_protocols_half_${part}" \
    --category_order "${part}" \
    --sample_rate "${VOLUME_RATES[i]}" \
    --val_sample_rate "${VOLUME_RATES[i]}" \
    --test_sample_rate "${VOLUME_RATES[i]}" \
    
  i=$((i + 1))
done
