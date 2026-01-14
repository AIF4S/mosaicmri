#!/bin/bash
# ================================
# run_varnet.sh
# ================================

# All anatomy categories
parts=(
  scapula_ribs_femur
  pelvis_pubalgia
  hip
  shoulder
  leg_tibfib
  knee
  spine
  elbow
  ankle
  foot
  wrist_hand_thumb
)

# Common args for all trainings
common_vars=(
  --mode train
  --data_path ../../../../../data/datasets/msk_mri_h5/varnet_msk_dataset2
  --data_loader categories
  --category_mapping_path ../../../../../data/datasets/msk_mri_h5/file_category_mapping.json
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
VOLUME_RATES=(
  1.00000   # scapula_ribs_femur (22)
  0.7   # pelvis_pubalgia (34)
  0.14   # hip (202)
  0.095   # shoulder (373)
  0.8   # leg_tibfib (28)
  0.06077   # knee (362)
  0.033   # spine (1304)
  0.7   # elbow (36)
  0.15827   # ankle (139)
  0.35   # foot (84)
  0.5   # wrist_hand_thumb (69)
)
# Loop over anatomies
i=0
for part in "${parts[@]}"; do
  echo ">>> Training anatomy: ${part}"
  CUDA_VISIBLE_DEVICES=4 python train_varnet_paula.py \
    "${common_vars[@]}" \
    --default_root_dir "./logs/varnet_msk_balanced_anatomy_${part}" \
    --category_order "${part}" \
    --volume_sample_rate "${VOLUME_RATES[i]}" \
    --val_volume_sample_rate "${VOLUME_RATES[i]}" \
    --test_volume_sample_rate "${VOLUME_RATES[i]}" \

  i=$((i + 1))
done
