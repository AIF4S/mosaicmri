#!/bin/bash
# ================================
# run_varnet.sh
# ================================

# All anatomy categories
parts=(  
  # scapula_ribs_femur
  # spine

  # knee
  # leg_tibfib

  # shoulder
  # elbow

  # hip
  # pelvis_pubalgia 
   
  ankle
  foot
  wrist_hand_thumb
)

# Common args for all trainings
common_vars=(
  --data_path ../../../../../data/datasets/msk_mri_h5/varnet_msk_dataset2
  --data_loader categories
  --category_mapping_path ../../../../../data/datasets/msk_mri_h5/file_category_mapping.json
  --mask_type random
  --center_fractions 0.04
  --accelerations 8
  --num_cascades 8
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
  --seed 24
)
# Loop over anatomies
i=0
for part in "${parts[@]}"; do
  echo ">>> Training anatomy: ${part}"
  CUDA_VISIBLE_DEVICES=4 python train_varnet_paula.py \
    "${common_vars[@]}" \
    --default_root_dir "../../../../../data/checkpoints/msk_mri_dataset/logs/logs/january_exps/anatomies_full/varnet_msk_anatomy_${part}" \
    --category_order "${part}" \

  i=$((i + 1))
done
