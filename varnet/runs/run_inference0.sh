#!/bin/bash
# ================================
# run_varnet.sh
# ================================

# Run training
python run_pretrained_varnet_inference.py \
  --challenge varnet_knee_mc \
  --device cuda \
  --output_path ./outputs/varnet_inference_msk_training \
  --state_dict_file ./logs/varnet_msk_mrpro2/checkpoints/epoch=4-step=249905.ckpt \
  --data_path ../../../../../data/datasets/msk_mri_h5/varnet_msk_dataset3/multicoil_test \
  #--data_path ./fast_mri_samples \