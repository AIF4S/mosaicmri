#!/bin/bash
# ================================
# run_varnet.sh
# ================================

# Run training
python run_pretrained_varnet_inference.py \
  --challenge varnet_brain_mc \
  --device cuda \
  --output_path ./outputs/varnet_inference_debug \
  --data_path ../../../../../data/datasets/msk_mri_h5/test2 \
  #--data_path ./fast_mri_samples \