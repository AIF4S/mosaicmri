#!/bin/bash
MSK_DATASET=../../../../../data/datasets/msk_mri_h5/varnet_msk_dataset2/multicoil_test
FASTMRI_DATASET=../../../../../data/datasets/fastMRI/knee/multicoil_val

###################################
### msk training on MSK dataset and fastmri CASC 8 ACC8
###################################
CUDA_VISIBLE_DEVICES=6 python run_pretrained_varnet_inference.py --data_path $FASTMRI_DATASET \
  --output_path ./outputs/msk_anatomy_order-shift/fastmri_anatomy_elbow_casc8_acc4 \
  --state_dict_file ./logs/varnet_msk_anatomy_elbow_casc8_acc4/epoch=7.ckpt \