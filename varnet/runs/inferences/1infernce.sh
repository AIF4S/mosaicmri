#!/bin/bash
MSK_DATASET=../../../../../data/datasets/msk_mri_h5/varnet_msk_dataset2/multicoil_test
FASTMRI_DATASET=../../../../../data/datasets/fastMRI/knee/multicoil_val

###################################
### finetune fastmri on MSK on MSK dataset and fastmri CASC 8 ACC4
##################################
CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
  --output_path ./outputs/finetune_fastmri-msk/msk_casc8_acc4_16 \
  --state_dict_file ./logs/varnet_mskmrpro_finetune_casc8_acc4/epoch=16.ckpt \
