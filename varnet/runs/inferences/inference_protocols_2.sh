#!/bin/bash
MSK_DATASET=../../../../../data/datasets/msk_mri_h5/varnet_msk_dataset2/multicoil_test
FASTMRI_DATASET=../../../../../data/datasets/fastMRI/knee/multicoil_val

LOGS_PATH="../../../../../data/checkpoints/msk_mri_dataset/logs/logs/"

###################################
# ### finetune fastmri on MSK on MSK dataset and fastmri CASC 8 ACC4
# ##################################
# CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
#   --output_path ./outputs/protocols/PD_casc8_acc8 \
#   --state_dict_file $LOGS_PATH/protocols/varnet_msk_protocols_PD_casc8_acc8/epoch=14.ckpt \

# CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
#   --output_path ./outputs/protocols/PD_FS_casc8_acc8 \
#   --state_dict_file $LOGS_PATH/protocols/varnet_msk_protocols_PD_FS_casc8_acc8/epoch=14.ckpt \

# CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
#   --output_path ./outputs/protocols/T1_casc8_acc8 \
#   --state_dict_file $LOGS_PATH/protocols/varnet_msk_protocols_T1_casc8_acc8/epoch=14.ckpt \

# CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
#   --output_path ./outputs/protocols/T1_FS_casc8_acc8 \
#   --state_dict_file $LOGS_PATH/protocols/varnet_msk_protocols_T1_FS_casc8_acc8/epoch=14.ckpt \

# CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
#   --output_path ./outputs/protocols/T2_casc8_acc8 \
#   --state_dict_file $LOGS_PATH/protocols/varnet_msk_protocols_T2_casc8_acc8/epoch=14.ckpt \

CUDA_VISIBLE_DEVICES=3 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
  --output_path ./outputs/protocols/T2_FS_casc8_acc8 \
  --state_dict_file $LOGS_PATH/protocols/varnet_msk_protocols_T2_FS_casc8_acc8/epoch=14.ckpt \

CUDA_VISIBLE_DEVICES=3 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
  --output_path ./outputs/protocols/STIR_casc8_acc8 \
  --state_dict_file $LOGS_PATH/protocols/varnet_msk_protocols_STIR_casc8_acc8/epoch=14.ckpt \

