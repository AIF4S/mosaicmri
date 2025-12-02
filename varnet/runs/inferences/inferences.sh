#!/bin/bash
###################################
### msk training on MSK dataset and fastmri CASC 8 ACC4
##################################
MSK_DATASET=../../../../../data/datasets/msk_mri_h5/varnet_msk_dataset2/multicoil_test
FASTMRI_DATASET=../../../../../data/datasets/fastMRI/knee/multicoil_val

CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
  --output_path ./outputs/msk_baseline/msk_casc8_acc4 \
  --state_dict_file ./logs/varnet_msk_baseline_casc8_acc4/epoch=8.ckpt \

CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $FASTMRI_DATASET \
  --output_path ./outputs/msk_baseline/fastmri_casc8_acc4 \
  --state_dict_file ./logs/varnet_msk_baseline_casc8_acc4/epoch=8.ckpt \

###################################
### msk training on MSK dataset and fastmri CASC 8 ACC4 50% of slices
##################################
CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
  --output_path ./outputs/msk_half_data/msk_50%_slices_casc8_acc4 \
  --state_dict_file ./logs/varnet_mskmrpro_50%_slices_casc8_acc4/epoch=8.ckpt \

CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $FASTMRI_DATASET \
  --output_path ./outputs/msk_half_data/fastmri_50%_slices_casc8_acc4 \
  --state_dict_file ./logs/varnet_mskmrpro_50%_slices_casc8_acc4/epoch=8.ckpt \

###################################
### msk training on MSK dataset and fastmri CASC 8 ACC4 50% of volumes
###################################
CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
  --output_path ./outputs/msk_half_data/msk_50%_volumes_casc8_acc4 \
  --state_dict_file ./logs/varnet_mskmrpro_50%_volumes_casc8_acc4/epoch=8.ckpt \

CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $FASTMRI_DATASET \
  --output_path ./outputs/msk_half_data/fastmri_50%_volumes_casc8_acc4 \
  --state_dict_file ./logs/varnet_mskmrpro_50%_volumes_casc8_acc4/epoch=8.ckpt \

###################################
### msk training on MSK dataset and fastmri CASC 8 ACC8
###################################
CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
  --output_path ./outputs/msk_baseline/msk_casc8_acc8 \
  --state_dict_file ./logs/varnet_msk_baseline_casc8_acc8/epoch=8.ckpt \

CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $FASTMRI_DATASET \
  --output_path ./outputs/msk_baseline/fastmri_casc8_acc8 \
  --state_dict_file ./logs/varnet_msk_baseline_casc8_acc8/epoch=8.ckpt \

###################################
### msk training on MSK dataset and fastmri CASC 12 ACC4
###################################
CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
  --output_path ./outputs/msk_baseline/msk_casc12_acc4 \
  --state_dict_file ./logs/varnet_msk_baseline_casc12_acc4/epoch=8.ckpt \

CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $FASTMRI_DATASET \
  --output_path ./outputs/msk_baseline/fastmri_casc12_acc4 \
  --state_dict_file ./logs/varnet_msk_baseline_casc12_acc4/epoch=8.ckpt \

###################################
### msk training on MSK dataset anatomy order CASC 8 ACC4
###################################
CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
  --output_path ./outputs/msk_anatomy_order-shift/msk_easy_to_hard_casc8_acc4 \
  --state_dict_file ./logs/varnet_msk_easy_to_hard_casc8_acc4/epoch=8.ckpt \

###################################
### msk training on MSK dataset anatomy SHIFT CASC 8 ACC4
###################################
CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
  --output_path ./outputs/msk_anatomy_order-shift/msk_anatomy_ankle_casc8_acc4 \
  --state_dict_file ./logs/varnet_msk_anatomy_ankle_casc8_acc4/epoch=8.ckpt \

CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
  --output_path ./outputs/msk_anatomy_order-shift/msk_anatomy_foot_casc8_acc4 \
  --state_dict_file ./logs/varnet_msk_anatomy_foot_casc8_acc4/epoch=8.ckpt \

CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
  --output_path ./outputs/msk_anatomy_order-shift/msk_anatomy_hip_casc8_acc4 \
  --state_dict_file ./logs/varnet_msk_anatomy_hip_casc8_acc4/epoch=8.ckpt \

CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
  --output_path ./outputs/msk_anatomy_order-shift/msk_anatomy_knee_casc8_acc4 \
  --state_dict_file ./logs/varnet_msk_anatomy_knee_casc8_acc4/epoch=8.ckpt \

CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
  --output_path ./outputs/msk_anatomy_order-shift/msk_anatomy_leg_tibfib_casc8_acc4 \
  --state_dict_file ./logs/varnet_msk_anatomy_leg_tibfib_casc8_acc4/epoch=8.ckpt \

CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
  --output_path ./outputs/msk_anatomy_order-shift/msk_anatomy_pelvis_pubalgia_casc8_acc4 \
  --state_dict_file ./logs/varnet_msk_anatomy_pelvis_pubalgia_casc8_acc4/epoch=8.ckpt \

CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
  --output_path ./outputs/msk_anatomy_order-shift/msk_anatomy_scapula_ribs_femur_casc8_acc4 \
  --state_dict_file ./logs/varnet_msk_anatomy_scapula_ribs_femur_casc8_acc4/epoch=8.ckpt \

CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
  --output_path ./outputs/msk_anatomy_order-shift/msk_anatomy_shoulder_casc8_acc4 \
  --state_dict_file ./logs/varnet_msk_anatomy_shoulder_casc8_acc4/epoch=8.ckpt \

CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
  --output_path ./outputs/msk_anatomy_order-shift/msk_anatomy_spine_casc8_acc4 \
  --state_dict_file ./logs/varnet_msk_anatomy_spine_casc8_acc4/epoch=8.ckpt \

###################################
### fastmri training on MSK dataset and fastmri CASC 8 ACC4
##################################
CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
  --output_path ./outputs/fastmri_baseline/msk_casc8_acc4 \
  --state_dict_file ./logs/varnet_msk_baseline_casc8_acc4/epoch=8.ckpt \

CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $FASTMRI_DATASET \
  --output_path ./outputs/fastmri_baseline/fastmri_casc8_acc4 \
  --state_dict_file ./logs/varnet_msk_baseline_casc8_acc4/epoch=8.ckpt \

###################################
### fastmri + MSK training on MSK dataset and fastmri CASC 8 ACC4
##################################
CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
  --output_path ./outputs/msk+fastmri/msk_casc8_acc4 \
  --state_dict_file ./logs/varnet_msk+fastmri_casc8_acc4/epoch=8.ckpt \

CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $FASTMRI_DATASET \
  --output_path ./outputs/msk+fastmri/fastmri_casc8_acc4 \
  --state_dict_file ./logs/varnet_msk+fastmri_casc8_acc4/epoch=8.ckpt \

###################################
### finetune fastmri on MSK on MSK dataset and fastmri CASC 8 ACC4
##################################
CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $MSK_DATASET \
  --output_path ./outputs/finetune_fastmri-msk/msk_casc8_acc4 \
  --state_dict_file ./logs/varnet_mskmrpro_finetune_casc8_acc4/epoch=8.ckpt \

CUDA_VISIBLE_DEVICES=0 python run_pretrained_varnet_inference.py --data_path $FASTMRI_DATASET \
  --output_path ./outputs/finetune_fastmri-msk/fastmri_casc8_acc4 \
  --state_dict_file ./logs/varnet_mskmrpro_finetune_casc8_acc4/epoch=8.ckpt \