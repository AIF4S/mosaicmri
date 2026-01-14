#!/bin/bash

DATASET=../../../../../data/datasets/msk_mri_h5/varnet_msk_dataset2/multicoil_val
SAVE_EVERY=10
GPU=0

source ./runs/inferences/inferences_base_file.sh

run_inference "january_exps/anatomies_full/varnet_msk_anatomy_elbow_casc8_acc8" 23 $GPU || exit $?

# run_inference "january_exps/anatomies_full/varnet_msk_anatomy_pelvis_pubalgia_casc8_acc8" 48 $GPU || exit $?

# run_inference "january_exps/anatomies_full/varnet_msk_anatomy_wrist_hand_thumb_casc8_acc8" 23 $GPU || exit $?

DATASET=../../../../../data/datasets/fastMRI/knee/multicoil_val

run_inference "january_exps/anatomies_full/varnet_msk_anatomy_elbow_casc8_acc8" 23 $GPU || exit $?

# run_inference "january_exps/anatomies_full/varnet_msk_anatomy_pelvis_pubalgia_casc8_acc8" 48 $GPU || exit $?

# run_inference "january_exps/anatomies_full/varnet_msk_anatomy_shoulder_casc8_acc8" 48 $GPU || exit $?

# run_inference "january_exps/anatomies_full/varnet_msk_anatomy_wrist_hand_thumb_casc8_acc8" 23 $GPU || exit $?