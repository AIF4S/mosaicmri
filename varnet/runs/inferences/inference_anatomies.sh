#!/bin/bash

DATASET=../../../../../data/datasets/msk_mri_h5/varnet_msk_dataset2/multicoil_val
SAVE_EVERY=10
GPU=0

source ./runs/inferences/inferences_base_file.sh

#1

#2
run_inference "january_exps/anatomies_full/varnet_msk_anatomy_pelvis_pubalgia_casc8_acc8" 43 $GPU || exit $?
#3
run_inference "january_exps/anatomies_full/varnet_msk_anatomy_hip_casc8_acc8" 43 $GPU || exit $?
#4
run_inference "january_exps/anatomies_full/varnet_msk_anatomy_shoulder_casc8_acc8" 43 $GPU || exit $?
#5
run_inference "january_exps/anatomies_full/varnet_msk_anatomy_leg_tibfib_casc8_acc8" 43 $GPU || exit $?
#6
run_inference "january_exps/anatomies_full/varnet_msk_anatomy_knee_casc8_acc8" 43 $GPU || exit $?
#7
run_inference "january_exps/anatomies_full/varnet_msk_anatomy_elbow_casc8_acc8" 43 $GPU || exit $?
#8
run_inference "january_exps/anatomies_full/varnet_msk_anatomy_ankle_casc8_acc8" 43 $GPU || exit $?
#9
run_inference "january_exps/anatomies_full/varnet_msk_anatomy_foot_casc8_acc8" 43 $GPU || exit $?
#10
run_inference "january_exps/anatomies_full/varnet_msk_anatomy_wrist_hand_thumb_casc8_acc8" 43 $GPU || exit $?
#11
run_inference "january_exps/anatomies_full/varnet_msk_anatomy_spine_casc8_acc8" 43 $GPU || exit $?

run_inference "january_exps/anatomies_full/varnet_msk_anatomy_scapula_ribs_femur_casc8_acc8" 43 $GPU || exit $?
