#!/bin/bash
SAVE_EVERY=10
GPU=4
accelerations=8

source ./runs/inferences/inferences_base_file.sh

DATASET=../../../../../data/datasets/msk_mri_h5/varnet_msk_dataset2/multicoil_test

#run_inference "january_exps/protocols_full/varnet_msk_protocols_PD_casc8_acc8" 59 $GPU $accelerations || exit $?

run_inference "january_exps/protocols_full/varnet_msk_protocols_PD_FS_casc8_acc8" 43 $GPU $accelerations || exit $?

run_inference "january_exps/anatomies_full/varnet_msk_anatomy_shoulder_casc8_acc8" 48 $GPU $accelerations || exit $?

run_inference "january_exps/anatomies_full/varnet_msk_anatomy_pelvis_pubalgia_casc8_acc8" 48 $GPU $accelerations || exit $?

run_inference "january_exps/anatomies_full/varnet_msk_anatomy_elbow_casc8_acc8" 23 $GPU $accelerations || exit $?

