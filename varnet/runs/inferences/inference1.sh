#!/bin/bash
SAVE_EVERY=10
GPU=1
accelerations=8

source ./runs/inferences/inferences_base_file.sh

DATASET=../../../../../data/datasets/msk_mri_h5/varnet_msk_dataset2/multicoil_test

#run_inference "january_exps/protocols_full/varnet_msk_protocols_T2_casc8_acc8" 44 $GPU $accelerations || exit $?

run_inference "january_exps/protocols_full/varnet_msk_protocols_T2_FS_casc8_acc8" 33 $GPU $accelerations || exit $?

run_inference "january_exps/anatomies_full/varnet_msk_anatomy_ankle_casc8_acc8" 56 $GPU $accelerations || exit $?

run_inference "january_exps/anatomies_full/varnet_msk_anatomy_foot_casc8_acc8" 23 $GPU $accelerations || exit $?

run_inference "january_exps/anatomies_full/varnet_msk_anatomy_leg_tibfib_casc8_acc8" 43 $GPU $accelerations || exit $?
