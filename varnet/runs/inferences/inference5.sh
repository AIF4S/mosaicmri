#!/bin/bash
SAVE_EVERY=10
GPU=5
accelerations=8

source ./runs/inferences/inferences_base_file.sh

DATASET=../../../../../data/datasets/msk_mri_h5/varnet_msk_dataset2/multicoil_test

run_inference "january_exps/protocols_full/varnet_msk_protocols_STIR_casc8_acc8" 36 $GPU $accelerations || exit $?

run_inference "january_exps/anatomies_full/varnet_msk_anatomy_wrist_hand_thumb_casc8_acc8" 23 $GPU $accelerations || exit $?

run_inference "january_exps/anatomies_full/varnet_msk_anatomy_spine_casc8_acc8" 39 $GPU $accelerations || exit $?

DATASET=../../../../../data/datasets/fastMRI/knee/multicoil_val

run_inference "january_exps/anatomies_full/varnet_msk_anatomy_scapula_ribs_femur_casc8_acc8" 49 $GPU $accelerations || exit $?

run_inference "january_exps/anatomies_full/varnet_msk_anatomy_knee_casc8_acc8" 39 $GPU $accelerations || exit $?
