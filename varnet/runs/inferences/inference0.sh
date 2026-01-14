#!/bin/bash
SAVE_EVERY=10
GPU=0
accelerations=8

source ./runs/inferences/inferences_base_file.sh

DATASET=../../../../../data/datasets/msk_mri_h5/varnet_msk_dataset2/multicoil_test


run_inference "january_exps/varnet_mskmrpro_FULL_casc8_acc8" 27 $GPU 8 || exit $?

# run_inference "january_exps/varnet_mskmrpro_10%_volumes_casc8_acc8" 43 $GPU 8 || exit $?
# run_inference "january_exps/varnet_mskmrpro_20%_volumes_casc8_acc8" 41 $GPU 8 || exit $?

# run_inference "january_exps/protocols_full/varnet_msk_protocols_T1_casc8_acc8" 33 $GPU $accelerations || exit $?

# run_inference "january_exps/protocols_full/varnet_msk_protocols_T1_FS_casc8_acc8" 51 $GPU $accelerations || exit $?

# run_inference "january_exps/anatomies_full/varnet_msk_anatomy_scapula_ribs_femur_casc8_acc8" 49 $GPU $accelerations || exit $?

# run_inference "january_exps/anatomies_full/varnet_msk_anatomy_knee_casc8_acc8" 39 $GPU $accelerations || exit $?

# run_inference "january_exps/anatomies_full/varnet_msk_anatomy_hip_casc8_acc8" 43 $GPU $accelerations || exit $?