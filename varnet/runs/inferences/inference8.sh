#!/bin/bash
SAVE_EVERY=10
GPU=0
accelerations=8

source ./runs/inferences/inferences_base_file.sh

DATASET=../../../../../data/datasets/fastMRI/knee/multicoil_val

# run_inference "january_exps/protocols_full/varnet_msk_protocols_T1_casc8_acc8" 33 $GPU $accelerations || exit $?

# run_inference "january_exps/protocols_full/varnet_msk_protocols_T1_FS_casc8_acc8" 51 $GPU $accelerations || exit $?

run_inference "january_exps/protocols_full/varnet_msk_protocols_T2_casc8_acc8" 44 $GPU $accelerations || exit $?

run_inference "january_exps/protocols_full/varnet_msk_protocols_T2_FS_casc8_acc8" 43 $GPU $accelerations || exit $?

run_inference "january_exps/protocols_full/varnet_msk_protocols_STIR_casc8_acc8" 36 $GPU $accelerations || exit $?

# run_inference "january_exps/anatomies_full/varnet_msk_anatomy_hip_casc8_acc8" 43 $GPU $accelerations || exit $?

# run_inference "january_exps/anatomies_full/varnet_msk_anatomy_ankle_casc8_acc8" 56 $GPU $accelerations || exit $?

# run_inference "january_exps/anatomies_full/varnet_msk_anatomy_foot_casc8_acc8" 23 $GPU $accelerations || exit $?



# run_inference "january_exps/anatomies_full/varnet_msk_anatomy_leg_tibfib_casc8_acc8" 43 $GPU $accelerations || exit $?

# run_inference "january_exps/anatomies_full/varnet_msk_anatomy_shoulder_casc8_acc8" 48 $GPU $accelerations || exit $?

# run_inference "january_exps/anatomies_full/varnet_msk_anatomy_pelvis_pubalgia_casc8_acc8" 48 $GPU $accelerations || exit $?

# run_inference "january_exps/anatomies_full/varnet_msk_anatomy_elbow_casc8_acc8" 23 $GPU $accelerations || exit $?

# run_inference "january_exps/anatomies_full/varnet_msk_anatomy_wrist_hand_thumb_casc8_acc8" 23 $GPU $accelerations || exit $?

# run_inference "january_exps/anatomies_full/varnet_msk_anatomy_spine_casc8_acc8" 39 $GPU $accelerations || exit $?


