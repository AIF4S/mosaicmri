#!/bin/bash

DATASET=../../../../../data/datasets/msk_mri_h5/varnet_msk_dataset2/multicoil_test
SAVE_EVERY=10
GPU=1

source ./runs/inferences/inferences_base_file.sh

run_inference "anatomies_full/varnet_msk_anatomy_pelvis_pubalgia_casc8_acc8" 7 "$DATASET" "$SAVE_EVERY" $GPU || exit $?

run_inference "anatomies_full/varnet_msk_anatomy_hip_casc8_acc8" 7 "$DATASET" "$SAVE_EVERY" $GPU || exit $?

run_inference "anatomies_full/varnet_msk_anatomy_shoulder_casc8_acc8" 7 "$DATASET" "$SAVE_EVERY" $GPU || exit $?

run_inference "anatomies_full/varnet_msk_anatomy_leg_tibfib_casc8_acc8" 7 "$DATASET" "$SAVE_EVERY" $GPU || exit $?

run_inference "anatomies_full/varnet_msk_anatomy_knee_casc8_acc8" 7 "$DATASET" "$SAVE_EVERY" $GPU || exit $?

run_inference "anatomies_full/varnet_msk_anatomy_elbow_casc8_acc8" 7 "$DATASET" "$SAVE_EVERY" $GPU || exit $?

run_inference "anatomies_full/varnet_msk_anatomy_ankle_casc8_acc8" 7 "$DATASET" "$SAVE_EVERY" $GPU || exit $?

run_inference "anatomies_full/varnet_msk_anatomy_foot_casc8_acc8" 7 "$DATASET" "$SAVE_EVERY" $GPU || exit $?

run_inference "anatomies_full/varnet_msk_anatomy_wrist_hand_thumb_casc8_acc8" 7 "$DATASET" "$SAVE_EVERY" $GPU || exit $?

run_inference "anatomies_full/varnet_msk_anatomy_spine_casc8_acc8" 7 "$DATASET" "$SAVE_EVERY" $GPU || exit $?

run_inference "anatomies_full/varnet_msk_anatomy_scapula_ribs_femur_casc8_acc8" 7 "$DATASET" "$SAVE_EVERY" $GPU || exit $?
