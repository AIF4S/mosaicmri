#!/bin/bash
source ./runs/inferences/inferences_base_file.sh

SAVE_EVERY=10
GPU=0


DATASET=../../../../../data/datasets/msk_mri_h5/varnet_msk_dataset2/multicoil_val

run_inference "january_exps/varnet_mskmrpro_FULL_casc8_acc8" 27 $GPU 8 || exit $?
# run_inference "january_exps/varnet_mskmrpro_20%_volumes_casc8_acc8" 41 $GPU 8 || exit $?

# DATASET=../../../../../data/datasets/fastMRI/knee/multicoil_val

# run_inference "january_exps/varnet_mskmrpro_10%_volumes_casc8_acc8" 43 $GPU 8 || exit $?
# run_inference "january_exps/varnet_mskmrpro_20%_volumes_casc8_acc8" 41 $GPU 8 || exit $?