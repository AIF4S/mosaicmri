#!/usr/bin/env bash
set -euo pipefail

# Sample training launcher for MosaicMRI VarNet
CONDA_ENV="mosaic_mri_varnet"

# Example:
# conda activate "$CONDA_ENV"
# python varnet/train_mosaic_mri_varnet.py \
#   --data_path /path/to/multicoil_train \
#   --default_root_dir /path/to/checkpoints

echo "Activate env: $CONDA_ENV"
echo "Then run: python varnet/train_mosaic_mri_varnet.py ..."
