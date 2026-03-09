#!/usr/bin/env bash
set -euo pipefail

# Sample inference launcher for MosaicMRI VarNet
CONDA_ENV="mosaic_mri_varnet"

# Example:
# conda activate "$CONDA_ENV"
# python varnet/run_pretrained_varnet_inference.py \
#   --data_path /path/to/multicoil_test \
#   --output_path /path/to/output

echo "Activate env: $CONDA_ENV"
echo "Then run: python varnet/run_pretrained_varnet_inference.py ..."
