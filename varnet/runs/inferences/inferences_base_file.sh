#!/bin/bash

LOGS_PATH="../../../../../data/checkpoints/msk_mri_dataset/logs/logs/"

find_ckpt() {
  local exp_name="$1"
  local epoch="$2"
  local exp_dir="$LOGS_PATH/$exp_name"

  if [[ ! -d "$exp_dir" ]]; then
    echo "ERROR: experiment directory not found: $exp_dir" >&2
    return 2
  fi

  # Prefer the checkpoint directly under the experiment directory.
  # If not present, fall back to searching subfolders (e.g., version_*).
  local ckpt=""
  if [[ -f "$exp_dir/epoch=$epoch.ckpt" ]]; then
    ckpt="$exp_dir/epoch=$epoch.ckpt"
  else
    ckpt=$(find "$exp_dir" -type f -name "epoch=$epoch.ckpt" -print -quit 2>/dev/null)
    if [[ -z "$ckpt" ]]; then
      ckpt=$(find "$exp_dir" -type f -name "epoch=$epoch*.ckpt" -print -quit 2>/dev/null)
    fi
  fi

  if [[ -z "$ckpt" ]]; then
    echo "ERROR: checkpoint not found for exp='$exp_name' epoch=$epoch under: $exp_dir" >&2
    return 3
  fi

  printf '%s' "$ckpt"
}


run_inference() {
  # Function to run inference for a given experiment and epoch.
  # Arguments:
  #   $1: Experiment name.
  #   $2: Epoch number.

  local exp_name="$1"
  local epoch="$2"
  local gpu="$3"
  local accelerations="$4"

  if [[ -z "${gpu}" ]]; then
    echo "ERROR: gpu id not provided to run_inference(exp, epoch, gpu)" >&2
    return 4
  fi
  if [[ ! "${gpu}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "ERROR: gpu must be an integer (e.g., 0) or a comma-separated list (e.g., 0,1). Got: '${gpu}'" >&2
    return 5
  fi

  # Find the checkpoint path for the given experiment and epoch.
  local ckpt_path
  ckpt_path=$(find_ckpt "$exp_name" "$epoch") || return $?

  # Run the inference script with the specified parameters.
  export CUDA_VISIBLE_DEVICES="${gpu}"
  if [[ "${DEBUG_GPU:-0}" == "1" ]]; then
    echo "[DEBUG] exp=${exp_name} epoch=${epoch} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >&2 || true
  fi

  python run_pretrained_varnet_inference.py --data_path "$DATASET" --save_every "$SAVE_EVERY" \
    --output_path ./outputs/$exp_name \
    --state_dict_file "$ckpt_path" \
    --accelerations "$accelerations"
}