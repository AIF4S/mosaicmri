#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# ---------------------- CONFIG ----------------------

# Resolve script dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parameter map base name (no extension)
FILE_NAME="parameter_maps/IsmrmrdParameterMap_Siemens_debug"
PARAM_XML="${SCRIPT_DIR}/${FILE_NAME}.xml"
PARAM_XSL="${SCRIPT_DIR}/${FILE_NAME}.xsl"

# Base paths (go two levels up)
cd ~/../..
PARENT_DIR="$(pwd)"

DATA_PATH="${PARENT_DIR}/data/datasets/msk_original"
OUT_DIR="${PARENT_DIR}/data/datasets/msk_mri_h5"

# Parallelism: default to # of cores; override with JOBS env
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null)}"
JOBS="${JOBS:-4}"

# Optional HDF5 stability tweaks (safe to leave)
export HDF5_USE_FILE_LOCKING=FALSE
export HDF5_DISABLE_VERSION_CHECK=2
export HDF5_CLEANUP_DISABLE=1
export OMP_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

# ----------------------------------------------------
log() {
  echo "[`date +"%Y-%m-%d %H:%M:%S"`][$$] $*"
}

log "DATA_PATH = ${DATA_PATH}"
log "OUT_DIR   = ${OUT_DIR}"
log "JOBS      = ${JOBS}"
log "PARAM_XML = ${PARAM_XML}"
log "PARAM_XSL = ${PARAM_XSL}"

# Basic checks
if [ ! -d "${DATA_PATH}" ]; then
    log "Error: Data path ${DATA_PATH} not found!"
    exit 1
fi

if [ ! -f "${PARAM_XML}" ]; then
    log "Warning: XML file ${PARAM_XML} not found, using built-in parameter map."
fi

if [ ! -f "${PARAM_XSL}" ]; then
    log "Warning: XSL file ${PARAM_XSL} not found, using built-in stylesheet."
fi

# Top-level output dirs (subdirs will mirror DATA_PATH)
mkdir -p "${OUT_DIR}/h5" "${OUT_DIR}/noise"

# ------------------ WORKER -------------------------

convert_one() {
  local dat_host="$1"   # absolute path to .dat file

  # Normalize path relative to DATA_PATH (like your docker script)
  local rel="${dat_host#${DATA_PATH}/}"       # e.g. "knee/A/meas_XXX.dat"
  local dat_base; dat_base="$(basename "$rel" .dat)"
  local subdir; subdir="$(dirname "$rel")"

  # Mirror subdir structure under OUT_DIR/h5 and OUT_DIR/noise
  local noise_host="${OUT_DIR}/noise/${rel%.dat}.h5"
  local h5_host="${OUT_DIR}/h5/${rel%.dat}.h5"
  local noise_dir_host; noise_dir_host="$(dirname "${noise_host}")"
  local h5_dir_host;    h5_dir_host="$(dirname "${h5_host}")"

  mkdir -p "${noise_dir_host}" "${h5_dir_host}"

  if [[ -f "${h5_host}" ]]; then
    log "Skip (exists) → ${h5_host}"
    return 0
  fi

  log "Converting: ${dat_host}"
  log "  noise → ${noise_host}"
  log "  h5    → ${h5_host}"

  # Build common options, add -m / -x only if files exist
  local common_opts=( -f "${dat_host}" --skipSyncData )
  [[ -f "${PARAM_XML}" ]] && common_opts+=( -m "${PARAM_XML}" )
  [[ -f "${PARAM_XSL}" ]] && common_opts+=( -x "${PARAM_XSL}" )

  # 1) Noise measurement (z=1)
  if ! siemens_to_ismrmrd "${common_opts[@]}" -z 1 -o "${noise_host}"; then
    log "ERROR (noise) converting: ${rel}"
    # don't return; maybe still try main dataset, or you can 'return 0' here
  fi

  # 2) Main dataset (z=2)
  if ! siemens_to_ismrmrd "${common_opts[@]}" -z 2 -o "${h5_host}"; then
    log "ERROR (main) converting: ${rel}"
    return 0
  fi

  # 3) Report result size
  if [[ -f "${h5_host}" ]]; then
    log "WROTE ${h5_host} ($(du -h "${h5_host}" | awk '{print $1}'))"
  else
    log "WARNING: file missing after conversion → ${h5_host}"
  fi
}

export -f log convert_one
export DATA_PATH OUT_DIR PARAM_XML PARAM_XSL FILE_NAME

# ---------------- PARALLEL WALK --------------------

log "Parallel jobs: ${JOBS}"

# Use fd if available, else fallback to find
if command -v fd >/dev/null 2>&1; then
  log "Using fd for file discovery."
  fd -t f -e dat -i --absolute-path \
     . "${DATA_PATH}" -0 \
  | xargs -0 -n1 -P "${JOBS}" bash -lc 'convert_one "$@"' _
else
  log "fd not found, using find."
  find "${DATA_PATH}" -type f -name '*.dat' -print0 \
  | xargs -0 -n1 -P "${JOBS}" bash -lc 'convert_one "$@"' _
fi

log "Done."
