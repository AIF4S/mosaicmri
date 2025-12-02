#!/bin/bash
set -Eeuo pipefail
IFS=$'\n\t'

# ---------------- CONFIG ----------------

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

echo "DATA_PATH  = ${DATA_PATH}"
echo "OUT_DIR    = ${OUT_DIR}"
echo "PARAM_XML  = ${PARAM_XML}"
echo "PARAM_XSL  = ${PARAM_XSL}"

# Basic checks
if [ ! -d "${DATA_PATH}" ]; then
    echo "Error: Data path ${DATA_PATH} not found!"
    exit 1
fi

if [ ! -f "${PARAM_XML}" ]; then
    echo "Warning: XML file ${PARAM_XML} not found, using built-in parameter map."
fi

if [ ! -f "${PARAM_XSL}" ]; then
    echo "Warning: XSL file ${PARAM_XSL} not found, using built-in stylesheet."
fi

mkdir -p "${OUT_DIR}/h5" "${OUT_DIR}/noise"

# ------------- FUNCTIONS ----------------

convert2mrd() {
    # $1: full path to the .dat file
    local input_dat="$1"
    local dat_file
    dat_file="$(basename "${input_dat}")"
    local dat_base="${dat_file%%.*}"

    local noise_out="${OUT_DIR}/noise/noise_${dat_base}.h5"
    local h5_out="${OUT_DIR}/h5/${dat_base}.h5"

    if [[ -f "${h5_out}" ]]; then
        echo "Skip (exists): ${h5_out}"
        return 0
    fi

    echo "Converting: ${input_dat} -> ${h5_out}"

    # Build common options, add -m / -x only if files exist
    local common_opts=( -f "${input_dat}" --skipSyncData )
    [[ -f "${PARAM_XML}" ]] && common_opts+=( -m "${PARAM_XML}" )
    [[ -f "${PARAM_XSL}" ]] && common_opts+=( -x "${PARAM_XSL}" )

    # Noise measurement
    siemens_to_ismrmrd "${common_opts[@]}" -z 1 -o "${noise_out}"

    # Main dataset
    siemens_to_ismrmrd "${common_opts[@]}" -z 2 -o "${h5_out}"
}

# ------------- MAIN: SEQUENTIAL WALK --------------

echo "Recursively converting all .dat under ${DATA_PATH}..."

count=0
while IFS= read -r -d '' dat_path; do
    echo "[$count] File: ${dat_path}"
    convert2mrd "${dat_path}"
    count=$((count + 1))
done < <(find "${DATA_PATH}" -type f -name '*.dat' -print0)

echo "Processed ${count} .dat files."
echo "Done."
