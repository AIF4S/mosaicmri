#!/usr/bin/env bash
set -euo pipefail

# ****************************
# Convert Siemens .dat files to ISMRMRD HDF5:
# - If multiple measurements: convert [1] as noise, [N] as data
# - Uses local parameter map files (XML/XSL) in this folder
# ****************************

# Directories (created if missing)
mkdir -p h5
mkdir -p noise

# Input folder with .dat files
INPUT_DIR="/home/paula/SDF_intro/MRI_new_dataset/data_samples/mri_new_data"
# Parameter maps reside next to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARAM_XML="${SCRIPT_DIR}/wip_070_fire_IsmrmrdParameterMap_Siemens_pd_rr.xml"
PARAM_XSL="${SCRIPT_DIR}/wip_070_fire_IsmrmrdParameterMap_Siemens_pd_rr.xsl"



SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd ~/../..
PARENT_DIR=$(pwd)

ls ${PARENT_DIR}
INPUT_DIR="${PARENT_DIR}/data/datasets/msk_original"
OUT_DIR="${PARENT_DIR}/data/datasets/msk_mri_h5"


# Optional: custom parameter map/stylesheet (if present)
FILE_NAME="parameter_maps/IsmrmrdParameterMap_Siemens_debug"
USER_MAP_XML="${SCRIPT_DIR}/${FILE_NAME}.xml"
USER_MAP_XSL="${SCRIPT_DIR}/${FILE_NAME}.xsl"


# Ensure converter exists
if ! command -v siemens_to_ismrmrd >/dev/null 2>&1; then
  echo "siemens_to_ismrmrd not found. Please install it and ensure it is on PATH." >&2
  exit 1
fi

# Ensure parameter maps exist
for f in "$PARAM_XML" "$PARAM_XSL"; do
  if [ ! -f "$f" ]; then
    echo "Missing parameter map file: $f" >&2
    exit 1
  fi
done

shopt -s nullglob
mapfile -t DAT_FILES < <(printf '%s\n' "$INPUT_DIR"/*.dat | sort)
if [ ${#DAT_FILES[@]} -eq 0 ]; then
  echo "No .dat files found in $INPUT_DIR" >&2
  exit 1
fi

for frr in "${DAT_FILES[@]}"; do
  echo "Processing $frr ..."

  # Probe number of measurements by attempting to access a high index and parse message
  set +e
  probe_out=$(siemens_to_ismrmrd -f "$frr" -z 9 2>&1)
  probe_rc=$?
  set -e

  # Handle unsupported VD header message gracefully
  if grep -q "Only VD line files with MrParcRaidFileHeader.hdSize_ == 0" <<<"$probe_out"; then
    echo "Skipping unsupported file format (VD header not supported by converter): $frr" >&2
    continue
  fi

  # # Extract NUMFILES if present, else fallback to 1
  # NUMFILES=$(grep -oE 'only [0-9]+' <<<"$probe_out" | awk '{print $2}' || true)
  # if [[ -z "${NUMFILES:-}" ]]; then
  #   # If the probe failed but no explicit count, assume 1 measurement
  #   NUMFILES=1
  # fi
  # echo "Detected measurements: $NUMFILES"

  base_name=$(basename "$frr")
  stem="${base_name%.*}"

  # If multiple measurements, create noise from measurement 1
  if [[ "$NUMFILES" =~ ^[0-9]+$ ]] && [ "$NUMFILES" -gt 1 ]; then
    siemens_to_ismrmrd \
      -f "$frr" \
      -z 1 \
      -o "noise/noise_${stem}.h5" \
      -m "$PARAM_XML" \
      -x "$PARAM_XSL" \
      --skipSyncData
  fi

  # Convert the last measurement (NUMFILES - 1) for data when NUMFILES>1, else 0
  if [ "$NUMFILES" -gt 1 ]; then
    data_meas=$((NUMFILES - 1))
  else
    data_meas=0
  fi

  siemens_to_ismrmrd \
    -f "$frr" \
    -z 2 \
    -o "h5/${stem}.h5" \
    -m "$PARAM_XML" \
    -x "$PARAM_XSL" \
    --skipSyncData 
done

