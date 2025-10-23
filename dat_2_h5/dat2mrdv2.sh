#!/bin/bash

rcrsv_cvt=false
CLEAN_MODE=false


# Resolve to the directory this script resides in (robust when called from elsewhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FILE_NAME="parameter_maps/IsmrmrdParameterMap_NewData"
cd ~/../..
PARENT_DIR=$(pwd)

ls ${PARENT_DIR}
DATA_PATH="${PARENT_DIR}/data/datasets/msk_mri"
OUT_DIR="${PARENT_DIR}/data/datasets/msk_mri_h5"

XSL_FILE="${FILE_NAME}.xsl"

# check if filename exists
if [ ! -f "${SCRIPT_DIR}/${FILE_NAME}.xml" ]; then
    echo "Error: XML file ${SCRIPT_DIR}/${FILE_NAME}.xml not found!"
    exit 1
fi

# check if data exists
if [ ! -d "${DATA_PATH}" ]; then
    echo "Error: Data path ${DATA_PATH} not found!"
    exit 1
fi

convert2mrd() {

    # $1: base name of the .dat file (without extension)
    local dat_base="$1"
    local input_dat="${DATA_PATH}/${dat_base}.dat"
    local noise_out="${OUT_DIR}/noise/noise_${dat_base}.h5"
    local h5_out="${OUT_DIR}/h5/${dat_base}.h5"

    # Adapted from R Ramasawmy convertRR_NX.sh 2019
    NUMFILES=$(siemens_to_ismrmrd -f "${input_dat}" -z 9 | grep only | grep -o '[0-9]*')

    if [ "$NUMFILES" -gt "1" ]; then
        # If .dat has noise dependency
        # Only process first noise measurement, and label it "noise_XXX.h5"
        siemens_to_ismrmrd -f "${input_dat}" -z 1 -o "${noise_out}" -m ${SCRIPT_DIR}/${FILE_NAME}.xml -x ${SCRIPT_DIR}/${XSL_FILE} --skipSyncData
    fi

    siemens_to_ismrmrd -f "${input_dat}" -z "$NUMFILES" -o "${h5_out}" -m ${SCRIPT_DIR}/${FILE_NAME}.xml -x ${SCRIPT_DIR}/${XSL_FILE} --skipSyncData
}

remove_mrd() {
    # $1: base name of the .dat file (without extension)
    local dat_base="$1"
    rm -v "${OUT_DIR}/noise/noise_${dat_base}.h5" "${OUT_DIR}/h5/${dat_base}.h5"
}

batch_remove () {
    local dat_file=$(basename "${1}")
    local dat_file=$(echo "$dat_file" | cut -f 1 -d '.')

    remove_mrd "${dat_file}"
}


batch_convert () {
    local dat_file=$(basename "${1}")
    local dat_file=$(echo "$dat_file" | cut -f 1 -d '.')

    mkdir -p "${OUT_DIR}/h5" "${OUT_DIR}/noise"

    echo Current file:
    echo ${dat_file}
    echo ${SCRIPT_DIR}

    convert2mrd "${dat_file}"
}

# Create output directories if does not exist

if [ -f "${DATA_PATH}" ]; then # If given path is to a file, just convert that file

    echo "Single file"

    if ${rcrsv_cvt}; then
        echo "-r is given with a file name, ignoring recursive option."
    fi

    # Parse filename without extension and folder name
    DAT_FILE=$(basename "${DATA_PATH}")
    DAT_FILE=$(echo "$DAT_FILE" | cut -f 1 -d '.')
    DATA_PATH=$(dirname "${DATA_PATH}")
    echo ${DATA_PATH}

    # If output directory is not given, make it the input dir
    if [ -z "${OUT_DIR}" ]; then
        OUT_DIR=$DATA_PATH
    else
        mkdir -p $OUT_DIR
    fi


    if ${CLEAN_MODE}; then
        # Remove from the chosen output directory (not necessarily the data path)
        remove_mrd "${DAT_FILE}"
    else
        mkdir -p "${OUT_DIR}/h5" "${OUT_DIR}/noise"
        echo "Converting ${DAT_FILE}.."
        convert2mrd "${DAT_FILE}"
    fi

else # Otherwise convert all .dat files inside the folder
    ROOT_PATH=${DATA_PATH}

    if ${rcrsv_cvt}; then

        if ${CLEAN_MODE}; then
            echo "Recursively removing..."

            export OUT_DIR DATA_PATH
            export -f remove_mrd
            export -f batch_remove
            find "${ROOT_PATH}" -name \*.dat -execdir bash -c "batch_remove \"{}\"" \;

        else
            echo "Recursively converting..."

            export XSL_FILE
            export SCRIPT_DIR
            export OUT_DIR DATA_PATH
            export -f convert2mrd
            export -f batch_convert
            find "${ROOT_PATH}" -name \*.dat -execdir bash -c "batch_convert \"{}\"" \;

        fi
    else
        DATA_PATH=${ROOT_PATH}

        # If output directory is not given, make it the input dir
        if [ -z "${OUT_DIR}" ]; then
            OUT_DIR=$DATA_PATH
        else
            mkdir -p $OUT_DIR
        fi

    if ${CLEAN_MODE}; then

            for i in ${DATA_PATH}/*.dat; do
                [ -f "$i" ] || break
                    DAT_FILE=$(basename "${i}")
                    DAT_FILE=$(echo "$DAT_FILE" | cut -f 1 -d '.')

            # Remove from the chosen output directory (not necessarily the data path)
            remove_mrd "${DAT_FILE}"
            done

        else

            mkdir -p "${OUT_DIR}/h5" "${OUT_DIR}/noise"

            for i in ${DATA_PATH}/*.dat; do
                [ -f "$i" ] || break
                    DAT_FILE=$(basename "${i}")
                    DAT_FILE=$(echo "$DAT_FILE" | cut -f 1 -d '.')

                    convert2mrd "${DAT_FILE}"
            done
        fi
    fi

fi

echo "Done."
