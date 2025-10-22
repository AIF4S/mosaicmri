#!/bin/bash

rcrsv_cvt=false
CLEAN_MODE=false


PARENT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_PATH="${PARENT_DIR}/data/datasets/msk_mri"
OUT_DIR="${PARENT_DIR}/data/datasets/msk_mri_h5"
FILE_NAME="parameter_maps/IsmrmrdParameterMap_NewData"
XSL_FILE=${2:-${FILE_NAME}.xsl}


# Resolve to the directory this script resides in (robust when called from elsewhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

convert2mrd() {

    # Adapted from R Ramasawmy convertRR_NX.sh 2019
    NUMFILES=$(siemens_to_ismrmrd -f "$1/$2.dat" -z 9 | grep only | grep -o '[0-9]*')

    if [ $NUMFILES -gt "1" ]
    then
	# If .dat has noise dependency
	# Only process first noise measurement, and label it "noise_XXX.h5"
    siemens_to_ismrmrd -f "$1/$2.dat" -z 1 -o "$3/noise/noise_$2.h5" -m ${SCRIPT_DIR}/${FILE_NAME}.xml -x ${SCRIPT_DIR}/${XSL_FILE} --skipSyncData
    fi

    siemens_to_ismrmrd -f "$1/$2.dat" -z $NUMFILES -o "$3/h5/$2.h5" -m ${SCRIPT_DIR}/${FILE_NAME}.xml -x ${SCRIPT_DIR}/${XSL_FILE} --skipSyncData
}

remove_mrd() {
    rm -v "$1/noise/noise_$2.h5" "$1/h5/$2.h5"
}

batch_remove () {
    local dat_file=$(basename "${1}")
    local dat_file=$(echo "$dat_file" | cut -f 1 -d '.')

    remove_mrd './' ${dat_file}
}


batch_convert () {
    local dat_file=$(basename "${1}")
    local dat_file=$(echo "$dat_file" | cut -f 1 -d '.')

    mkdir -p "h5" "noise"

    echo Current file:
    echo ${dat_file}
    echo ${SCRIPT_DIR}

    # Pass current directory as output directory for correctness
    convert2mrd '.' ${dat_file} '.'
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
    remove_mrd ${OUT_DIR} ${DAT_FILE}
    else
    mkdir -p "${OUT_DIR}/h5" "${OUT_DIR}/noise"
        echo "Converting ${DAT_FILE}.."
        convert2mrd ${DATA_PATH} ${DAT_FILE} ${OUT_DIR}
    fi

else # Otherwise convert all .dat files inside the folder
    ROOT_PATH=${DATA_PATH}

    if ${rcrsv_cvt}; then

        if ${CLEAN_MODE}; then
            echo "Recursively removing..."

            export -f remove_mrd
            export -f batch_remove
            find "${ROOT_PATH}" -name \*.dat -execdir bash -c "batch_remove \"{}\"" \;

        else
            echo "Recursively converting..."

            export XSL_FILE
            export SCRIPT_DIR
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
            remove_mrd ${OUT_DIR} ${DAT_FILE}
            done

        else

            mkdir -p "${OUT_DIR}/h5" "${OUT_DIR}/noise"

            for i in ${DATA_PATH}/*.dat; do
                [ -f "$i" ] || break
                    DAT_FILE=$(basename "${i}")
                    DAT_FILE=$(echo "$DAT_FILE" | cut -f 1 -d '.')

                    convert2mrd ${DATA_PATH} ${DAT_FILE} ${OUT_DIR}
            done
        fi
    fi

fi

echo "Done."
