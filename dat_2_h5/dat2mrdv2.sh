#!/bin/bash

rcrsv_cvt=false
CLEAN_MODE=false
out_dir=""

while getopts ":hrco:" option; do
  case $option in
    h) echo "usage: $0 [-h] [-r] [-c] [-o <string>] path_to_convert ..."; exit ;;
    r) rcrsv_cvt=true ;;
    c) CLEAN_MODE=true; echo "Cleaning mode..." ;;
    o) out_dir=${OPTARG} ;;
    ?) echo "error: option -$OPTARG is not implemented"; exit ;;
  esac
done

# remove the options from the positional parameters
shift $(( OPTIND - 1 ))

full_name=$1
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
    siemens_to_ismrmrd -f "$1/$2.dat" -z 1 -o "$3/noise_v3/noise_$2.h5" -m ${SCRIPT_DIR}/${FILE_NAME}.xml -x ${SCRIPT_DIR}/${XSL_FILE} --skipSyncData
    fi

    siemens_to_ismrmrd -f "$1/$2.dat" -z $NUMFILES -o "$3/h5_v3/$2.h5" -m ${SCRIPT_DIR}/${FILE_NAME}.xml -x ${SCRIPT_DIR}/${XSL_FILE} --skipSyncData
}

remove_mrd() {
    rm -v "$1/noise_v3/noise_$2.h5" "$1/h5_v3/$2.h5"
}

batch_remove () {
    local dat_file=$(basename "${1}")
    local dat_file=$(echo "$dat_file" | cut -f 1 -d '.')

    remove_mrd './' ${dat_file}
}


batch_convert () {
    local dat_file=$(basename "${1}")
    local dat_file=$(echo "$dat_file" | cut -f 1 -d '.')

    mkdir -p "h5_v3" "noise_v3"

    echo Current file:
    echo ${dat_file}
    echo ${SCRIPT_DIR}

    # Pass current directory as output directory for correctness
    convert2mrd '.' ${dat_file} '.'
}

# Create output directories if does not exist

if [ -f "${full_name}" ]; then # If given path is to a file, just convert that file

    echo "Single file"

    if ${rcrsv_cvt}; then
        echo "-r is given with a file name, ignoring recursive option."
    fi

    # Parse filename without extension and folder name
    DAT_FILE=$(basename "${full_name}")
    DAT_FILE=$(echo "$DAT_FILE" | cut -f 1 -d '.')
    DATA_PATH=$(dirname "${full_name}")
    echo ${DATA_PATH}

    # If output directory is not given, make it the input dir
    if [ -z "${out_dir}" ]; then
        out_dir=$DATA_PATH
    else
        mkdir -p $out_dir
    fi


    if ${CLEAN_MODE}; then
    # Remove from the chosen output directory (not necessarily the data path)
    remove_mrd ${out_dir} ${DAT_FILE}
    else
    mkdir -p "${out_dir}/h5_v3" "${out_dir}/noise_v3"
        echo "Converting ${DAT_FILE}.."
        convert2mrd ${DATA_PATH} ${DAT_FILE} ${out_dir}
    fi

else # Otherwise convert all .dat files inside the folder
    ROOT_PATH=${full_name}

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
        if [ -z "${out_dir}" ]; then
            out_dir=$DATA_PATH
        else
            mkdir -p $out_dir
        fi

    if ${CLEAN_MODE}; then

            for i in ${DATA_PATH}/*.dat; do
                [ -f "$i" ] || break
                    DAT_FILE=$(basename "${i}")
                    DAT_FILE=$(echo "$DAT_FILE" | cut -f 1 -d '.')

            # Remove from the chosen output directory (not necessarily the data path)
            remove_mrd ${out_dir} ${DAT_FILE}
            done

        else

            mkdir -p "${out_dir}/h5_v3" "${out_dir}/noise_v3"

            for i in ${DATA_PATH}/*.dat; do
                [ -f "$i" ] || break
                    DAT_FILE=$(basename "${i}")
                    DAT_FILE=$(echo "$DAT_FILE" | cut -f 1 -d '.')

                    convert2mrd ${DATA_PATH} ${DAT_FILE} ${out_dir}
            done
        fi
    fi

fi

echo "Done."
