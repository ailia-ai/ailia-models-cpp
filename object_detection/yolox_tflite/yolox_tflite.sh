#!/bin/bash

MODEL=yolox_tflite
WEIGHT=yolox_tiny_full_integer_quant.opt.tflite

# download
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${WEIGHT} ]; then
        echo "Downloading tflite file... save path: ${WEIGHT}"
        curl -L https://storage.googleapis.com/ailia-models-tflite/yolox/${WEIGHT} -o ${WEIGHT}
    fi
    echo "TFLite file is prepared!"
fi

# execute
./${MODEL} $*
