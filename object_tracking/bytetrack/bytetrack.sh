#!/bin/bash

MODEL="bytetrack"
WEIGHT_MODEL="yolox"
FILE1="${WEIGHT_MODEL}_s.opt.onnx"
FILE2="${WEIGHT_MODEL}_s.opt.onnx.prototxt"
SAMPLE_VIDEO="demo.mp4"

# download
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE1} ]; then
        echo "Downloading onnx file... save path: ${FILE1}"
        curl https://storage.googleapis.com/ailia-models/${WEIGHT_MODEL}/${FILE1} -o ${FILE1}
    fi
    if [ ! -e ${FILE2} ]; then
        echo "Downloading prototxt file... save path: ${FILE2}"
        curl https://storage.googleapis.com/ailia-models/${WEIGHT_MODEL}/${FILE2} -o ${FILE2}
    fi
    if [ ! -e ${SAMPLE_VIDEO} ]; then
        echo "Downloading sample video... save path: ${SAMPLE_VIDEO}"
        curl -L https://storage.googleapis.com/ailia-models/${MODEL}/${SAMPLE_VIDEO} -o ${SAMPLE_VIDEO}
    fi
    echo "ONNX file and Prototxt file are prepared!"
fi

# execute
if [ $# -eq 0 ]; then
    ./${MODEL} ${SAMPLE_VIDEO}
else
    ./${MODEL} $*
fi
