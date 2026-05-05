#!/bin/bash

MODEL="face_alignment"
REMOTE="2DFAN-4"
FILE1="${MODEL}.onnx"
FILE2="${MODEL}.onnx.prototxt"

#download
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE1} ]; then
        echo "Downloading onnx file... save path: ${FILE1}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${REMOTE}.onnx -o ${FILE1}
    fi
fi
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE2} ]; then
        echo "Downloading onnx file... save path: ${FILE2}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${REMOTE}.onnx.prototxt -o ${FILE2}
    fi
    if [ ! -e aflw-test.jpg ]; then
        echo "Downloading sample image file... save path: aflw-test.jpg"
        curl -L https://github.com/ailia-ai/ailia-models/raw/refs/heads/master/face_recognition/face_alignment/aflw-test.jpg -o aflw-test.jpg
    fi
    echo "ONNX file and Prototxt file are prepared!"
fi

#execute
./${MODEL} $*
