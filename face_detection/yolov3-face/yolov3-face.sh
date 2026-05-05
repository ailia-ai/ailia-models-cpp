#!/bin/bash

MODEL="yolov3-face"
FILE1="${MODEL}.opt.onnx"
FILE2="${MODEL}.opt.onnx.prototxt"

#download
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE1} ]; then
        echo "Downloading onnx file... save path: ${FILE1}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE1} -o ${FILE1}
#        wget https://storage.googleapis.com/ailia-models/${MODEL}/${FILE1}
    fi
fi
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE2} ]; then
        echo "Downloading onnx file... save path: ${FILE2}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE2} -o ${FILE2}
#        wget https://storage.googleapis.com/ailia-models/${MODEL}/${FILE2}
    fi
    if [ ! -e couple.jpg ]; then
        echo "Downloading sample image file... save path: couple.jpg"
        curl -L https://github.com/ailia-ai/ailia-models/raw/refs/heads/master/face_detection/yolov3-face/couple.jpg -o couple.jpg
    fi
    echo "ONNX file and Prototxt file are prepared!"
fi

#execute
./${MODEL} $*
