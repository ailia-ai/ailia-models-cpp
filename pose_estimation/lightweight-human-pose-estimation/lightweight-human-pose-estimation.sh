#!/bin/bash

MODEL="lightweight-human-pose-estimation"
FILE1="${MODEL}.opt.onnx"
FILE2="${MODEL}.opt.onnx.prototxt"

for n in "$@"
do
    if [ "$n" = "-n" ] || [ "$n" = "--normal" ]; then
        FILE1="${MODEL}.onnx"
        FILE2="${MODEL}.onnx.prototxt"
    fi
done

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
    if [ ! -e balloon.png ]; then
        echo "Downloading sample image file... save path: balloon.png"
        curl -L https://github.com/ailia-ai/ailia-models/raw/refs/heads/master/pose_estimation/lightweight-human-pose-estimation/input.jpg -o balloon.png
    fi
    echo "ONNX file and Prototxt file are prepared!"
fi

#execute
./${MODEL} $*
