#!/bin/bash

MODEL="silero-vad"
FILE1="silero_vad.onnx"
FILE2="silero_vad.onnx.prototxt"

#download
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE1} ]; then
        echo "Downloading onnx file... save path: ${FILE1}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE1} -o ${FILE1}
    fi
fi
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE2} ]; then
        echo "Downloading onnx file... save path: ${FILE2}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE2} -o ${FILE2}
    fi
    if [ ! -e en_example.wav ]; then
        echo "Downloading sample wav file... save path: en_example.wav"
        curl -L https://github.com/ailia-ai/ailia-models/raw/refs/heads/master/audio_processing/silero-vad/en_example.wav -o en_example.wav
    fi
    echo "ONNX file and Prototxt file are prepared!"
fi
#execute
./${MODEL} $*
