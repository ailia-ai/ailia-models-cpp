#!/bin/bash

MODEL="t5_whisper_medical"
FILE1="t5_whisper_medical-decoder-with-lm-head.obf.onnx"
FILE2="t5_whisper_medical-decoder-with-lm-head.onnx.prototxt"
FILE3="t5_whisper_medical-encoder.obf.onnx"
FILE4="t5_whisper_medical-encoder.onnx.prototxt"
FILE5="spiece.model"

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
    echo "ONNX file and Prototxt file are prepared!"
fi
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE3} ]; then
        echo "Downloading onnx file... save path: ${FILE3}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE3} -o ${FILE3}
    fi
fi
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE4} ]; then
        echo "Downloading onnx file... save path: ${FILE4}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE4} -o ${FILE4}
    fi
    echo "SPM files are prepared!"
fi
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE5} ]; then
        echo "Downloading spm file... save path: ${FILE5}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE5} -o ${FILE5}
    fi
    echo "SPM files are prepared!"
fi
#execute
./${MODEL} $*
