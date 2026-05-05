#!/bin/bash

MODEL="u2net"
FILE1="${MODEL}.onnx"
FILE2="${MODEL}.onnx.prototxt"
ARCH="large"
OPSET="10"

status=0
for n in "$@"
do
    if [ "$n" = "-a" ] || [ "$n" = "--arch" ]; then
        status=1
    elif [ "$n" = "-o" ] || [ "$n" = "--opset" ]; then
        status=2
    elif [ $status -eq 1 ]; then
        if [ "$n" = "small" ] || [ "$n" = "large" ]; then
            ARCH="$n"
        else
            break;
        fi
        status=0
    elif [ $status -eq 2 ]; then
        if [ "$n" = "10" ] || [ "$n" = "11" ]; then
            OPSET="$n"
        else
            break;
        fi
        status=0
    else
        status=0
    fi
done

#download
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ] && [ $status -eq 0 ]; then
    if [ "${ARCH}" = "small" ]; then
        if [ "${OPSET}" = "10" ]; then
            FILE1="${MODEL}p.onnx"
        else
            FILE1="${MODEL}p_opset11.onnx"
        fi
    else
        if [ "${OPSET}" = "10" ]; then
            FILE1="${MODEL}.onnx"
        else
            FILE1="${MODEL}_opset11.onnx"
        fi
    fi
    if [ ! -e ${FILE1} ]; then
        echo "Downloading onnx file... save path: ${FILE1}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE1} -o ${FILE1}
#        wget https://storage.googleapis.com/ailia-models/${MODEL}/${FILE1}
    fi
fi
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ] && [ $status -eq 0 ]; then
    if [ "${ARCH}" = "small" ]; then
        if [ "${OPSET}" = "10" ]; then
            FILE2="${MODEL}p.onnx.prototxt"
        else
            FILE2="${MODEL}p_opset11.onnx.prototxt"
        fi
    else
        if [ "${OPSET}" = "10" ]; then
            FILE2="${MODEL}.onnx.prototxt"
        else
            FILE2="${MODEL}_opset11.onnx.prototxt"
        fi
    fi
    if [ ! -e ${FILE2} ]; then
        echo "Downloading onnx file... save path: ${FILE2}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE2} -o ${FILE2}
#        wget https://storage.googleapis.com/ailia-models/${MODEL}/${FILE2}
    fi
    if [ ! -e input.png ]; then
        echo "Downloading sample image file... save path: input.png"
        curl -L https://github.com/ailia-ai/ailia-models/raw/refs/heads/master/background_removal/u2net/input.png -o input.png
    fi
    echo "ONNX file and Prototxt file are prepared!"
fi

#execute
./${MODEL} $*
