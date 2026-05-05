#!/bin/bash

MODEL="arcface"
URL_ROOT="https://storage.googleapis.com/ailia-models"
ARCH="${MODEL}"
FILE1="${ARCH}.onnx"
FILE2="${ARCH}.onnx.prototxt"

FACE_ARCH="yolov3-face"
FACE_OPT=".opt"
FACE_FILE1="${FACE_ARCH}${FACE_OPT}.onnx"
FACE_FILE2="${FACE_ARCH}${FACE_OPT}.onnx.prototxt"

status=0
video=0
for n in "$@"
do
    if [ "$n" = "-a" ] || [ "$n" = "--arch" ]; then
        status=1
    elif [ "$n" = "-f" ] || [ "$n" = "--face" ]; then
        status=2
    elif [ $status -eq 1 ]; then
        if [ "$n" = "arcface" ] || [ "$n" = "arcface_mixed_90_82" ] || [ "$n" = "arcface_mixed_90_99" ] || [ "$n" = "arcface_mixed_eq_90_89" ]; then
            ARCH="$n"
        else
            break;
        fi
        status=0
    elif [ $status -eq 2 ]; then
        if [ "$n" = "yolov3-face" ]; then
            FACE_ARCH="$n"
            FACE_OPT=".opt"
        elif [ "$n" = "blazeface" ]; then
            FACE_ARCH="$n"
            FACE_OPT=""
        else
            break;
        fi
        status=0
    elif [ "$n" = "-v" ] || [ "$n" = "--video" ]; then
        video=1
    else
        status=0
    fi
done

#download
FILE1="${ARCH}.onnx"
FILE2="${ARCH}.onnx.prototxt"
FACE_FILE1="${FACE_ARCH}${FACE_OPT}.onnx"
FACE_FILE2="${FACE_ARCH}${FACE_OPT}.onnx.prototxt"

if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ] && [ $status -eq 0 ]; then
    if [ ! -e ${FILE1} ]; then
        echo "Downloading onnx file... save path: ${FILE1}"
        curl ${URL_ROOT}/${MODEL}/${FILE1} -o ${FILE1}
#        wget ${URL_ROOT}/${MODEL}/${FILE1}
    fi
fi
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ] && [ $status -eq 0 ]; then
    if [ ! -e ${FILE2} ]; then
        echo "Downloading onnx file... save path: ${FILE2}"
        curl ${URL_ROOT}/${MODEL}/${FILE2} -o ${FILE2}
#        wget ${URL_ROOT}/${MODEL}/${FILE2}
    fi
    echo "ONNX file and Prototxt file are prepared!"
fi

if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ] && [ $video -eq 1 ] && [ $status -eq 0 ]; then
    if [ ! -e ${FACE_FILE1} ]; then
        echo "Downloading onnx file... save path: ${FACE_FILE1}"
        curl ${URL_ROOT}/${FACE_ARCH}/${FACE_FILE1} -o ${FACE_FILE1}
#        wget ${URL_ROOT}/${FACE_ARCH}/${FACE_FILE1}
    fi
fi
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ] && [ $video -eq 1 ] && [ $status -eq 0 ]; then
    if [ ! -e ${FACE_FILE2} ]; then
        echo "Downloading onnx file... save path: ${FACE_FILE2}"
        curl ${URL_ROOT}/${FACE_ARCH}/${FACE_FILE2} -o ${FACE_FILE2}
#        wget ${URL_ROOT}/${FACE_ARCH}/${FACE_FILE2}
    fi
    echo "ONNX file and Prototxt file are prepared!"
fi

if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e correct_pair_1.jpg ]; then
        echo "Downloading sample image file... save path: correct_pair_1.jpg"
        curl -L https://github.com/ailia-ai/ailia-models/raw/refs/heads/master/face_identification/arcface/correct_pair_1.jpg -o correct_pair_1.jpg
    fi
    if [ ! -e correct_pair_2.jpg ]; then
        echo "Downloading sample image file... save path: correct_pair_2.jpg"
        curl -L https://github.com/ailia-ai/ailia-models/raw/refs/heads/master/face_identification/arcface/correct_pair_2.jpg -o correct_pair_2.jpg
    fi
fi

#execute
./${MODEL} $*
