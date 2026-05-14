#!/bin/bash

MODEL="gemma"
EXE="gemma-multimodal"
FILE1="gemma-4-E2B-it-Q4_K_M.gguf"
FILE2="gemma-4-E2B-it-mmproj-F16.gguf"
FILE3="sample.jpg"

#download
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE1} ]; then
        echo "Downloading gguf file... save path: ${FILE1}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE1} -o ${FILE1}
    fi
fi
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE2} ]; then
        echo "Downloading mmproj gguf file... save path: ${FILE2}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE2} -o ${FILE2}
    fi
fi
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE3} ]; then
        echo "Downloading sample image... save path: ${FILE3}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE3} -o ${FILE3}
    fi
fi

#execute
if [ $# -eq 0 ]; then
    ./${EXE} ${FILE1} ${FILE2} ${FILE3}
else
    ./${EXE} $*
fi
