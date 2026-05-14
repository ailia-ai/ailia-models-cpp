#!/bin/bash

MODEL="gemma"
EXE="gemma"
FILE1="gemma-4-E2B-it-Q4_K_M.gguf"

#download
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE1} ]; then
        echo "Downloading gguf file... save path: ${FILE1}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE1} -o ${FILE1}
    fi
fi

#execute
if [ $# -eq 0 ]; then
    ./${EXE} -m ${FILE1}
else
    ./${EXE} $*
fi
