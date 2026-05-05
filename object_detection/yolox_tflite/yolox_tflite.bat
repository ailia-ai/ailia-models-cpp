@echo off
setlocal enabledelayedexpansion
cd %~dp0

set MODEL=yolox_tflite
set WEIGHT=yolox_tiny_full_integer_quant.opt.tflite

rem download
if not "%1" == "-h" if not "%1" == "--help" (
    if not exist %WEIGHT% (
        echo Downloading tflite file... ^(save path: %WEIGHT%^)
        curl -L https://storage.googleapis.com/ailia-models-tflite/yolox/%WEIGHT% -o %WEIGHT%
    )
    echo TFLite file is prepared^^!
)

rem execute
.\%MODEL%.exe %*
