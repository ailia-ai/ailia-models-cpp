@echo off
setlocal enabledelayedexpansion
cd %~dp0

set MODEL=bytetrack
set WEIGHT_MODEL=yolox
set FILE1=%WEIGHT_MODEL%_s.opt.onnx
set FILE2=%WEIGHT_MODEL%_s.opt.onnx.prototxt
set SAMPLE_VIDEO=demo.mp4

rem download
if not "%1" == "-h" if not "%1" == "--help" (
    if not exist %FILE1% (
        echo Downloading onnx file... ^(save path: %FILE1%^)
        curl https://storage.googleapis.com/ailia-models/%WEIGHT_MODEL%/%FILE1% -o %FILE1%
    )
    if not exist %FILE2% (
        echo Downloading prototxt file... ^(save path: %FILE2%^)
        curl https://storage.googleapis.com/ailia-models/%WEIGHT_MODEL%/%FILE2% -o %FILE2%
    )
    if not exist %SAMPLE_VIDEO% (
        echo Downloading sample video... ^(save path: %SAMPLE_VIDEO%^)
        curl -L https://storage.googleapis.com/ailia-models/%MODEL%/%SAMPLE_VIDEO% -o %SAMPLE_VIDEO%
    )
    echo ONNX file and Prototxt file are prepared^^!
)

rem execute
if "%~1" == "" (
    .\%MODEL%.exe %SAMPLE_VIDEO%
) else (
    .\%MODEL%.exe %*
)
