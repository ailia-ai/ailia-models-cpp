@echo off
setlocal enabledelayedexpansion
cd %~dp0

set MODEL=gemma
set EXE=gemma4-multimodal
set FILE1=gemma-4-E2B-it-Q4_K_M.gguf
set FILE2=gemma-4-E2B-it-mmproj-F16.gguf
set FILE3=sample_image.jpg

rem download
if not "%1" == "-h" if not "%1" == "--help" (
    if not exist %FILE1% (
        echo Downloading gguf file... ^(save path: %FILE1%^)
        curl https://storage.googleapis.com/ailia-models/%MODEL%/%FILE1% -o %FILE1%
    )
)
if not "%1" == "-h" if not "%1" == "--help" (
    if not exist %FILE2% (
        echo Downloading mmproj gguf file... ^(save path: %FILE2%^)
        curl https://storage.googleapis.com/ailia-models/%MODEL%/%FILE2% -o %FILE2%
    )
)
if not "%1" == "-h" if not "%1" == "--help" (
    if not exist %FILE3% (
        echo Downloading sample image... ^(save path: %FILE3%^)
        curl https://storage.googleapis.com/ailia-models/misc/%FILE3% -o %FILE3%
    )
)

rem execute
if "%~1" == "" (
    .\%EXE%.exe %FILE1% %FILE2% %FILE3%
) else (
    .\%EXE%.exe %*
)
