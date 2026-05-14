@echo off
setlocal enabledelayedexpansion
cd %~dp0

set MODEL=gemma
set EXE=gemma4
set FILE1=gemma-4-E2B-it-Q4_K_M.gguf

rem download
if not "%1" == "-h" if not "%1" == "--help" (
    if not exist %FILE1% (
        echo Downloading gguf file... ^(save path: %FILE1%^)
        curl https://storage.googleapis.com/ailia-models/%MODEL%/%FILE1% -o %FILE1%
    )
)

rem execute
if "%~1" == "" (
    .\%EXE%.exe -m %FILE1%
) else (
    .\%EXE%.exe %*
)
