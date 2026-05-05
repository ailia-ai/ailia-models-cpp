#!/bin/bash
#
# Remove the macOS quarantine attribute from all bundled .dylib files
# under each ailia submodule. On macOS, libraries downloaded from the
# Internet (or fetched via git from GitHub) are tagged with
# "com.apple.quarantine", and Gatekeeper refuses to load them with
# the message "libailia_voice.dylib is damaged" (or similar).
#
# Run this script after `git submodule update --init --recursive`.
#

set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

DYLIB_PATHS=(
    "${ROOT_DIR}/ailia/library/mac"
    "${ROOT_DIR}/ailia_audio/library/mac"
    "${ROOT_DIR}/ailia_tokenizer/library/mac"
    "${ROOT_DIR}/ailia_speech/library/mac"
    "${ROOT_DIR}/ailia_voice/library/mac"
)

for path in "${DYLIB_PATHS[@]}"; do
    if [ -d "${path}" ]; then
        for dylib in "${path}"/*.dylib; do
            if [ -e "${dylib}" ]; then
                echo "Removing quarantine attribute: ${dylib}"
                xattr -d com.apple.quarantine "${dylib}" 2>/dev/null || true
            fi
        done
    fi
done

echo "Done."
