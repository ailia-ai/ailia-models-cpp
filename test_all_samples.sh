#!/bin/bash
# Test runner for all ailia-models-cpp samples.
#
# Build the project first with cmake, then run this script. For each
# sample, the script:
#   1. Copies the freshly built binary into the sample directory.
#   2. Invokes the sample's .sh launcher (which downloads model weights).
#   3. Captures stdout/stderr and prints PASS / FAIL with timing.
#
# Usage:
#   cd build && cmake -DWITH_OPENCV=ON ..  && make -j$(nproc)
#   cd ..    && ./test_all_samples.sh [build_dir]   # default: ./build
#
set -u

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${1:-${ROOT_DIR}/build}"
LOG_DIR="${ROOT_DIR}/test_logs"
SAMPLE_TIMEOUT=600   # seconds per sample
TOTAL_TIMEOUT=10800  # seconds total

if [ ! -d "${BUILD_DIR}" ]; then
    echo "Error: build directory '${BUILD_DIR}' not found." >&2
    echo "Run 'mkdir build && cd build && cmake .. && make' first." >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

# Set runtime library search path so the bundled .so files are picked up.
export LD_LIBRARY_PATH="${ROOT_DIR}/ailia/library/linux/x64:${ROOT_DIR}/ailia_audio/library/linux/x64:${ROOT_DIR}/ailia_tokenizer/library/linux/x64:${ROOT_DIR}/ailia_speech/library/linux/x64:${ROOT_DIR}/ailia_voice/library/linux/x64:${ROOT_DIR}/ailia_tracker/library/linux/x64:${ROOT_DIR}/ailia_tflite/library/linux/x64:${LD_LIBRARY_PATH:-}"

# (sample_dir, sample_binary, args)   -- args are appended verbatim
SAMPLES=(
    "audio_processing/silero-vad|silero-vad|"
    "audio_processing/gpt-sovits|gpt-sovits|"
    "audio_processing/clap|clap|"
    "audio_processing/whisper|whisper|"
    "audio_processing/gpt-sovits-v2-pro|gpt-sovits-v2-pro|"
    "background_removal/u2net|u2net|"
    "image_classification/resnet50|resnet50|"
    "image_classification/clip|clip|"
    "face_detection/yolov3-face|yolov3-face|"
    "face_detection/retinaface|retinaface|"
    "face_identification/arcface|arcface|"
    "face_recognition/face_alignment|face_alignment|"
    "face_recognition/mediapipe_iris|mediapipe_iris|"
    "natural_language_processing/g2p_en|g2p_en|"
    "natural_language_processing/bert_maskedlm|bert_maskedlm|"
    "natural_language_processing/fugumt-ja-en|fugumt-ja-en|"
    "natural_language_processing/fugumt-en-ja|fugumt-en-ja|"
    "natural_language_processing/sentence_transformers|sentence_transformers|"
    "natural_language_processing/t5_whisper_medical|t5_whisper_medical|"
    "natural_language_processing/multilingual-e5|multilingual-e5|"
    "object_detection/yolov3-tiny|yolov3-tiny|"
    "object_detection/yolox|yolox|"
    "object_detection/m2det|m2det|"
    "object_detection/yolox_tflite|yolox_tflite|"
    "object_tracking/bytetrack|bytetrack|"
    "pose_estimation/lightweight-human-pose-estimation|lightweight-human-pose-estimation|"
)

passed=()
failed=()
skipped=()

start_total=$(date +%s)

for entry in "${SAMPLES[@]}"; do
    IFS='|' read -r rel_dir bin_name extra_args <<< "${entry}"
    sample_dir="${ROOT_DIR}/${rel_dir}"
    built_bin="${BUILD_DIR}/${rel_dir}/${bin_name}"
    log_file="${LOG_DIR}/$(echo "${rel_dir}" | tr '/' '_').log"

    elapsed_total=$(( $(date +%s) - start_total ))
    if [ ${elapsed_total} -gt ${TOTAL_TIMEOUT} ]; then
        echo "[SKIP ] ${rel_dir} (total timeout ${TOTAL_TIMEOUT}s reached)"
        skipped+=("${rel_dir}")
        continue
    fi

    if [ ! -x "${built_bin}" ]; then
        echo "[SKIP ] ${rel_dir} (binary not built: ${built_bin})"
        skipped+=("${rel_dir}")
        continue
    fi

    cp "${built_bin}" "${sample_dir}/${bin_name}"
    echo "[ RUN ] ${rel_dir}  -> log: ${log_file}"

    start=$(date +%s)
    (
        cd "${sample_dir}" && \
        timeout ${SAMPLE_TIMEOUT} bash "${bin_name}.sh" ${extra_args}
    ) > "${log_file}" 2>&1
    rc=$?
    end=$(date +%s)
    duration=$(( end - start ))

    if [ ${rc} -eq 0 ]; then
        echo "[ OK  ] ${rel_dir}  (${duration}s)"
        passed+=("${rel_dir}")
    else
        echo "[FAIL ] ${rel_dir}  (rc=${rc}, ${duration}s)"
        echo "        last 5 lines:"
        tail -n 5 "${log_file}" | sed 's/^/          /'
        failed+=("${rel_dir} (rc=${rc})")
    fi
done

echo
echo "================================================================"
echo "Summary"
echo "================================================================"
echo "Passed  : ${#passed[@]}"
for s in "${passed[@]}";  do echo "  + ${s}"; done
echo "Failed  : ${#failed[@]}"
for s in "${failed[@]}";  do echo "  - ${s}"; done
echo "Skipped : ${#skipped[@]}"
for s in "${skipped[@]}"; do echo "  ~ ${s}"; done

if [ ${#failed[@]} -gt 0 ]; then
    exit 1
fi
exit 0
