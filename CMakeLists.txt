#*******************************************************************
#
#    DESCRIPTION:
#      AILIA C++ SAMPLE BUILD SCRIPT
#    COPYRIGHT:
#      AXELL CORPORATION, ax Inc.
#
#******************************************************************/

cmake_minimum_required(VERSION 3.10)
project(samples C CXX)

set(CMAKE_CXX_STANDARD 17)

if(MSVC)
    # Silence the C4996 deprecation warnings for the standard CRT (fopen,
    # sprintf, etc.) that MSVC tags as "unsafe". The samples use the
    # portable C API on purpose so that the same source builds on every
    # platform.
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)

    # Treat source files as UTF-8 so that Japanese string literals do not
    # trigger C4566 ("character cannot be represented in the current code
    # page (1252)").
    add_compile_options(/utf-8)
endif()

# When OFF, skip every sample that requires OpenCV. Useful when only
# audio/NLP samples (whisper, gpt-sovits-v2-pro, etc.) are needed and
# OpenCV is unavailable on the build host.
option(WITH_OPENCV "Build samples that require OpenCV" ON)

set(AILIA_LIBRARY_PATH ${CMAKE_SOURCE_DIR}/ailia/library)
set(AILIA_TOKENIZER_PATH ${CMAKE_SOURCE_DIR}/ailia_tokenizer/library)
set(AILIA_AUDIO_PATH ${CMAKE_SOURCE_DIR}/ailia_audio/library)
set(AILIA_SPEECH_PATH ${CMAKE_SOURCE_DIR}/ailia_speech/library)
set(AILIA_VOICE_PATH ${CMAKE_SOURCE_DIR}/ailia_voice/library)
set(AILIA_TRACKER_PATH ${CMAKE_SOURCE_DIR}/ailia_tracker/library)
set(AILIA_TFLITE_PATH ${CMAKE_SOURCE_DIR}/ailia_tflite/library)

set (INCLUDE_PATH ${AILIA_LIBRARY_PATH}/include ${AILIA_AUDIO_PATH}/include ${AILIA_TOKENIZER_PATH}/include ${AILIA_SPEECH_PATH}/include ${AILIA_VOICE_PATH}/include ${AILIA_TRACKER_PATH}/include ${AILIA_TFLITE_PATH}/include ../../util)
if(WIN32)
    set (LIBRARY_PATH ${AILIA_LIBRARY_PATH}/windows/x64 ${AILIA_AUDIO_PATH}/windows/x64 ${AILIA_TOKENIZER_PATH}/windows/x64 ${AILIA_SPEECH_PATH}/windows/x64 ${AILIA_VOICE_PATH}/windows/x64 ${AILIA_TRACKER_PATH}/windows/x64 ${AILIA_TFLITE_PATH}/windows/x64)
elseif(APPLE)
    set (LIBRARY_PATH ${AILIA_LIBRARY_PATH}/mac ${AILIA_AUDIO_PATH}/mac ${AILIA_TOKENIZER_PATH}/mac ${AILIA_SPEECH_PATH}/mac ${AILIA_VOICE_PATH}/mac ${AILIA_TRACKER_PATH}/mac ${AILIA_TFLITE_PATH}/mac)
elseif(UNIX AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set (LIBRARY_PATH ${AILIA_LIBRARY_PATH}/linux/arm64-v8a ${AILIA_AUDIO_PATH}/linux/arm64-v8a ${AILIA_TOKENIZER_PATH}/linux/arm64-v8a ${AILIA_SPEECH_PATH}/linux/arm64-v8a ${AILIA_VOICE_PATH}/linux/arm64-v8a ${AILIA_TRACKER_PATH}/linux/arm64-v8a ${AILIA_TFLITE_PATH}/linux/arm64-v8a)
elseif(UNIX AND CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7-a")
    set (LIBRARY_PATH ${AILIA_LIBRARY_PATH}/linux/armeabi-v7a ${AILIA_AUDIO_PATH}/linux/armeabi-v7a ${AILIA_TOKENIZER_PATH}/linux/armeabi-v7a ${AILIA_SPEECH_PATH}/linux/armeabi-v7a)
else(UNIX)
    set (LIBRARY_PATH ${AILIA_LIBRARY_PATH}/linux/x64 ${AILIA_AUDIO_PATH}/linux/x64 ${AILIA_TOKENIZER_PATH}/linux/x64 ${AILIA_SPEECH_PATH}/linux/x64 ${AILIA_VOICE_PATH}/linux/x64 ${AILIA_TRACKER_PATH}/linux/x64 ${AILIA_TFLITE_PATH}/linux/x64)
endif(WIN32)

# require ailia SDK and ailia Audio (no OpenCV)
add_subdirectory(audio_processing/silero-vad)
add_subdirectory(audio_processing/gpt-sovits)

add_subdirectory(natural_language_processing/g2p_en)

# require OpenCV
if(WITH_OPENCV)
    add_subdirectory(background_removal/u2net)

    add_subdirectory(image_classification/resnet50)
    add_subdirectory(image_classification/clip)

    add_subdirectory(face_detection/yolov3-face)
    add_subdirectory(face_detection/retinaface)

    add_subdirectory(face_identification/arcface)

    add_subdirectory(face_recognition/face_alignment)
    add_subdirectory(face_recognition/mediapipe_iris)

    add_subdirectory(object_detection/yolov3-tiny)
    add_subdirectory(object_detection/yolox)
    add_subdirectory(object_detection/m2det)
    add_subdirectory(object_detection/yolox_tflite)

    add_subdirectory(object_tracking/bytetrack)

    add_subdirectory(pose_estimation/lightweight-human-pose-estimation)
endif()

# require ailia Tokenizer and ailia Speech
if(UNIX AND CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7-a")
else()
add_subdirectory(audio_processing/clap)
add_subdirectory(audio_processing/whisper)
add_subdirectory(natural_language_processing/bert_maskedlm)
add_subdirectory(natural_language_processing/fugumt-ja-en)
add_subdirectory(natural_language_processing/fugumt-en-ja)
add_subdirectory(natural_language_processing/sentence_transformers)
add_subdirectory(natural_language_processing/t5_whisper_medical)
add_subdirectory(natural_language_processing/multilingual-e5)
endif()

# require ailia Voice
if(UNIX AND CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7-a")
else()
add_subdirectory(audio_processing/gpt-sovits-v2-pro)
endif()
