# ailia-models-cpp

The collection of pre-trained, state-of-the-art models for C++.

## About ailia SDK

ailia SDK is a cross-platform high speed inference SDK. The ailia SDK provides a consistent C++ API on Windows, Mac, Linux, iOS, Android and Jetson. It supports Unity, Python and JNI for efficient AI implementation. The ailia SDK makes great use of the GPU via Vulkan and Metal to serve accelerated computing.

## Install ailia SDK

### Download ailia SDK

The ailia SDK can be installed from a git submodule.

```
git submodule init
git submodule update
```

#### Mac

On macOS, the bundled `.dylib` files come with the `com.apple.quarantine` attribute, which causes Gatekeeper to reject them with errors such as `libailia_voice.dylib is damaged`. Strip the attribute by running the helper script after `git submodule update`:

```
./xattr.sh
```

### Download license file

Download the ailia SDK license file with the following command. The license file is valid for one month.

```
cd ailia
python3 download_license.py
```

Alternatively, the license file can be obtained by requesting the evaluation version of the ailia SDK.

[Request trial version](https://axinc.jp/en/trial/)

### Install dependent libraries

#### Windows

cmake, opencv and Visual Studio 2019 or newer are required.
- https://cmake.org/download/
- https://opencv.org/releases/

Please set the OpenCV path to the environment variable.

- OpenCV_DIR: C:\opencv\build

#### Mac

Xcode Commandline Tools are required, they can be installed by running the command.

```
xcode-select --install
```

cmake and OpenCV is required, it can be installed by running the command.

```
brew install cmake
brew install opencv
```

#### Linux

cmake and OpenCV is required, it can be installed by running the command.

```
apt install cmake
apt install libopencv-dev
```

### Build

Please run the following command in the root folder of your local repository.

```
cmake .
cmake --build .
```

To build only the audio/NLP samples (e.g. whisper and gpt-sovits-v2-pro) without OpenCV, pass `-DWITH_OPENCV=OFF`.

```
cmake -DWITH_OPENCV=OFF .
cmake --build .
```

### Run

Move to the model folder, execute sh or bat, then the model file will be downloaded and the model will run.

### Windows

In the case of Windows, please copy dll files such as ailia.dll to the execution directory.

```
cd object_detection/yolox
./yolox.bat
```

#### Mac and Linux

```
cd object_detection/yolox
./yolox.sh
```

You can use the web camera by adding the -v 0 option.

```
cd object_detection/yolox
./yolox.sh -v 0
```

# Supporting Models

## Audio processing

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
| [silero-vad](/audio_processing/silero-vad/) | [Silero VAD](https://github.com/snakers4/silero-vad) | Pytorch | 1.2.15 and later | |
| [clap](/audio_processing/clap/) | [CLAP](https://github.com/LAION-AI/CLAP) | Pytorch | 1.3.0 and later | |
| [whisper](/audio_processing/whisper/) | [Whisper](https://github.com/openai/whisper) | Pytorch | 1.2.16 and later | [JP](https://medium.com/axinc/whisper-%E6%97%A5%E6%9C%AC%E8%AA%9E%E3%82%92%E5%90%AB%E3%82%8099%E8%A8%80%E8%AA%9E%E3%82%92%E8%AA%8D%E8%AD%98%E3%81%A7%E3%81%8D%E3%82%8B%E9%9F%B3%E5%A3%B0%E8%AA%8D%E8%AD%98%E3%83%A2%E3%83%87%E3%83%AB-b6e578f55c87) |
| [gpt-sovits](/audio_processing/gpt-sovits/) | [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | Pytorch | 1.4.0 and later | [EN](https://medium.com/axinc-ai/gpt-sovits-a-zero-shot-speech-synthesis-model-with-customizable-fine-tuning-e4c72cd75d87) [JP](https://medium.com/axinc/gpt-sovits-%E3%83%95%E3%82%A1%E3%82%A4%E3%83%B3%E3%83%81%E3%83%A5%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0%E3%81%A7%E3%81%8D%E3%82%8B0%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%E3%81%AE%E9%9F%B3%E5%A3%B0%E5%90%88%E6%88%90%E3%83%A2%E3%83%87%E3%83%AB-2212eeb5ad20) |
| [gpt-sovits-v2-pro](/audio_processing/gpt-sovits-v2-pro/) | [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | Pytorch | 1.4.0 and later | [EN](https://medium.com/axinc-ai/ailia-ai-voice-voice-synthesis-library-for-unity-and-c-76cd9137a06d) [JP](https://medium.com/axinc/ailia-ai-voice-unity%E3%82%84c-%E3%81%8B%E3%82%89%E4%BD%BF%E7%94%A8%E3%81%A7%E3%81%8D%E3%82%8Bai%E9%9F%B3%E5%A3%B0%E5%90%88%E6%88%90%E3%83%A9%E3%82%A4%E3%83%96%E3%83%A9%E3%83%AA-586e17aa717a) |

## Background removal

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
| [U-2-Net](/background_removal/u2net/) | [U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://github.com/NathanUA/U-2-Net) | Pytorch | 1.2.2 and later |

## Image classification

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
| [resnet50](/image_classification/resnet50/) | [Deep Residual Learning for Image Recognition]( https://github.com/KaimingHe/deep-residual-networks) | Chainer | 1.2.0 and later |
| [clip](/image_classification/clip/) | [CLIP](https://github.com/openai/CLIP) | Pytorch | 1.2.9 and later |

## Face detection

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
| [yolov3-face](/face_detection/yolov3-face/) | [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/) | ONNX Runtime | 1.2.1 and later |
| [retinaface](/face_detection/retinaface/) | [RetinaFace: Single-stage Dense Face Localisation in the Wild.](https://github.com/biubug6/Pytorch_Retinaface) | Pytorch | 1.2.5 and later |

## Face identification

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
|[arcface](/face_identification/arcface/) | [pytorch implement of arcface](https://github.com/ronghuaiyang/arcface-pytorch) | Pytorch | 1.2.1 and later |

## Face recognition

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
|[face_alignment](/face_recognition/face_alignment/)| [2D and 3D Face alignment library build using pytorch](https://github.com/1adrianb/face-alignment) | Pytorch | 1.2.1 and later |
|[mediapipe_iris](/face_recognition/mediapipe_iris/) | [irislandmarks.pytorch](https://github.com/cedriclmenard/irislandmarks.pytorch) | Pytorch | 1.2.2 and later |

## Natural language processing

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
|[fugumt-en-ja](/natural_language_processing/fugumt-en-ja) | [Fugu-Machine Translator](https://github.com/s-taka/fugumt) | Pytorch | 1.2.9 and later |
|[fugumt-ja-en](/natural_language_processing/fugumt-ja-en) | [Fugu-Machine Translator](https://github.com/s-taka/fugumt) | Pytorch | 1.2.10 and later |
|[bert_maskedlm](/natural_language_processing/bert_maskedlm) | [huggingface/transformers](https://github.com/huggingface/transformers) | Pytorch | 1.2.5 and later |
|[sentence_transformers](/natural_language_processing/sentence_transformers) | [sentence transformers](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) | Pytorch | 1.2.7 and later |
|[t5_whisper_medical](/natural_language_processing/t5_whisper_medical) | error correction of medical terms using t5 | Pytorch | 1.2.13 and later | |
|[multilingual-e5](/natural_language_processing/multilingual-e5) | [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) | Pytorch | 1.2.15 and later | [JP](https://medium.com/axinc/multilingual-e5-%E5%A4%9A%E8%A8%80%E8%AA%9E%E3%81%AE%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88%E3%82%92embedding%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-71f1dec7c4f0) |
|[g2p_en](/natural_language_processing/g2p_en) | [g2p_en](https://github.com/Kyubyong/g2p) | Pytorch | 1.2.14 and later | |

## Large language model

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
|[gemma4](/large_language_model/gemma4) | Gemma 4n (E2B) text chat (gguf) | gguf | 1.4.1 and later | |
|[gemma4-multimodal](/large_language_model/gemma4) | Gemma 4n (E2B) multimodal (gguf + mmproj) | gguf | 1.4.1 and later | |

## Object detection

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
| [yolov3-tiny](/object_detection/yolov3-tiny/) | [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/) | ONNX Runtime | 1.2.1 and later |
| [m2det](/object_detection/m2det/) | [M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network](https://github.com/qijiezhao/M2Det) | Pytorch | 1.2.3 and later |
| [yolox](/object_detection/yolox/) | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Pytorch | 1.2.6 and later |
| [yolox_tflite](/object_detection/yolox_tflite/) | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Pytorch | 1.1.0 and later |

## Object tracking

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
| [bytetrack](/object_tracking/bytetrack/) | [ByteTrack](https://github.com/ifzhang/ByteTrack) | Pytorch | 1.2.6 and later |

## Pose estimation

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
|[lightweight-human-pose-estimation](/pose_estimation/lightweight-human-pose-estimation/) | [Fast and accurate human pose estimation in PyTorch. Contains implementation of "Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose" paper.](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) | Pytorch | 1.2.1 and later |

# Other languages

- [python version](https://github.com/ailia-ai/ailia-models)
- [unity version](https://github.com/ailia-ai/ailia-models-unity)
- [flutter version](https://github.com/ailia-ai/ailia-models-flutter)
- [rust version](https://github.com/ailia-ai/ailia-models-rust)
