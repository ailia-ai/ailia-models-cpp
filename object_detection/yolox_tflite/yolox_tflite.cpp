/*******************************************************************
*
*    DESCRIPTION:
*      AILIA TFLite Runtime YOLOX Sample
*    AUTHOR:
*      AXELL CORPORATION, ax Inc.
*
*******************************************************************/

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "ailia_tflite.h"

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
typedef LARGE_INTEGER ailia_time_t;
static void get_ailia_time(ailia_time_t *t) {
    QueryPerformanceCounter(t);
}
static int64_t calc_predict_time(ailia_time_t start, ailia_time_t finish) {
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    return (finish.QuadPart - start.QuadPart) * (int64_t)1000000 / freq.QuadPart;
}
#else
#include <time.h>
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
typedef int64_t ailia_time_t;
static void get_ailia_time(ailia_time_t *t) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    *t = now.tv_sec * (int64_t)1000000000 + now.tv_nsec;
}
static int64_t calc_predict_time(ailia_time_t start, ailia_time_t finish) {
    return (finish - start) / (int64_t)1000;
}
#endif

// ======================
// Parameters
// ======================

#define WEIGHT_PATH "yolox_tiny_full_integer_quant.opt.tflite"
#define IMAGE_PATH  "input.jpg"

#define DETECTION_THRESHOLD 0.5f

static const std::vector<const char*> COCO_CATEGORY = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

static std::string weight(WEIGHT_PATH);
static std::string image_path(IMAGE_PATH);
static int args_env_id = AILIA_TFLITE_ENV_REFERENCE;

// ======================
// File I/O
// ======================

static int load_file(std::vector<uint8_t>& buf, const char *path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    buf.resize(size);
    size_t r = fread(buf.data(), size, 1, fp);
    fclose(fp);
    return (r == 1) ? 0 : -1;
}

// ======================
// Pre/Post Processing
// ======================

static int load_image(AILIATFLiteTensorType input_tensor_type,
                      uint8_t* input_buffer, const int32_t* input_shape,
                      const std::string& path) {
    cv::Mat img = cv::imread(path);
    if (img.empty()) {
        return -1;
    }

    const bool channel_last = (input_shape[3] == 3);
    const int target_h = channel_last ? input_shape[1] : input_shape[2];
    const int target_w = channel_last ? input_shape[2] : input_shape[3];

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(target_w, target_h));

    PRINT_OUT("input image: %s, channel_last=%d, target=%dx%d\n",
              path.c_str(), channel_last ? 1 : 0, target_w, target_h);

    const int H = target_h;
    const int W = target_w;
    if (input_tensor_type == AILIA_TFLITE_TENSOR_TYPE_FLOAT32) {
        float* dst = reinterpret_cast<float*>(input_buffer);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                cv::Vec3b px = resized.at<cv::Vec3b>(y, x); // BGR
                if (channel_last) {
                    dst[(y * W + x) * 3 + 0] = (float)px[2]; // R
                    dst[(y * W + x) * 3 + 1] = (float)px[1]; // G
                    dst[(y * W + x) * 3 + 2] = (float)px[0]; // B
                } else {
                    dst[0 * H * W + y * W + x] = (float)px[2];
                    dst[1 * H * W + y * W + x] = (float)px[1];
                    dst[2 * H * W + y * W + x] = (float)px[0];
                }
            }
        }
    } else {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                cv::Vec3b px = resized.at<cv::Vec3b>(y, x); // BGR
                if (channel_last) {
                    input_buffer[(y * W + x) * 3 + 0] = px[2];
                    input_buffer[(y * W + x) * 3 + 1] = px[1];
                    input_buffer[(y * W + x) * 3 + 2] = px[0];
                } else {
                    input_buffer[0 * H * W + y * W + x] = px[2];
                    input_buffer[1 * H * W + y * W + x] = px[1];
                    input_buffer[2 * H * W + y * W + x] = px[0];
                }
            }
        }
        if (input_tensor_type == AILIA_TFLITE_TENSOR_TYPE_INT8) {
            const size_t n = (size_t)H * W * 3;
            for (size_t i = 0; i < n; ++i) {
                input_buffer[i] = input_buffer[i] - 0x80;
            }
        }
    }
    return 0;
}

static inline float dequantize(uint8_t val, float scale, int64_t zero_point,
                               AILIATFLiteTensorType tensor_type) {
    if (tensor_type == AILIA_TFLITE_TENSOR_TYPE_INT8) {
        int8_t v = *reinterpret_cast<int8_t*>(&val);
        return ((float)v - (float)zero_point) * scale;
    }
    return ((float)val - (float)zero_point) * scale;
}

static void post_process_yolox(const int32_t* input_shape, const int32_t* output_shape,
                               const uint8_t* output_buffer,
                               AILIATFLiteTensorType output_tensor_type,
                               float quant_scale, int64_t quant_zero_point,
                               const std::vector<const char*>& category, float threshold) {
    int ih, iw;
    if (input_shape[3] == 3) {
        ih = input_shape[1];
        iw = input_shape[2];
    } else {
        ih = input_shape[2];
        iw = input_shape[3];
    }
    const int oh[3] = { ih / 8,  ih / 16, ih / 32 };
    const int ow[3] = { iw / 8,  iw / 16, iw / 32 };
    const int num_cells = oh[0] * ow[0] + oh[1] * ow[1] + oh[2] * ow[2];
    const int num_category = (int)category.size();
    const int num_elements = 5 + num_category;
    if (num_cells != output_shape[1] || num_elements != output_shape[2]) {
        PRINT_ERR("error! yolox output_shape[1,2]=(%d, %d) != (%d, %d)\n",
                  output_shape[1], output_shape[2], num_cells, num_elements);
        return;
    }

    const uint8_t* buf = output_buffer;
    for (int s = 0; s < 3; ++s) {
        const float stride = powf(2.f, 3.f + s);
        for (int y = 0; y < oh[s]; ++y) {
            for (int x = 0; x < ow[s]; ++x) {
                uint8_t max_score = 0;
                int max_class = 0;
                const uint8_t* cats = buf + 5;
                for (int cls = 0; cls < num_category; ++cls) {
                    if (output_tensor_type == AILIA_TFLITE_TENSOR_TYPE_INT8) {
                        int8_t v = *reinterpret_cast<const int8_t*>(cats + cls);
                        if ((uint8_t)v > max_score) {
                            max_score = (uint8_t)v;
                            max_class = cls;
                        }
                    } else {
                        if (cats[cls] > max_score) {
                            max_score = cats[cls];
                            max_class = cls;
                        }
                    }
                }
                float score = dequantize(max_score, quant_scale, quant_zero_point, output_tensor_type);
                float c = dequantize(buf[4], quant_scale, quant_zero_point, output_tensor_type);
                score *= c;
                if (score >= threshold) {
                    float cx = dequantize(buf[0], quant_scale, quant_zero_point, output_tensor_type);
                    float cy = dequantize(buf[1], quant_scale, quant_zero_point, output_tensor_type);
                    float w  = dequantize(buf[2], quant_scale, quant_zero_point, output_tensor_type);
                    float h  = dequantize(buf[3], quant_scale, quant_zero_point, output_tensor_type);
                    float bb_cx = (cx + x) * stride;
                    float bb_cy = (cy + y) * stride;
                    float bb_w  = expf(w) * stride + 1.f;
                    float bb_h  = expf(h) * stride + 1.f;
                    PRINT_OUT("class[%d, %s], score=%f, bb=[cx=%f, cy=%f, w=%f, h=%f]\n",
                              max_class, category[max_class], score, bb_cx, bb_cy, bb_w, bb_h);
                }
                buf += num_elements;
            }
        }
    }
}

static void post_process_yolox_fp32(const int32_t* input_shape, const int32_t* output_shape,
                                    const float* output_buffer,
                                    const std::vector<const char*>& category, float threshold) {
    int ih, iw;
    if (input_shape[3] == 3) {
        ih = input_shape[1];
        iw = input_shape[2];
    } else {
        ih = input_shape[2];
        iw = input_shape[3];
    }
    const int oh[3] = { ih / 8,  ih / 16, ih / 32 };
    const int ow[3] = { iw / 8,  iw / 16, iw / 32 };
    const int num_cells = oh[0] * ow[0] + oh[1] * ow[1] + oh[2] * ow[2];
    const int num_category = (int)category.size();
    const int num_elements = 5 + num_category;
    if (num_cells != output_shape[1] || num_elements != output_shape[2]) {
        PRINT_ERR("error! yolox output_shape[1,2]=(%d, %d) != (%d, %d)\n",
                  output_shape[1], output_shape[2], num_cells, num_elements);
        return;
    }

    const float* buf = output_buffer;
    for (int s = 0; s < 3; ++s) {
        const float stride = powf(2.f, 3.f + s);
        for (int y = 0; y < oh[s]; ++y) {
            for (int x = 0; x < ow[s]; ++x) {
                float max_score = 0;
                int max_class = 0;
                const float* cats = buf + 5;
                for (int cls = 0; cls < num_category; ++cls) {
                    if (cats[cls] > max_score) {
                        max_score = cats[cls];
                        max_class = cls;
                    }
                }
                float score = max_score * buf[4];
                if (score >= threshold) {
                    float bb_cx = (buf[0] + x) * stride;
                    float bb_cy = (buf[1] + y) * stride;
                    float bb_w  = expf(buf[2]) * stride + 1.f;
                    float bb_h  = expf(buf[3]) * stride + 1.f;
                    PRINT_OUT("class[%d, %s], score=%f, bb=[cx=%f, cy=%f, w=%f, h=%f]\n",
                              max_class, category[max_class], score, bb_cx, bb_cy, bb_w, bb_h);
                }
                buf += num_elements;
            }
        }
    }
}

// ======================
// Argument Parser
// ======================

static void print_usage() {
    PRINT_OUT("usage: yolox_tflite [-h] [-i FILE] [-w MODEL] [-e ENV_ID]\n");
}

static void print_help() {
    PRINT_OUT("\n");
    PRINT_OUT("yolox model with ailia TFLite Runtime\n");
    PRINT_OUT("\n");
    PRINT_OUT("optional arguments:\n");
    PRINT_OUT("  -h, --help            show this help message and exit\n");
    PRINT_OUT("  -i FILE, --input FILE\n");
    PRINT_OUT("                        The input image file (default: input.jpg)\n");
    PRINT_OUT("  -w FILE, --weight FILE\n");
    PRINT_OUT("                        The .tflite model file (default: %s)\n", WEIGHT_PATH);
    PRINT_OUT("  -e ENV_ID, --env_id ENV_ID\n");
    PRINT_OUT("                        AILIA_TFLITE_ENV_* (default: 0=REFERENCE)\n");
}

static int argument_parser(int argc, char** argv) {
    int status = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (status == 0) {
            if (arg == "-h" || arg == "--help") {
                print_usage();
                print_help();
                return -1;
            } else if (arg == "-i" || arg == "--input") {
                status = 1;
            } else if (arg == "-w" || arg == "--weight") {
                status = 2;
            } else if (arg == "-e" || arg == "--env_id") {
                status = 3;
            } else {
                print_usage();
                PRINT_ERR("yolox_tflite: error: unrecognized argument: %s\n", arg.c_str());
                return -1;
            }
        } else {
            switch (status) {
                case 1: image_path = arg; break;
                case 2: weight = arg; break;
                case 3: args_env_id = atoi(arg.c_str()); break;
            }
            status = 0;
        }
    }
    return AILIA_TFLITE_STATUS_SUCCESS;
}

// ======================
// Main
// ======================

int main(int argc, char** argv) {
    if (argument_parser(argc, argv) != AILIA_TFLITE_STATUS_SUCCESS) {
        return -1;
    }

    std::vector<uint8_t> tflite;
    if (load_file(tflite, weight.c_str()) != 0) {
        PRINT_ERR("Failed to load model: %s\n", weight.c_str());
        return -1;
    }

    struct AILIATFLiteInstance* instance = NULL;
    AILIATFLiteStatus status = ailiaTFLiteCreate(&instance, tflite.data(), tflite.size(),
                                                 NULL, NULL, NULL, NULL,
                                                 args_env_id,
                                                 AILIA_TFLITE_MEMORY_MODE_DEFAULT,
                                                 AILIA_TFLITE_FLAG_NONE);
    if (status != AILIA_TFLITE_STATUS_SUCCESS) {
        PRINT_ERR("ailiaTFLiteCreate failed %d\n", status);
        if (status == AILIA_TFLITE_STATUS_LICENSE_NOT_FOUND) {
            PRINT_ERR("License file not found.\n");
        }
        if (status == AILIA_TFLITE_STATUS_LICENSE_EXPIRED) {
            PRINT_ERR("License file expired.\n");
        }
        return -1;
    }

    /* dump device info */
    {
        size_t device_count = 0;
        ailiaTFLiteGetDeviceCount(instance, &device_count);
        PRINT_OUT("device_count = %d\n", (int)device_count);
        for (size_t d = 0; d < device_count; ++d) {
            char* name = NULL;
            char* info = NULL;
            ailiaTFLiteGetDeviceName(instance, (int32_t)d, &name);
            ailiaTFLiteGetDeviceExtraInfo(instance, (int32_t)d, &info);
            PRINT_OUT("[%zu] name='%s', info='%s'\n", d,
                      name ? name : "", info ? info : "");
        }
    }

    ailiaTFLiteAllocateTensors(instance);

    int32_t input_tensor_index = 0;
    ailiaTFLiteGetInputTensorIndex(instance, &input_tensor_index, 0);

    uint8_t* input_buffer = NULL;
    ailiaTFLiteGetTensorBuffer(instance, reinterpret_cast<void**>(&input_buffer), input_tensor_index);

    int32_t input_shape[4] = { 0, 0, 0, 0 };
    ailiaTFLiteGetTensorShape(instance, input_shape, input_tensor_index);

    AILIATFLiteTensorType input_tensor_type = AILIA_TFLITE_TENSOR_TYPE_UINT8;
    ailiaTFLiteGetTensorType(instance, &input_tensor_type, input_tensor_index);
    PRINT_OUT("input tensor type=%d, shape=%d,%d,%d,%d\n",
              input_tensor_type, input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

    if (load_image(input_tensor_type, input_buffer, input_shape, image_path) != 0) {
        PRINT_ERR("Failed to load image: %s\n", image_path.c_str());
        ailiaTFLiteDestroy(instance);
        return -1;
    }

    {
        ailia_time_t start, end;
        get_ailia_time(&start);
        status = ailiaTFLitePredict(instance);
        get_ailia_time(&end);
        if (status != AILIA_TFLITE_STATUS_SUCCESS) {
            PRINT_ERR("ailiaTFLitePredict failed %d\n", status);
            ailiaTFLiteDestroy(instance);
            return -1;
        }
        PRINT_OUT("inference time %d [ms]\n", (int32_t)(calc_predict_time(start, end) / 1000));
    }

    int32_t output_tensor_index = 0;
    ailiaTFLiteGetOutputTensorIndex(instance, &output_tensor_index, 0);

    uint8_t* output_buffer = NULL;
    ailiaTFLiteGetTensorBuffer(instance, reinterpret_cast<void**>(&output_buffer), output_tensor_index);

    int32_t output_shape[4] = { 0, 0, 0, 0 };
    ailiaTFLiteGetTensorShape(instance, output_shape, output_tensor_index);

    AILIATFLiteTensorType output_tensor_type = AILIA_TFLITE_TENSOR_TYPE_UINT8;
    ailiaTFLiteGetTensorType(instance, &output_tensor_type, output_tensor_index);
    PRINT_OUT("output tensor type=%d, shape=%d,%d,%d,%d\n",
              output_tensor_type, output_shape[0], output_shape[1], output_shape[2], output_shape[3]);

    if (output_tensor_type == AILIA_TFLITE_TENSOR_TYPE_FLOAT32) {
        post_process_yolox_fp32(input_shape, output_shape,
                                reinterpret_cast<float*>(output_buffer),
                                COCO_CATEGORY, DETECTION_THRESHOLD);
    } else if (output_tensor_type == AILIA_TFLITE_TENSOR_TYPE_UINT8 ||
               output_tensor_type == AILIA_TFLITE_TENSOR_TYPE_INT8) {
        int32_t quant_count = 0;
        ailiaTFLiteGetTensorQuantizationCount(instance, &quant_count, output_tensor_index);
        if (quant_count != 1) {
            PRINT_ERR("output tensor quant count=%d != 1\n", quant_count);
        } else {
            float quant_scale = 1.f;
            int64_t quant_zero_point = 0;
            int32_t quant_axis = 0;
            ailiaTFLiteGetTensorQuantizationScale(instance, &quant_scale, output_tensor_index);
            ailiaTFLiteGetTensorQuantizationZeroPoint(instance, &quant_zero_point, output_tensor_index);
            ailiaTFLiteGetTensorQuantizationQuantizedDimension(instance, &quant_axis, output_tensor_index);
            PRINT_OUT("output tensor scale=%f, zero_point=%lld, axis=%d\n",
                      quant_scale, (long long)quant_zero_point, quant_axis);
            post_process_yolox(input_shape, output_shape, output_buffer, output_tensor_type,
                               quant_scale, quant_zero_point,
                               COCO_CATEGORY, DETECTION_THRESHOLD);
        }
    } else {
        PRINT_ERR("unsupported output tensor type=%d\n", output_tensor_type);
    }

    ailiaTFLiteDestroy(instance);
    return 0;
}
