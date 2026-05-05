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
#include <algorithm>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "ailia_tflite.h"

#include "utils.h"
#include "webcamera_utils.h"

#include <time.h>

#if defined(_WIN32) || defined(_WIN64)
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
#else
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif

// ======================
// Parameters
// ======================

#define WEIGHT_PATH     "yolox_tiny_full_integer_quant.opt.tflite"
#define IMAGE_PATH      "input.jpg"
#define SAVE_IMAGE_PATH "output.png"

#define DETECTION_THRESHOLD 0.5f
#define IOU_THRESHOLD       0.45f
#define BENCHMARK_ITERS     5

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
static std::string video_path("0");
static std::string save_image_path(SAVE_IMAGE_PATH);

static bool benchmark = false;
static bool video_mode = false;
static int args_env_id = AILIA_TFLITE_ENV_REFERENCE;

// ======================
// Detection struct
// ======================

struct Detection {
    int class_id;
    float score;
    // bbox in input image (model input) coordinates: top-left corner + size
    float x, y, w, h;
};

// ======================
// File I/O
// ======================

static int load_file(std::vector<uint8_t>& buf, const char* path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return -1;
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    buf.resize(size);
    size_t r = fread(buf.data(), size, 1, fp);
    fclose(fp);
    return (r == 1) ? 0 : -1;
}

// ======================
// Pre-processing
// ======================

static void fill_input_tensor(AILIATFLiteTensorType input_tensor_type,
                              uint8_t* input_buffer, const int32_t* input_shape,
                              const cv::Mat& bgr) {
    const bool channel_last = (input_shape[3] == 3);
    const int H = channel_last ? input_shape[1] : input_shape[2];
    const int W = channel_last ? input_shape[2] : input_shape[3];

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(W, H));

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
                cv::Vec3b px = resized.at<cv::Vec3b>(y, x);
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
}

// ======================
// Post-processing
// ======================

static inline float dequantize(uint8_t val, float scale, int64_t zero_point,
                               AILIATFLiteTensorType tensor_type) {
    if (tensor_type == AILIA_TFLITE_TENSOR_TYPE_INT8) {
        int8_t v = *reinterpret_cast<int8_t*>(&val);
        return ((float)v - (float)zero_point) * scale;
    }
    return ((float)val - (float)zero_point) * scale;
}

static void decode_yolox_quant(const int32_t* input_shape, const int32_t* output_shape,
                               const uint8_t* output_buffer,
                               AILIATFLiteTensorType output_tensor_type,
                               float scale, int64_t zero_point,
                               int num_category, float threshold,
                               std::vector<Detection>& dets) {
    int ih = (input_shape[3] == 3) ? input_shape[1] : input_shape[2];
    int iw = (input_shape[3] == 3) ? input_shape[2] : input_shape[3];
    const int oh[3] = { ih / 8, ih / 16, ih / 32 };
    const int ow[3] = { iw / 8, iw / 16, iw / 32 };
    const int num_elements = 5 + num_category;

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
                        if ((uint8_t)v > max_score) { max_score = (uint8_t)v; max_class = cls; }
                    } else {
                        if (cats[cls] > max_score) { max_score = cats[cls]; max_class = cls; }
                    }
                }
                float score = dequantize(max_score, scale, zero_point, output_tensor_type);
                float c = dequantize(buf[4], scale, zero_point, output_tensor_type);
                score *= c;
                if (score >= threshold) {
                    float cx = dequantize(buf[0], scale, zero_point, output_tensor_type);
                    float cy = dequantize(buf[1], scale, zero_point, output_tensor_type);
                    float w  = dequantize(buf[2], scale, zero_point, output_tensor_type);
                    float h  = dequantize(buf[3], scale, zero_point, output_tensor_type);
                    Detection det;
                    det.class_id = max_class;
                    det.score = score;
                    float bb_cx = (cx + x) * stride;
                    float bb_cy = (cy + y) * stride;
                    float bb_w  = expf(w) * stride + 1.f;
                    float bb_h  = expf(h) * stride + 1.f;
                    det.x = bb_cx - bb_w * 0.5f;
                    det.y = bb_cy - bb_h * 0.5f;
                    det.w = bb_w;
                    det.h = bb_h;
                    dets.push_back(det);
                }
                buf += num_elements;
            }
        }
    }
}

static void decode_yolox_fp32(const int32_t* input_shape, const int32_t* output_shape,
                              const float* output_buffer,
                              int num_category, float threshold,
                              std::vector<Detection>& dets) {
    int ih = (input_shape[3] == 3) ? input_shape[1] : input_shape[2];
    int iw = (input_shape[3] == 3) ? input_shape[2] : input_shape[3];
    const int oh[3] = { ih / 8, ih / 16, ih / 32 };
    const int ow[3] = { iw / 8, iw / 16, iw / 32 };
    const int num_elements = 5 + num_category;

    const float* buf = output_buffer;
    for (int s = 0; s < 3; ++s) {
        const float stride = powf(2.f, 3.f + s);
        for (int y = 0; y < oh[s]; ++y) {
            for (int x = 0; x < ow[s]; ++x) {
                float max_score = 0;
                int max_class = 0;
                const float* cats = buf + 5;
                for (int cls = 0; cls < num_category; ++cls) {
                    if (cats[cls] > max_score) { max_score = cats[cls]; max_class = cls; }
                }
                float score = max_score * buf[4];
                if (score >= threshold) {
                    Detection det;
                    det.class_id = max_class;
                    det.score = score;
                    float bb_cx = (buf[0] + x) * stride;
                    float bb_cy = (buf[1] + y) * stride;
                    float bb_w  = expf(buf[2]) * stride + 1.f;
                    float bb_h  = expf(buf[3]) * stride + 1.f;
                    det.x = bb_cx - bb_w * 0.5f;
                    det.y = bb_cy - bb_h * 0.5f;
                    det.w = bb_w;
                    det.h = bb_h;
                    dets.push_back(det);
                }
                buf += num_elements;
            }
        }
    }
}

static float iou(const Detection& a, const Detection& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w);
    float y2 = std::min(a.y + a.h, b.y + b.h);
    float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    float u = a.w * a.h + b.w * b.h - inter;
    return (u > 0.f) ? (inter / u) : 0.f;
}

static std::vector<Detection> nms(std::vector<Detection> dets, float iou_threshold) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> result;
    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j] || dets[j].class_id != dets[i].class_id) continue;
            if (iou(dets[i], dets[j]) > iou_threshold) suppressed[j] = true;
        }
    }
    return result;
}

static void draw_detections(cv::Mat& img, const std::vector<Detection>& dets,
                            int input_w, int input_h) {
    const float sx = (float)img.cols / (float)input_w;
    const float sy = (float)img.rows / (float)input_h;
    for (const Detection& d : dets) {
        int x = (int)(d.x * sx);
        int y = (int)(d.y * sy);
        int w = (int)(d.w * sx);
        int h = (int)(d.h * sy);
        cv::Rect r(x, y, w, h);
        cv::rectangle(img, r, cv::Scalar(0, 255, 0), 2);
        char label[256];
        const char* name = (d.class_id >= 0 && d.class_id < (int)COCO_CATEGORY.size())
            ? COCO_CATEGORY[d.class_id] : "?";
        snprintf(label, sizeof(label), "%s %.2f", name, d.score);
        int base = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        cv::rectangle(img, cv::Point(x, y - ts.height - 4),
                      cv::Point(x + ts.width, y), cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(img, label, cv::Point(x, y - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

// ======================
// Inference helpers
// ======================

static int decode_output(struct AILIATFLiteInstance* instance,
                         const int32_t* input_shape,
                         std::vector<Detection>& dets) {
    int32_t output_tensor_index = 0;
    ailiaTFLiteGetOutputTensorIndex(instance, &output_tensor_index, 0);

    uint8_t* output_buffer = NULL;
    ailiaTFLiteGetTensorBuffer(instance, reinterpret_cast<void**>(&output_buffer), output_tensor_index);

    int32_t output_shape[4] = { 0, 0, 0, 0 };
    ailiaTFLiteGetTensorShape(instance, output_shape, output_tensor_index);

    AILIATFLiteTensorType output_tensor_type = AILIA_TFLITE_TENSOR_TYPE_UINT8;
    ailiaTFLiteGetTensorType(instance, &output_tensor_type, output_tensor_index);

    const int num_category = (int)COCO_CATEGORY.size();
    int ih = (input_shape[3] == 3) ? input_shape[1] : input_shape[2];
    const int num_cells = (ih / 8) * (ih / 8) + (ih / 16) * (ih / 16) + (ih / 32) * (ih / 32);
    if (output_shape[1] != num_cells || output_shape[2] != 5 + num_category) {
        PRINT_ERR("error! yolox output_shape[1,2]=(%d, %d) != (%d, %d)\n",
                  output_shape[1], output_shape[2], num_cells, 5 + num_category);
        return -1;
    }

    if (output_tensor_type == AILIA_TFLITE_TENSOR_TYPE_FLOAT32) {
        decode_yolox_fp32(input_shape, output_shape,
                          reinterpret_cast<const float*>(output_buffer),
                          num_category, DETECTION_THRESHOLD, dets);
    } else if (output_tensor_type == AILIA_TFLITE_TENSOR_TYPE_UINT8 ||
               output_tensor_type == AILIA_TFLITE_TENSOR_TYPE_INT8) {
        int32_t quant_count = 0;
        ailiaTFLiteGetTensorQuantizationCount(instance, &quant_count, output_tensor_index);
        if (quant_count != 1) {
            PRINT_ERR("output tensor quant count=%d != 1\n", quant_count);
            return -1;
        }
        float scale = 1.f;
        int64_t zero_point = 0;
        ailiaTFLiteGetTensorQuantizationScale(instance, &scale, output_tensor_index);
        ailiaTFLiteGetTensorQuantizationZeroPoint(instance, &zero_point, output_tensor_index);
        decode_yolox_quant(input_shape, output_shape, output_buffer, output_tensor_type,
                           scale, zero_point, num_category, DETECTION_THRESHOLD, dets);
    } else {
        PRINT_ERR("unsupported output tensor type=%d\n", output_tensor_type);
        return -1;
    }
    return 0;
}

static int predict(struct AILIATFLiteInstance* instance, const cv::Mat& bgr,
                   std::vector<Detection>& dets, int& input_w, int& input_h,
                   long& predict_time_ms) {
    int32_t input_tensor_index = 0;
    ailiaTFLiteGetInputTensorIndex(instance, &input_tensor_index, 0);

    uint8_t* input_buffer = NULL;
    ailiaTFLiteGetTensorBuffer(instance, reinterpret_cast<void**>(&input_buffer), input_tensor_index);

    int32_t input_shape[4] = { 0, 0, 0, 0 };
    ailiaTFLiteGetTensorShape(instance, input_shape, input_tensor_index);

    AILIATFLiteTensorType input_tensor_type = AILIA_TFLITE_TENSOR_TYPE_UINT8;
    ailiaTFLiteGetTensorType(instance, &input_tensor_type, input_tensor_index);

    input_w = (input_shape[3] == 3) ? input_shape[2] : input_shape[3];
    input_h = (input_shape[3] == 3) ? input_shape[1] : input_shape[2];

    fill_input_tensor(input_tensor_type, input_buffer, input_shape, bgr);

    clock_t start = clock();
    AILIATFLiteStatus status = ailiaTFLitePredict(instance);
    clock_t end = clock();
    predict_time_ms = (long)((end - start) * 1000 / CLOCKS_PER_SEC);
    if (status != AILIA_TFLITE_STATUS_SUCCESS) {
        PRINT_ERR("ailiaTFLitePredict failed %d\n", status);
        return -1;
    }

    std::vector<Detection> raw;
    if (decode_output(instance, input_shape, raw) != 0) {
        return -1;
    }
    dets = nms(raw, IOU_THRESHOLD);
    return 0;
}

// ======================
// Image / Video flows
// ======================

static int recognize_from_image(struct AILIATFLiteInstance* instance) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        PRINT_ERR("[ERROR] Failed to load image: %s\n", image_path.c_str());
        return -1;
    }
    PRINT_OUT("input image shape: (%d, %d, %d)\n", img.cols, img.rows, img.channels());

    PRINT_OUT("Start inference...\n");
    std::vector<Detection> dets;
    int input_w = 0, input_h = 0;
    long predict_ms = 0;

    if (benchmark) {
        PRINT_OUT("BENCHMARK mode\n");
        for (int i = 0; i < BENCHMARK_ITERS; i++) {
            dets.clear();
            if (predict(instance, img, dets, input_w, input_h, predict_ms) != 0) return -1;
            PRINT_OUT("\tailia processing time %ld ms\n", predict_ms);
        }
    } else {
        if (predict(instance, img, dets, input_w, input_h, predict_ms) != 0) return -1;
        PRINT_OUT("inference time %ld ms\n", predict_ms);
    }

    for (const Detection& d : dets) {
        const char* name = (d.class_id >= 0 && d.class_id < (int)COCO_CATEGORY.size())
            ? COCO_CATEGORY[d.class_id] : "?";
        PRINT_OUT("class[%d, %s], score=%f, bbox=[x=%.1f, y=%.1f, w=%.1f, h=%.1f]\n",
                  d.class_id, name, d.score, d.x, d.y, d.w, d.h);
    }

    draw_detections(img, dets, input_w, input_h);
    cv::imwrite(save_image_path, img);
    PRINT_OUT("Saved to %s\n", save_image_path.c_str());

    PRINT_OUT("Program finished successfully.\n");
    return 0;
}

static int recognize_from_video(struct AILIATFLiteInstance* instance) {
    cv::VideoCapture capture;
    if (video_path == "0") {
        PRINT_OUT("[INFO] webcamera mode is activated\n");
        capture.open(0);
        if (!capture.isOpened()) {
            PRINT_ERR("[ERROR] webcamera not found\n");
            return -1;
        }
    } else {
        if (!check_file_existance(video_path.c_str())) {
            PRINT_ERR("[ERROR] \"%s\" not found\n", video_path.c_str());
            return -1;
        }
        capture.open(video_path);
    }

    while (true) {
        cv::Mat frame;
        capture >> frame;
        if ((char)cv::waitKey(1) == 'q' || frame.empty()) break;

        std::vector<Detection> dets;
        int input_w = 0, input_h = 0;
        long predict_ms = 0;
        if (predict(instance, frame, dets, input_w, input_h, predict_ms) != 0) {
            return -1;
        }
        draw_detections(frame, dets, input_w, input_h);
        cv::imshow("frame", frame);
    }
    capture.release();
    cv::destroyAllWindows();

    PRINT_OUT("Program finished successfully.\n");
    return 0;
}

// ======================
// Argument Parser
// ======================

static void print_usage() {
    PRINT_OUT("usage: yolox_tflite [-h] [-i IMAGE] [-v VIDEO] [-s SAVE_IMAGE_PATH] "
              "[-w MODEL] [-b] [-e ENV_ID]\n");
}

static void print_help() {
    PRINT_OUT("\n");
    PRINT_OUT("yolox model with ailia TFLite Runtime\n");
    PRINT_OUT("\n");
    PRINT_OUT("optional arguments:\n");
    PRINT_OUT("  -h, --help            show this help message and exit\n");
    PRINT_OUT("  -i IMAGE, --input IMAGE\n");
    PRINT_OUT("                        The input image path (default: input.jpg)\n");
    PRINT_OUT("  -v VIDEO, --video VIDEO\n");
    PRINT_OUT("                        The input video path. If 0, the webcam is used.\n");
    PRINT_OUT("  -s SAVE_IMAGE_PATH, --savepath SAVE_IMAGE_PATH\n");
    PRINT_OUT("                        Save path for the output image (default: output.png)\n");
    PRINT_OUT("  -w FILE, --weight FILE\n");
    PRINT_OUT("                        The .tflite model file (default: %s)\n", WEIGHT_PATH);
    PRINT_OUT("  -b, --benchmark       Run inference 5 times to measure performance\n");
    PRINT_OUT("                        (image mode only).\n");
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
            } else if (arg == "-v" || arg == "--video") {
                video_mode = true;
                status = 2;
            } else if (arg == "-s" || arg == "--savepath") {
                status = 3;
            } else if (arg == "-w" || arg == "--weight") {
                status = 4;
            } else if (arg == "-b" || arg == "--benchmark") {
                benchmark = true;
            } else if (arg == "-e" || arg == "--env_id") {
                status = 5;
            } else {
                print_usage();
                PRINT_ERR("yolox_tflite: error: unrecognized argument: %s\n", arg.c_str());
                return -1;
            }
        } else {
            switch (status) {
                case 1: image_path = arg; break;
                case 2: video_path = arg; break;
                case 3: save_image_path = arg; break;
                case 4: weight = arg; break;
                case 5: args_env_id = atoi(arg.c_str()); break;
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

    ailiaTFLiteAllocateTensors(instance);

    int rc = video_mode ? recognize_from_video(instance) : recognize_from_image(instance);

    ailiaTFLiteDestroy(instance);
    return rc;
}
