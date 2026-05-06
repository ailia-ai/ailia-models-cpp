/*******************************************************************
*
*    DESCRIPTION:
*      AILIA yolov3-tiny sample
*    AUTHOR:
*
*    DATE:2020/08/20
*
*******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#undef UNICODE

#include "ailia.h"
#include "ailia_detector.h"
#include "utils.h"
#include "detector_utils.h"
#include "webcamera_utils.h"


// ======================
// Parameters
// ======================

#define WEIGHT_PATH "yolov3-tiny.opt.onnx"
#define MODEL_PATH  "yolov3-tiny.opt.onnx.prototxt"

#define IMAGE_PATH      "couple.jpg"
#define SAVE_IMAGE_PATH "output.png"

#define MODEL_INPUT_WIDTH  416
#define MODEL_INPUT_HEIGHT 416
#define IMAGE_WIDTH        416 // for video mode
#define IMAGE_HEIGHT       416 // for video mode

#define THRESHOLD 0.4f
#define IOU       0.45f

#if defined(_WIN32) || defined(_WIN64)
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
#else
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif

#define BENCHMARK_ITERS 5

static std::string weight(WEIGHT_PATH);
static std::string model(MODEL_PATH);

static std::string image_path(IMAGE_PATH);
static std::string video_path("0");
static std::string save_image_path(SAVE_IMAGE_PATH);

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

static bool benchmark  = false;
static bool video_mode = false;


// ======================
// Arguemnt Parser
// ======================

static void print_usage()
{
    PRINT_OUT("usage: yolov3-tiny [-h] [-i IMAGE] [-v VIDEO] [-s SAVE_IMAGE_PATH] [-b]\n");
    return;
}


static void print_help()
{
    PRINT_OUT("\n");
    PRINT_OUT("yolov3 tiny model\n");
    PRINT_OUT("\n");
    PRINT_OUT("optional arguments:\n");
    PRINT_OUT("  -h, --help            show this help message and exit\n");
    PRINT_OUT("  -i IMAGE, --input IMAGE\n");
    PRINT_OUT("                        The input image path.\n");
    PRINT_OUT("  -v VIDEO, --video VIDEO\n");
    PRINT_OUT("                        The input video path. If the VIDEO argument is set to\n");
    PRINT_OUT("                        0, the webcam input will be used.\n");
    PRINT_OUT("  -s SAVE_IMAGE_PATH, --savepath SAVE_IMAGE_PATH\n");
    PRINT_OUT("                        Save path for the output image.\n");
    PRINT_OUT("  -b, --benchmark       Running the inference on the same input 5 times to\n");
    PRINT_OUT("                        measure execution performance. (Cannot be used in\n");
    PRINT_OUT("                        video mode)\n");
    return;
}


static void print_error(std::string arg)
{
    PRINT_ERR("yolov3-tiny: error: unrecognized arguments: %s\n", arg.c_str());
    return;
}


static int argument_parser(int argc, char **argv)
{
    int status = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (status == 0) {
            if (arg == "-i" || arg == "--input") {
                status = 1;
            }
            else if (arg == "-v" || arg == "--video") {
                video_mode = true;
                status = 2;
            }
            else if (arg == "-s" || arg == "--savepath") {
                status = 3;
            }
            else if (arg == "-b" || arg == "--benchmark") {
                benchmark = true;
            }
            else if (arg == "-h" || arg == "--help") {
                print_usage();
                print_help();
                return -1;
            }
            else {
                print_usage();
                print_error(arg);
                return -1;
            }
        }
        else if (arg[0] != '-') {
            switch (status) {
            case 1:
                image_path = arg;
                break;
            case 2:
                video_path = arg;
                break;
            case 3:
                save_image_path = arg;
                break;
            default:
                print_usage();
                print_error(arg);
                return -1;
            }
            status = 0;
        }
        else {
            print_usage();
            print_error(arg);
            return -1;
        }
    }

    return AILIA_STATUS_SUCCESS;
}


// ======================
// Main functions
// ======================

static int recognize_from_image(AILIADetector* detector)
{
    // prepare input data
    cv::Mat img;
    int status = load_image(img, image_path.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }
    PRINT_OUT("input image shape: (%d, %d, %d)\n",
              img.cols, img.rows, img.channels());

    // inference
    PRINT_OUT("Start inference...\n");
    if (benchmark) {
        PRINT_OUT("BENCHMARK mode\n");
        for (int i = 0; i < BENCHMARK_ITERS; i++) {
            clock_t start = clock();
            status = ailiaDetectorCompute(detector, img.data,
                                          img.cols*4, img.cols, img.rows,
                                          AILIA_IMAGE_FORMAT_BGRA, THRESHOLD, IOU);
            clock_t end = clock();
            if (status != AILIA_STATUS_SUCCESS) {
                PRINT_ERR("ailiaDetectorCompute failed %d\n", status);
                return -1;
            }
            PRINT_OUT("\tailia processing time %ld ms\n", ((end-start)*1000)/CLOCKS_PER_SEC);
        }
    }
    else {
        status = ailiaDetectorCompute(detector, img.data,
                                      img.cols*4, img.cols, img.rows,
                                      AILIA_IMAGE_FORMAT_BGRA, THRESHOLD, IOU);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaDetectorCompute failed %d\n", status);
            return -1;
        }
    }

    status = plot_result(detector, img, COCO_CATEGORY);
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }

    cv::imwrite(save_image_path.c_str(), img);

    PRINT_OUT("Program finished successfully.\n");

    return AILIA_STATUS_SUCCESS;
}


static int recognize_from_video(AILIADetector* detector)
{
    // inference
    cv::VideoCapture capture;
    if (video_path == "0") {
        PRINT_OUT("[INFO] webcamera mode is activated\n");
        capture = cv::VideoCapture(0);
        if (!capture.isOpened()) {
            PRINT_ERR("[ERROR] webcamera not found\n");
            return -1;
        }
    }
    else {
        if (check_file_existance(video_path.c_str())) {
            capture = cv::VideoCapture(video_path.c_str());
        }
        else {
            PRINT_ERR("[ERROR] \"%s\" not found\n", video_path.c_str());
            return -1;
        }
    }

    while (1) {
        cv::Mat frame;
        capture >> frame;
        if ((char)cv::waitKey(1) == 'q' || frame.empty()) {
            break;
        }
        cv::Mat resized_img, img;
        adjust_frame_size(frame, resized_img, IMAGE_WIDTH, IMAGE_HEIGHT);
        cv::cvtColor(resized_img, img, cv::COLOR_BGR2BGRA);

        int status = ailiaDetectorCompute(detector, img.data,
                                          MODEL_INPUT_WIDTH*4, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT,
                                          AILIA_IMAGE_FORMAT_BGRA, THRESHOLD, IOU);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaDetectorCompute failed %d\n", status);
            return -1;
        }

        status = plot_result(detector, resized_img, COCO_CATEGORY, false);
        if (status != AILIA_STATUS_SUCCESS) {
            return -1;
        }
        cv::imshow("frame", resized_img);
    }
    capture.release();
    cv::destroyAllWindows();

    PRINT_OUT("Program finished successfully.\n");

    return AILIA_STATUS_SUCCESS;
}


int main(int argc, char **argv)
{
    int status = argument_parser(argc, argv);
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }

    // net initialize
    AILIANetwork *ailia;
    int env_id = AILIA_ENVIRONMENT_ID_AUTO;
    status = ailiaCreate(&ailia, env_id, AILIA_MULTITHREAD_AUTO);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaCreate failed %d\n", status);
        return -1;
    }

    AILIAEnvironment *env_ptr = nullptr;
    status = ailiaGetSelectedEnvironment(ailia, &env_ptr, AILIA_ENVIRONMENT_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetSelectedEnvironment failed %d\n", status);
        ailiaDestroy(ailia);
        return -1;
    }

//    PRINT_OUT("env_id: %d\n", env_ptr->id);
    PRINT_OUT("env_name: %s\n", env_ptr->name);

    status = ailiaOpenStreamFile(ailia, model.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaOpenStreamFile failed %d\n", status);
        PRINT_ERR("ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(ailia));
        ailiaDestroy(ailia);
        return -1;
    }

    status = ailiaOpenWeightFile(ailia, weight.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
        ailiaDestroy(ailia);
        return -1;
    }
    const unsigned int flags = AILIA_DETECTOR_FLAG_NORMAL;

    AILIADetector *detector;
    status = ailiaCreateDetector(&detector, ailia,
                                 AILIA_NETWORK_IMAGE_FORMAT_RGB,
                                 AILIA_NETWORK_IMAGE_CHANNEL_FIRST,
                                 AILIA_NETWORK_IMAGE_RANGE_UNSIGNED_FP32,
                                 AILIA_DETECTOR_ALGORITHM_YOLOV3,
                                 (unsigned int)COCO_CATEGORY.size(), flags);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaCreateDetector failed %d\n", status);
        ailiaDestroy(ailia);
        return -1;
    }

    status = ailiaDetectorSetInputShape(detector, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaDetectorSetInputShape(w=%u, h=%u) failed %d\n",
              MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, status);
        ailiaDestroyDetector(detector);
        ailiaDestroy(ailia);
        return -1;
    }

    if (video_mode) {
        status = recognize_from_video(detector);
    }
    else {
        status = recognize_from_image(detector);
    }

    ailiaDestroyDetector(detector);
    ailiaDestroy(ailia);

    return status;
}
