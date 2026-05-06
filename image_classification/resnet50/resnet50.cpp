/*******************************************************************
*
*    DESCRIPTION:
*      AILIA resnet50 sample
*    AUTHOR:
*
*    DATE:2020/08/27
*
*******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#define sleep(n) Sleep(n*1000)
#else
#include <unistd.h>
#endif

#undef UNICODE

#include "ailia.h"
#include "ailia_classifier.h"
#include "resnet50_labels.h"
#include "utils.h"
#include "webcamera_utils.h"


// ======================
// Parameters
// ======================

#define WEIGHT_PATH "resnet50.opt.onnx"
#define MODEL_PATH  "resnet50.opt.onnx.prototxt"

#define IMAGE_PATH  "pizza.jpg"

#define IMAGE_WIDTH  224 // for video mode
#define IMAGE_HEIGHT 224 // for video mode

#define MAX_CLASS_COUNT 5
#define SLEEP_TIME      3

#if defined(_WIN32) || defined(_WIN64)
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
#else
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif

#define BENCHMARK_ITERS 5

static const std::vector<const char*> MODEL_NAMES = {"resnet50.opt", "resnet50", "resnet50_pytorch"};

static std::string weight(WEIGHT_PATH);
static std::string model(MODEL_PATH);

static std::string image_path(IMAGE_PATH);
static std::string video_path("0");

static bool benchmark  = false;
static bool video_mode = false;


// ======================
// Arguemnt Parser
// ======================

static void print_usage()
{
    PRINT_OUT("usage: resnet50 [-h] [-i IMAGE] [-v VIDEO] [-a ARCH] [-b]\n");
    return;
}


static void print_help()
{
    PRINT_OUT("\n");
    PRINT_OUT("resnet50 ImageNet classification model\n");
    PRINT_OUT("\n");
    PRINT_OUT("optional arguments:\n");
    PRINT_OUT("  -h, --help            show this help message and exit\n");
    PRINT_OUT("  -i IMAGE, --input IMAGE\n");
    PRINT_OUT("                        The input image path.\n");
    PRINT_OUT("  -v VIDEO, --video VIDEO\n");
    PRINT_OUT("                        The input video path. If the VIDEO argument is set to\n");
    PRINT_OUT("                        0, the webcam input will be used.\n");
    PRINT_OUT("  -a ARCH, --arch ARCH  model architecture: resnet50.opt | resnet50 |\n");
    PRINT_OUT("                        resnet50_pytorch (default: resnet50.opt)\n");
    PRINT_OUT("  -b, --benchmark       Running the inference on the same input 5 times to\n");
    PRINT_OUT("                        measure execution performance. (Cannot be used in\n");
    PRINT_OUT("                        video mode)\n");
    return;
}


static void print_error(std::string arg)
{
    PRINT_ERR("resnet50: error: unrecognized arguments: %s\n", arg.c_str());
    return;
}


static void print_arch_error(std::string arg)
{
    PRINT_ERR("resnet50: error: argument -a/--arch: invalid choice: \'%s\' (choose from", arg.c_str());
    for (size_t i = 0; i < MODEL_NAMES.size(); i++) {
        PRINT_ERR(" \'%s\'", MODEL_NAMES[i]);
        if (i < MODEL_NAMES.size() - 1) {
            PRINT_ERR(",");
        }
    }
    PRINT_ERR(")\n");
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
            else if (arg == "-a" || arg == "--arch") {
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
                size_t j;
                for (j = 0; j < MODEL_NAMES.size(); j++) {
                    if (arg == MODEL_NAMES[j]) {
                        weight = arg + ".onnx";
                        model  = arg + ".onnx.prototxt";
                        break;
                    }
                }
                if (j >= MODEL_NAMES.size()) {
                    print_usage();
                    print_arch_error(arg);
                    return -1;
                }
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
// Utils
// ======================

static void preprocess_image(const cv::Mat& simg, cv::Mat& dimg)
{
    if (simg.channels() == 3) {
        cv::cvtColor(simg, dimg, cv::COLOR_BGR2BGRA);
    }
    else if (simg.channels() == 1) {
        cv::cvtColor(simg, dimg, cv::COLOR_GRAY2BGRA);
    }
    else {
//        dimg = simg.clone();
        simg.copyTo(dimg);
    }

    return;
}


// ======================
// Main functions
// ======================

static int recognize_from_image(AILIAClassifier *classifier)
{
    // prepare input data
    cv::Mat simg = cv::imread(image_path.c_str(), cv::IMREAD_UNCHANGED);
    if (simg.empty()) {
        PRINT_ERR("\'%s\' image not found\n", image_path.c_str());
        return -1;
    }
    cv::Mat img;
    preprocess_image(simg, img);
//    PRINT_OUT("input image shape: (%d, %d, %d)\n",
//              img.cols, img.rows, img.channels());

    // inference
    PRINT_OUT("Start inference...\n");
    int status;
    if (benchmark) {
        PRINT_OUT("BENCHMARK mode\n");
        for (int i = 0; i < BENCHMARK_ITERS; i++) {
            clock_t start = clock();
            status = ailiaClassifierCompute(classifier, img.data,
                                            img.cols*4, img.cols, img.rows,
                                            AILIA_IMAGE_FORMAT_BGRA, MAX_CLASS_COUNT);
            clock_t end = clock();
            if (status != AILIA_STATUS_SUCCESS) {
                PRINT_ERR("ailiaClassifierCompute failed %d\n", status);
                return -1;
            }
            PRINT_OUT("\tailia processing time %ld ms\n", ((end-start)*1000)/CLOCKS_PER_SEC);
        }
    }
    else {
        status = ailiaClassifierCompute(classifier, img.data,
                                        img.cols*4, img.cols, img.rows,
                                        AILIA_IMAGE_FORMAT_BGRA, MAX_CLASS_COUNT);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaClassifierCompute failed %d\n", status);
            return -1;
        }
    }

    unsigned int count = 0;
    status = ailiaClassifierGetClassCount(classifier, &count);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaClassifierGetClassCount failed %d\n", status);
        return -1;
    }
    count = std::min<unsigned int>(count, MAX_CLASS_COUNT);
    PRINT_OUT("class_count: %d\n", count);
    for (unsigned int idx = 0; idx < count; idx++) {
        AILIAClassifierClass info;
        status = ailiaClassifierGetClass(classifier, &info, idx, AILIA_CLASSIFIER_CLASS_VERSION);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaClassifierGetClass failed %d\n", status);
            return -1;
        }
        PRINT_OUT("+ idx=%d\n", idx);
        PRINT_OUT("  category=%d [ %s ]\n", info.category, IMAGENET_CATEGORY[info.category]);
        PRINT_OUT("  prob=%.18lf\n", info.prob);
    }

    PRINT_OUT("Program finished successfully.\n");

    return AILIA_STATUS_SUCCESS;
}


static int recognize_from_video(AILIAClassifier *classifier)
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
        cv::Mat frame0;
        capture >> frame0;
        if ((char)cv::waitKey(1) == 'q') {
            break;
        }
        if (frame0.empty()) {
            continue;
        }
        cv::Mat in_frame, frame1, frame;
        adjust_frame_size(frame0, in_frame, frame1, IMAGE_WIDTH, IMAGE_HEIGHT);
        preprocess_image(frame1, frame);

        int status = ailiaClassifierCompute(classifier, frame.data,
                                            frame.cols*4, frame.cols, frame.rows,
                                            AILIA_IMAGE_FORMAT_BGRA, MAX_CLASS_COUNT);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaClassifierCompute failed %d\n", status);
            return -1;
        }

        unsigned int count = 0;
        status = ailiaClassifierGetClassCount(classifier, &count);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaClassifierGetClassCount failed %d\n", status);
            return -1;
        }
        count = std::min<unsigned int>(count, MAX_CLASS_COUNT);
        PRINT_OUT("==============================================================\n");
        PRINT_OUT("class_count: %d\n", count);
        for (unsigned int idx = 0; idx < count; idx++) {
            AILIAClassifierClass info;
            status = ailiaClassifierGetClass(classifier, &info, idx, AILIA_CLASSIFIER_CLASS_VERSION);
            if (status != AILIA_STATUS_SUCCESS) {
                PRINT_ERR("ailiaClassifierGetClass failed %d\n", status);
                return -1;
            }
            PRINT_OUT("+ idx=%d\n", idx);
            PRINT_OUT("  category=%d [ %s ]\n", info.category, IMAGENET_CATEGORY[info.category]);
            PRINT_OUT("  prob=%.18lf\n", info.prob);
        }

        cv::imshow("frame", in_frame);
        sleep(SLEEP_TIME);
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

    AILIAClassifier *classifier;
    status = ailiaCreateClassifier(&classifier, ailia,
                                   AILIA_NETWORK_IMAGE_FORMAT_RGB,
                                   AILIA_NETWORK_IMAGE_CHANNEL_FIRST,
                                   AILIA_NETWORK_IMAGE_RANGE_SIGNED_INT8);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaCreateClassifier failed %d\n", status);
        ailiaDestroy(ailia);
        return -1;
    }

    if (video_mode) {
        status = recognize_from_video(classifier);
    }
    else {
        status = recognize_from_image(classifier);
    }

    ailiaDestroyClassifier(classifier);
    ailiaDestroy(ailia);

    return status;
}
