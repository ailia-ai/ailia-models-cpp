/*******************************************************************
*
*    DESCRIPTION:
*      AILIA face alignment sample
*    AUTHOR:
*
*    DATE:2020/09/23
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
#include "utils.h"
#include "image_utils.h"
#include "webcamera_utils.h"


// ======================
// Parameters
// ======================

#define WEIGHT_PATH "face_alignment.onnx"
#define MODEL_PATH  "face_alignment.onnx.prototxt"

#define IMAGE_PATH      "aflw-test.jpg"
#define SAVE_IMAGE_PATH "output.png"

#define IMAGE_WIDTH  256
#define IMAGE_HEIGHT 256

#define THRESHOLD 0.1

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

static bool benchmark  = false;
static bool video_mode = false;


// ======================
// Arguemnt Parser
// ======================

static void print_usage()
{
    PRINT_OUT("usage: face_alignment [-h] [-i IMAGEFILE_PATH] [-v VIDEO]\n");
    PRINT_OUT("                      [-s SAVE_IMAGE_PATH] [-b]\n");
    return;
}


static void print_help()
{
    PRINT_OUT("\n");
    PRINT_OUT("Face alignment model\n");
    PRINT_OUT("\n");
    PRINT_OUT("optional arguments:\n");
    PRINT_OUT("  -h, --help            show this help message and exit\n");
    PRINT_OUT("  -i IMAGEFILE_PATH, --input IMAGEFILE_PATH\n");
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
    PRINT_ERR("face_alignment: error: unrecognized arguments: %s\n", arg.c_str());
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
// Utils
// ======================

//static int plot_images()

static void visualize_plots(cv::Mat& image, const cv::Mat& preds_ailia)
{
    float* preds_data = (float*)preds_ailia.data;
    int rows = preds_ailia.size[1];
    int cols = preds_ailia.size[2];
    for (int i = 0; i < preds_ailia.size[0]; i++) {
        double prob;
        cv::Point point;
        cv::Mat probMap = cv::Mat(rows, cols, CV_32FC1, &preds_data[rows*cols*i]);
        cv::minMaxLoc(probMap, NULL, &prob, NULL, &point);
        if (prob > THRESHOLD) {
            float x = ((float)image.cols * point.x) / (float)cols;
            float y = ((float)image.rows * point.y) / (float)rows;
            int circle_size = 4;
            cv::Scalar color = cv::Scalar(0, 255, 255);
            int thickness = -1;
            int linetype = 4;
            cv::circle(image, cv::Point((int)x, (int)y), circle_size, color, thickness, linetype);
        }
    }

    return;
}


// ======================
// Main functions
// ======================

static int recognize_from_image(AILIANetwork *net)
{
    // prepare input data;
    cv::Mat input_img = cv::imread(image_path.c_str(), -1);
    if (input_img.empty()) {
        PRINT_ERR("\'%s\' image not found\n", image_path.c_str());
        return -1;
    }
    cv::Mat input;
    int status = load_image(input, image_path.c_str(), cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), true, "255", true);
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }

    AILIAShape input_shape;
    status = ailiaGetInputShape(net, &input_shape, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetInputShape failed %d\n", status);
        return -1;
    }
//    PRINT_OUT("input shape %d %d %d %d %d\n",input_shape.x, input_shape.y, input_shape.z, input_shape.w, input_shape.dim);
    int input_size = input_shape.x*input_shape.y*input_shape.z*input_shape.w*sizeof(float);

    AILIAShape output_shape;
    status = ailiaGetOutputShape(net, &output_shape, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetOutputShape failed %d\n", status);
        return -1;
    }
//    PRINT_OUT("output shape %d %d %d %d %d\n", output_shape.x, output_shape.y, output_shape.z, output_shape.w, output_shape.dim);
    int preds_size = output_shape.x*output_shape.y*output_shape.z*output_shape.w*sizeof(float);
    std::vector<int> shape = {(int)output_shape.z, (int)output_shape.y, (int)output_shape.x};
    cv::Mat preds_ailia = cv::Mat((int)shape.size(), &shape[0], CV_32FC1);

    // inference
    PRINT_OUT("Start inference...\n");
    if (benchmark) {
        PRINT_OUT("BENCHMARK mode\n");
        for (int i = 0; i < BENCHMARK_ITERS; i++) {
            clock_t start = clock();
            status = ailiaPredict(net, preds_ailia.data, preds_size, input.data, input_size);
            clock_t end = clock();
            if (status != AILIA_STATUS_SUCCESS) {
                PRINT_ERR("ailiaPredict failed %d\n", status);
                return -1;
            }
            PRINT_OUT("\tailia processing time %ld ms\n", ((end-start)*1000)/CLOCKS_PER_SEC);
        }
    }
    else {
        status = ailiaPredict(net, preds_ailia.data, preds_size, input.data, input_size);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaPredict failed %d\n", status);
            return -1;
        }
    }

    visualize_plots(input_img, preds_ailia);
    cv::imwrite(save_image_path.c_str(), input_img);

    PRINT_OUT("Program finished successfully.\n");

    return AILIA_STATUS_SUCCESS;
}


static int recognize_from_video(AILIANetwork *net)
{
    AILIAShape input_shape;
    int status = ailiaGetInputShape(net, &input_shape, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetInputShape failed %d\n", status);
        return -1;
    }
//    PRINT_OUT("input shape %d %d %d %d %d\n",input_shape.x, input_shape.y, input_shape.z, input_shape.w, input_shape.dim);
    int input_size = input_shape.x*input_shape.y*input_shape.z*input_shape.w*sizeof(float);

    AILIAShape output_shape;
    status = ailiaGetOutputShape(net, &output_shape, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetOutputShape failed %d\n", status);
        return -1;
    }
//    PRINT_OUT("output shape %d %d %d %d %d\n", output_shape.x, output_shape.y, output_shape.z, output_shape.w, output_shape.dim);
    int preds_size = output_shape.x*output_shape.y*output_shape.z*output_shape.w*sizeof(float);
    std::vector<int> shape = {(int)output_shape.z, (int)output_shape.y, (int)output_shape.x};
    cv::Mat preds_ailia = cv::Mat((int)shape.size(), &shape[0], CV_32FC1);

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
        cv::Mat input_image, input;
        preprocess_frame(frame, input_image, input, IMAGE_WIDTH, IMAGE_HEIGHT, true, "255");

        // inference
        status = ailiaPredict(net, preds_ailia.data, preds_size, input.data, input_size);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaPredict failed %d\n", status);
            return -1;
        }

        // post processing
        visualize_plots(input_image, preds_ailia);
        cv::imshow("frame", input_image);
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
    AILIANetwork *net;
    int env_id = AILIA_ENVIRONMENT_ID_AUTO;
    status = ailiaCreate(&net, env_id, AILIA_MULTITHREAD_AUTO);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaCreate failed %d\n", status);
        return -1;
    }

    AILIAEnvironment *env_ptr = nullptr;
    status = ailiaGetSelectedEnvironment(net, &env_ptr, AILIA_ENVIRONMENT_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetSelectedEnvironment failed %d\n", status);
        ailiaDestroy(net);
        return -1;
    }

//    PRINT_OUT("env_id: %d\n", env_ptr->id);
    PRINT_OUT("env_name: %s\n", env_ptr->name);

    status = ailiaOpenStreamFile(net, model.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaOpenStreamFile failed %d\n", status);
        PRINT_ERR("ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(net));
        ailiaDestroy(net);
        return -1;
    }

    status = ailiaOpenWeightFile(net, weight.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
        ailiaDestroy(net);
        return -1;
    }

    if (video_mode) {
        status = recognize_from_video(net);
    }
    else {
        status = recognize_from_image(net);
    }

    ailiaDestroy(net);

    return status;
}
