/*******************************************************************
*
*    DESCRIPTION:
*      AILIA u2net sample
*    AUTHOR:
*
*    DATE:2020/09/01
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
#include "u2net_utils.h"
#include "utils.h"
#include "webcamera_utils.h"


// ======================
// Parameters
// ======================

#define WEIGHT_PATH "u2net.onnx"
#define MODEL_PATH  "u2net.onnx.prototxt"

#define IMAGE_PATH      "input.png"
#define SAVE_IMAGE_PATH "output.png"

#define IMAGE_SIZE 320

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

static const std::vector<const char*> MODEL_LISTS = {"small", "large"};
static const std::vector<const char*> OPSET_LISTS = {"10", "11"};

static bool benchmark  = false;
static bool video_mode = false;


// ======================
// Arguemnt Parser
// ======================

static void print_usage()
{
    PRINT_OUT("usage: u2net [-h] [-i IMAGE] [-v VIDEO] [-a ARCH] [-s SAVE_IMAGE_PATH] [-b]\n");
    PRINT_OUT("             [-o OPSET]\n");
    return;
}


static void print_help()
{
    PRINT_OUT("\n");
    PRINT_OUT("U square net\n");
    PRINT_OUT("\n");
    PRINT_OUT("optional arguments:\n");
    PRINT_OUT("  -h, --help            show this help message and exit\n");
    PRINT_OUT("  -i IMAGE, --input IMAGE\n");
    PRINT_OUT("                        The input image path. (default: input.png)\n");
    PRINT_OUT("  -v VIDEO, --video VIDEO\n");
    PRINT_OUT("                        The input video path. If the VIDEO argument is set to\n");
    PRINT_OUT("                        0, the webcam input will be used. (default: None)\n");
    PRINT_OUT("  -a ARCH, --arch ARCH  model lists: small | large (default: large)\n");
    PRINT_OUT("  -s SAVE_IMAGE_PATH, --savepath SAVE_IMAGE_PATH\n");
    PRINT_OUT("                        Save path for the output image. (default: output.png)\n");
    PRINT_OUT("  -b, --benchmark       Running the inference on the same input 5 times to\n");
    PRINT_OUT("                        measure execution performance. (Cannot be used in\n");
    PRINT_OUT("                        video mode) (default: False)\n");
    PRINT_OUT("  -o OPSET, --opset OPSET\n");
    PRINT_OUT("                        opset lists: 10 | 11 (default: 10)\n");
    return;
}


static void print_error(std::string arg)
{
    PRINT_ERR("u2net: error: unrecognized arguments: %s\n", arg.c_str());
    return;
}


static void print_arch_error(std::string arg)
{
    PRINT_ERR("u2net: error: argument -a/--arch: invalid choice: \'%s\' (choose from", arg.c_str());
    for (size_t i = 0; i < MODEL_LISTS.size(); i++) {
        PRINT_ERR(" \'%s\'", MODEL_LISTS[i]);
        if (i < MODEL_LISTS.size() - 1) {
            PRINT_ERR(",");
        }
    }
    PRINT_ERR(")\n");
    return;
}


static void print_opset_error(std::string arg)
{
    PRINT_ERR("u2net: error: argument -o/--opset: invalid choice: \'%s\' (choose from", arg.c_str());
    for (size_t i = 0; i < OPSET_LISTS.size(); i++) {
        PRINT_ERR(" \'%s\'", OPSET_LISTS[i]);
        if (i < OPSET_LISTS.size() - 1) {
            PRINT_ERR(",");
        }
    }
    PRINT_ERR(")\n");
    return;
}


static int argument_parser(int argc, char **argv)
{
    int status = 0;
    std::string arch  = "large";
    std::string opset = "10";

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
            else if (arg == "-s" || arg == "--savepath") {
                status = 4;
            }
            else if (arg == "-o" || arg == "--opset") {
                status = 5;
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
            size_t j;
            switch (status) {
            case 1:
                image_path = arg;
                break;
            case 2:
                video_path = arg;
                break;
            case 3:
                for (j = 0; j < MODEL_LISTS.size(); j++) {
                    if (arg == MODEL_LISTS[j]) {
                        arch = arg;
                        break;
                    }
                }
                if (j >= MODEL_LISTS.size()) {
                    print_usage();
                    print_arch_error(arg);
                    return -1;
                }
                break;
            case 4:
                save_image_path = arg;
                break;
            case 5:
                for (j = 0; j < OPSET_LISTS.size(); j++) {
                    if (arg == OPSET_LISTS[j]) {
                        opset = arg;
                        break;
                    }
                }
                if (j >= OPSET_LISTS.size()) {
                    print_usage();
                    print_opset_error(arg);
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

    if (arch == "large") {
        arch = "";
    }
    else {
        arch = "p";
    }
    if (opset == "10") {
        opset = "";
    }
    else {
        opset = "_opset11";
    }

    weight = "u2net" + arch + opset + ".onnx";
    model  = weight + ".prototxt";

    return AILIA_STATUS_SUCCESS;
}


// ======================
// Main functions
// ======================

static int recognize_from_image(AILIANetwork *net)
{
    // prepare input data
    cv::Mat  input;
    cv::Size src_size;
    int status = load_image(input, src_size, image_path.c_str(), cv::Size(IMAGE_SIZE, IMAGE_SIZE));
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
    cv::Mat preds_ailia(output_shape.y, output_shape.x, CV_32FC1);

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

    std::vector<int> new_shape = {(int)output_shape.y, (int)output_shape.x};
    preds_ailia.reshape(1, (int)new_shape.size(), &new_shape[0]);
    status = save_result(preds_ailia, save_image_path.c_str(), src_size);
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }

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
    cv::Mat preds_ailia(output_shape.y, output_shape.x, CV_32FC1);

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

    // create video writer if savepath is specified as video format
    int f_w = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int f_h = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::VideoWriter writer;
    if (save_image_path != SAVE_IMAGE_PATH) {
        status = get_writer(writer, save_image_path.c_str(), cv::Size(f_w, f_h), false);
        if (status != AILIA_STATUS_SUCCESS) {
            return -1;
        }
    }

    while (1) {
        cv::Mat frame;
        capture >> frame;
        if ((char)cv::waitKey(1) == 'q' || frame.empty()) {
            break;
        }
        cv::Mat input;
        transform(frame, input, cv::Size(IMAGE_SIZE, IMAGE_SIZE));

        // inference
        status = ailiaPredict(net, preds_ailia.data, preds_size, input.data, input_size);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaPredict failed %d\n", status);
            return -1;
        }

        // post processing
        cv::Mat pred;
        cv::resize(preds_ailia, pred, cv::Size(f_w, f_h), 0, 0);
        cv::imshow("frame", pred);

        // save results
        if (writer.isOpened()) {
            writer.write(pred);
        }
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
