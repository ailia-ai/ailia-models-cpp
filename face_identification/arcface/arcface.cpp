/*******************************************************************
*
*    DESCRIPTION:
*      AILIA arcface sample
*    AUTHOR:
*
*    DATE:2020/09/07
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
#include "mat_utils.h"
#include "image_utils.h"
#include "webcamera_utils.h"
#include "blazeface_utils.h"


// ======================
// Parameters
// ======================

#define WEIGHT_PATH "arcface.onnx"
#define MODEL_PATH  "arcface.onnx.prototxt"

#define IMAGE_PATH_1 "correct_pair_1.jpg"
#define IMAGE_PATH_2 "correct_pair_2.jpg"

#define IMAGE_WIDTH  128
#define IMAGE_HEIGHT 128

// the threshold was calculated by the `test_performance` function in `test.py`
// of the original repository
#define THRESHOLD 0.25572845f
//#define THRESHOLD 0.45f // for mixed model

#define FACE_WEIGHT_PATH "yolov3-face.opt.onnx"
#define FACE_MODEL_PATH  "yolov3-face.opt.onnx.prototxt"

#define YOLOV3_MODEL_INPUT_WIDTH     416
#define YOLOV3_MODEL_INPUT_HEIGHT    416
#define BLAZEFACE_INPUT_IMAGE_WIDTH  128
#define BLAZEFACE_INPUT_IMAGE_HEIGHT 128

#define YOLOV3_FACE_THRESHOLD 0.2f
#define YOLOV3_FACE_IOU       0.45f

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

static std::string image_path_1(IMAGE_PATH_1);
static std::string image_path_2(IMAGE_PATH_2);
static std::string video_path("0");

static std::string face_weight(FACE_WEIGHT_PATH);
static std::string face_model(FACE_MODEL_PATH);

static const std::vector<const char*> MODEL_LISTS = {"arcface", "arcface_mixed_90_82", "arcface_mixed_90_99", "arcface_mixed_eq_90_89"};
static const std::vector<const char*> FACE_MODEL_LISTS = {"yolov3", "blazeface"};

static std::string arch(MODEL_LISTS[0]);
static std::string face(FACE_MODEL_LISTS[0]);
static bool benchmark  = false;
static bool video_mode = false;

static float threshold = THRESHOLD;


// ======================
// Arguemnt Parser
// ======================

static void print_usage()
{
    PRINT_OUT("usage: arcface [-h] [-i IMAGE IMAGE] [-v VIDEO] [-b] [-a ARCH]\n");
    PRINT_OUT("               [-f FACE_ARCH] [-t THRESHOLD]\n");
    return;
}


static void print_help()
{
    PRINT_OUT("\n");
    PRINT_OUT("Determine if the person is the same from two facial images.\n");
    PRINT_OUT("\n");
    PRINT_OUT("optional arguments:\n");
    PRINT_OUT("  -h, --help            show this help message and exit\n");
    PRINT_OUT("  -i IMAGE IMAGE, --inputs IMAGE IMAGE\n");
    PRINT_OUT("                        Two image paths for calculating the face match.\n");
    PRINT_OUT("  -v VIDEO, --video VIDEO\n");
    PRINT_OUT("                        The input video path. If the VIDEO argument is set to\n");
    PRINT_OUT("                        0, the webcam input will be used.\n");
    PRINT_OUT("  -b, --benchmark       Running the inference on the same input 5 times to\n");
    PRINT_OUT("                        measure execution performance. (Cannot be used in\n");
    PRINT_OUT("                        video mode)\n");
    PRINT_OUT("  -a ARCH, --arch ARCH  model lists: arcface | arcface_mixed_90_82 |\n");
    PRINT_OUT("                        arcface_mixed_90_99 | arcface_mixed_eq_90_89\n");
    PRINT_OUT("  -f FACE_ARCH, --face FACE_ARCH\n");
    PRINT_OUT("                        face detection model lists: yolov3 | blazeface\n");
    PRINT_OUT("  -t THRESHOLD, --threshold THRESHOLD\n");
    PRINT_OUT("                        Similality threshold for identification\n");
    return;
}


static void print_error(std::string arg)
{
    PRINT_ERR("arcface: error: unrecognized arguments: %s\n", arg.c_str());
    return;
}


static void print_inputs_error()
{
    PRINT_ERR("arcface: error: argument -i/--inputs: expected 2 arguments\n");
    return;
}


static void print_arch_error(std::string arg)
{
    PRINT_ERR("arcface: error: argument -a/--arch: invalid choice: \'%s\' (choose from", arg.c_str());
    for (size_t i = 0; i < MODEL_LISTS.size(); i++) {
        PRINT_ERR(" \'%s\'", MODEL_LISTS[i]);
        if (i < MODEL_LISTS.size() - 1) {
            PRINT_ERR(",");
        }
    }
    PRINT_ERR(")\n");
    return;
}


static void print_face_error(std::string arg)
{
    PRINT_ERR("arcface: error: argument -f/--face: invalid choice: \'%s\' (choose from", arg.c_str());
    for (size_t i = 0; i < FACE_MODEL_LISTS.size(); i++) {
        PRINT_ERR(" \'%s\'", FACE_MODEL_LISTS[i]);
        if (i < FACE_MODEL_LISTS.size() - 1) {
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
            if (arg == "-i" || arg == "--inputs") {
                status = 1;
            }
            else if (arg == "-v" || arg == "--video") {
                video_mode = true;
                status = 3;
            }
            else if (arg == "-a" || arg == "--arch") {
                status = 4;
            }
            else if (arg == "-f" || arg == "--face") {
                status = 5;
            }
            else if (arg == "-t" || arg == "--threshold") {
                status = 6;
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
                image_path_1 = arg;
                break;
            case 2:
                image_path_2 = arg;
                break;
            case 3:
                video_path = arg;
                break;
            case 4:
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
            case 5:
                for (j = 0; j < FACE_MODEL_LISTS.size(); j++) {
                    if (arg == FACE_MODEL_LISTS[j]) {
                        face = arg;
                        break;
                    }
                }
                if (j >= FACE_MODEL_LISTS.size()) {
                    print_usage();
                    print_face_error(arg);
                    return -1;
                }
                break;
            default:
                print_usage();
                print_error(arg);
                return -1;
            }
            if (status == 1) {
                status = 2;
            }
            else {
                status = 0;
            }
        }
        else if (status == 1) {
            print_usage();
            print_inputs_error();
            return -1;
        }
        else {
            print_usage();
            print_error(arg);
            return -1;
        }
    }

    weight = arch + ".onnx";
    model  = weight + ".prototxt";
    if (face == "yolov3") {
        face_weight = face + "-face.opt.onnx";
    }
    else {
        face_weight = face + ".onnx";
    }
    face_model  = face_weight + ".prototxt";

    return AILIA_STATUS_SUCCESS;
}


// ======================
// Utils
// ======================

static void preprocess_image(const cv::Mat& simg, cv::Mat& dimg, bool input_is_bgr=false)
{
    cv::Mat mimg0, mimg1, mimg2, mimg3, mimg4;
    if (input_is_bgr) {
        cv::cvtColor(simg, mimg0, cv::COLOR_BGR2GRAY);
    }
    else {
        simg.copyTo(mimg0);
    }
    if (weight.find("_eq") != std::string::npos) {
        cv::equalizeHist(mimg0, mimg1);
    }
    else {
        mimg0.copyTo(mimg1);
    }

    cv::flip(mimg1, mimg2, 1);
    std::vector<cv::Mat> mergev;
    mergev.push_back(mimg1);
    mergev.push_back(mimg2);
    cv::merge(mergev, mimg3);
    transpose(mimg3, mimg4, {2, 0, 1});

    std::vector<int> new_shape = {(int)mimg4.size[0], (int)mimg4.size[1], (int)mimg4.size[2]};
    dimg = cv::Mat_<float>((int)new_shape.size(), &new_shape[0]);
    unsigned char* sdata = (unsigned char*)mimg4.data;
    float*         ddata = (float*)dimg.data;
    for (int i = 0; i < mimg4.size[0]*mimg4.size[1]*mimg4.size[2]; i++) {
        ddata[i] = (float)sdata[i] / 127.5f - 1.0f;
    }

    return;
}


static int prepare_input_data(cv::Mat& img, const char* path)
{
    cv::Mat oimg;

    int status = load_image(oimg, path, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), false, "None");
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }

    preprocess_image(oimg, img);

    return 0;
}


static float cosin_metric(const cv::Mat& x1, const cv::Mat& x2)
{
    return (float)(x1.dot(x2) / (cv::norm(x1)*cv::norm(x2)));
}


static int face_identification(std::vector<cv::Mat>& fe_list, AILIANetwork *net, const cv::Mat& resized_frame,
                               int& id_sim, float& score_sim)
{
    AILIAShape input_shape;
    int status = ailiaGetInputShape(net, &input_shape, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetInputShape failed %d\n", status);
        return -1;
    }
    int input_size = input_shape.x*input_shape.y*input_shape.z*input_shape.w*sizeof(float);
    int batch_size = input_shape.w;

    // prepare target face and input face
    cv::Mat input_frame, input;
    preprocess_image(resized_frame, input_frame, true);
    if (batch_size == 4) {
        concatenate(input_frame, input_frame, input, 0);
    }
    else {
        input_frame.copyTo(input);
    }

    AILIAShape output_shape;
    status = ailiaGetOutputShape(net, &output_shape, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetOutputShape failed %d\n", status);
        return -1;
    }
    int preds_size = output_shape.x*output_shape.y*output_shape.z*output_shape.w*sizeof(float);

    cv::Mat preds_ailia = cv::Mat(output_shape.y, output_shape.x, CV_32FC1);

    // inference
    status = ailiaPredict(net, preds_ailia.data, preds_size, input.data, input_size);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaPredict failed %d\n", status);
        return -1;
    }

    // postprocessing
    cv::Mat fe_1, fe_2;
    fe_1 = preds_ailia.rowRange(cv::Range(0, 2));
    if (batch_size == 4) {
        fe_2 = preds_ailia.rowRange(cv::Range(2, 4));
    }
    else {
        fe_1.copyTo(fe_2);
    }

    // identification
    id_sim = 0;
    score_sim = 0.0f;
    for (size_t i = 0; i < fe_list.size(); i++) {
        cv::Mat fe;
        fe_list[i].copyTo(fe);;
        float sim = cosin_metric(fe, fe_2);
        if (sim > score_sim) {
            id_sim = i;
            score_sim = sim;
        }
    }
    if (score_sim < threshold) {
        id_sim = (int)fe_list.size();
        fe_list.push_back(fe_2);
        score_sim = 0.0f;
    }

    return 0;
}


// ======================
// Main functions
// ======================

static int compare_images(AILIANetwork *net)
{
    // prepare input data
    cv::Mat imgs_1, imgs_2, input;
    prepare_input_data(imgs_1, image_path_1.c_str());
    prepare_input_data(imgs_2, image_path_2.c_str());
    concatenate(imgs_1, imgs_2, input, 0);

    AILIAShape input_shape;
    int status = ailiaGetInputShape(net, &input_shape, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetInputShape failed %d\n", status);
        return -1;
    }
//    PRINT_OUT("input shape %d %d %d %d %d\n",input_shape.x, input_shape.y, input_shape.z, input_shape.w, input_shape.dim);
    int input_size = input_shape.x*input_shape.y*input_shape.z*input_shape.w*sizeof(float);
    int batch_size = input_shape.w;

    AILIAShape output_shape;
    status = ailiaGetOutputShape(net, &output_shape, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetOutputShape failed %d\n", status);
        return -1;
    }
//    PRINT_OUT("output shape %d %d %d %d %d\n", output_shape.x, output_shape.y, output_shape.z, output_shape.w, output_shape.dim);
    int preds_size = output_shape.x*output_shape.y*output_shape.z*output_shape.w*sizeof(float);

    cv::Mat preds_ailia;
    if (batch_size == 2) {
        preds_ailia = cv::Mat(output_shape.y*2, output_shape.x, CV_32FC1);
    }
    else {
        preds_ailia = cv::Mat(output_shape.y, output_shape.x, CV_32FC1);
    }

    // inference
    PRINT_OUT("Start inference...\n");
    if (benchmark) {
        PRINT_OUT("BENCHMARK mode\n");
        if (batch_size == 2) {
            for (int i = 0; i < BENCHMARK_ITERS; i++) {
                clock_t start0 = clock();
                status = ailiaPredict(net, preds_ailia.data, preds_size, input.data, input_size);
                clock_t end0 = clock();
                if (status != AILIA_STATUS_SUCCESS) {
                    PRINT_ERR("ailiaPredict failed %d\n", status);
                    return -1;
                }
                clock_t start1 = clock();
                status = ailiaPredict(net, (char*)preds_ailia.data+preds_size, preds_size, (char*)input.data+input_size, input_size);
                clock_t end1 = clock();
                if (status != AILIA_STATUS_SUCCESS) {
                    PRINT_ERR("ailiaPredict failed %d\n", status);
                    return -1;
                }
                PRINT_OUT("\tailia processing time %ld ms\n", (((end0-start0)+(end1-start1))*1000)/CLOCKS_PER_SEC);
            }
         }
         else {
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
    }
    else {
        if (batch_size == 2) {
            status = ailiaPredict(net, preds_ailia.data, preds_size, input.data, input_size);
            if (status != AILIA_STATUS_SUCCESS) {
                PRINT_ERR("ailiaPredict failed %d\n", status);
                return -1;
            }
            status = ailiaPredict(net, (char*)preds_ailia.data+preds_size, preds_size, (char*)input.data+input_size, input_size);
            if (status != AILIA_STATUS_SUCCESS) {
                PRINT_ERR("ailiaPredict failed %d\n", status);
                return -1;
            }
         }
         else {
            status = ailiaPredict(net, preds_ailia.data, preds_size, input.data, input_size);
            if (status != AILIA_STATUS_SUCCESS) {
                PRINT_ERR("ailiaPredict failed %d\n", status);
                return -1;
            }
         }
    }

    // postprocessing
    cv::Mat fe_1, fe_2;
    fe_1 = preds_ailia.rowRange(cv::Range(0, 2));
    if (batch_size == 4) {
        fe_2 = preds_ailia.rowRange(cv::Range(2, 4));
    }
    else {
        fe_1.copyTo(fe_2);
    }

    float sim = cosin_metric(fe_1, fe_2);

    PRINT_OUT("Similarity of (%s, %s) : %.3f\n", image_path_1.c_str(), image_path_2.c_str(), sim);

    if (sim < threshold) {
        PRINT_OUT("They are not the same face!\n");
    }
    else {
        PRINT_OUT("They are the same face!\n");
    }

    PRINT_OUT("Program finished successfully.\n");

    return AILIA_STATUS_SUCCESS;
}


static int compare_video(AILIANetwork *net)
{
    AILIANetwork  *detector_net;
    AILIADetector *detector;
    std::vector<cv::Mat> fe_list;

    // net initialize
    int env_id = AILIA_ENVIRONMENT_ID_AUTO;
    int status = ailiaCreate(&detector_net, env_id, AILIA_MULTITHREAD_AUTO);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaCreate failed %d\n", status);
        return -1;
    }

    status = ailiaOpenStreamFile(detector_net, face_model.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaOpenStreamFile failed %d\n", status);
        PRINT_ERR("ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(detector_net));
        ailiaDestroy(detector_net);
        return -1;
    }

    status = ailiaOpenWeightFile(detector_net, face_weight.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaOpenWeightFile failed %d %s\n", status,face_weight.c_str());
        ailiaDestroy(detector_net);
        return -1;
    }

    // detector initialize
    unsigned int in_inds = 0, out0_inds = 0, out1_inds = 0; // for BlazeFace
    int input_size = 0, out0_size = 0, out1_size = 0;       // for BlazeFace
    cv::Mat preds_ailia0, preds_ailia1;                     // for BlazeFace
    if (face == "yolov3") {
        status = ailiaCreateDetector(&detector, detector_net,
                                     AILIA_NETWORK_IMAGE_FORMAT_RGB,
                                     AILIA_NETWORK_IMAGE_CHANNEL_FIRST,
                                     AILIA_NETWORK_IMAGE_RANGE_UNSIGNED_FP32,
                                     AILIA_DETECTOR_ALGORITHM_YOLOV3,
                                     1, AILIA_DETECTOR_FLAG_NORMAL);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaCreateDetector failed %d\n", status);
            ailiaDestroy(detector_net);
            return -1;
        }

        status = ailiaDetectorSetInputShape(detector, YOLOV3_MODEL_INPUT_WIDTH, YOLOV3_MODEL_INPUT_HEIGHT);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaDetectorSetInputShape(w=%u, h=%u) failed %d\n",
                      YOLOV3_MODEL_INPUT_WIDTH, YOLOV3_MODEL_INPUT_HEIGHT, status);
            ailiaDestroyDetector(detector);
            ailiaDestroy(detector_net);
            return -1;
        }
    }
    else {
        status = ailiaFindBlobIndexByName(detector_net, &in_inds, "x.1");
        if (status != AILIA_STATUS_SUCCESS) {
            ailiaDestroy(detector_net);
            return -1;
        }

        status = ailiaFindBlobIndexByName(detector_net, &out0_inds, "199");
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaFindBlobIndexByName failed %d\n", status);
            ailiaDestroy(detector_net);
            return -1;
        }

        status = ailiaFindBlobIndexByName(detector_net, &out1_inds, "180");
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaFindBlobIndexByName failed %d\n", status);
            ailiaDestroy(detector_net);
            return -1;
        }

        AILIAShape in_shape;
        status = ailiaGetBlobShape(detector_net, &in_shape, in_inds, AILIA_SHAPE_VERSION);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaGetBlobShape failed %d\n", status);
            ailiaDestroy(detector_net);
            return -1;
        }
        input_size = in_shape.x*in_shape.y*in_shape.z*in_shape.w*sizeof(float);

        AILIAShape out0_shape;
        status = ailiaGetBlobShape(detector_net, &out0_shape, out0_inds, AILIA_SHAPE_VERSION);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaGetBlobShape failed %d\n", status);
            ailiaDestroy(detector_net);
            return -1;
        }
        out0_size = out0_shape.x*out0_shape.y*out0_shape.z*out0_shape.w*sizeof(float);
        preds_ailia0 = cv::Mat(out0_shape.y, out0_shape.x, CV_32FC1);

        AILIAShape out1_shape;
        status = ailiaGetBlobShape(detector_net, &out1_shape, out1_inds, AILIA_SHAPE_VERSION);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaGetBlobShape failed %d\n", status);
            ailiaDestroy(detector_net);
            return -1;
        }
        out1_size = out1_shape.x*out1_shape.y*out1_shape.z*out1_shape.w*sizeof(float);
        preds_ailia1 = cv::Mat(out1_shape.y, out1_shape.x, CV_32FC1);
    }

    // web camera
    cv::VideoCapture capture;
    if (video_path == "0") {
        PRINT_OUT("[INFO] webcamera mode is activated\n");
        capture = cv::VideoCapture(0);
        if (!capture.isOpened()) {
            PRINT_ERR("[ERROR] webcamera not found\n");
            if (face == "yolov3") ailiaDestroyDetector(detector);
            ailiaDestroy(detector_net);
            return -1;
        }
    }
    else {
        if (check_file_existance(video_path.c_str())) {
            capture = cv::VideoCapture(video_path.c_str());
        }
        else {
            PRINT_ERR("[ERROR] \"%s\" not found\n", video_path.c_str());
            if (face == "yolov3") ailiaDestroyDetector(detector);
            ailiaDestroy(detector_net);
            return -1;
        }
    }

    // inference loop
    while (1) {
        cv::Mat frame;
        capture >> frame;
        if ((char)cv::waitKey(1) == 'q' || frame.empty()) {
            break;
        }
        int w = frame.cols;
        int h = frame.rows;

        // detect face
        unsigned int count;
        cv::Mat img;
        std::vector<cv::Mat> detections; // for BlazeFace
        if (face == "yolov3") {
            cv::Mat rsz_img;
            cv::cvtColor(frame, img, cv::COLOR_BGR2BGRA);
            cv::resize(img, rsz_img, cv::Size(YOLOV3_MODEL_INPUT_WIDTH, YOLOV3_MODEL_INPUT_HEIGHT));
            status = ailiaDetectorCompute(detector, rsz_img.data,
                                          YOLOV3_MODEL_INPUT_WIDTH*4, YOLOV3_MODEL_INPUT_WIDTH, YOLOV3_MODEL_INPUT_HEIGHT,
                                          AILIA_IMAGE_FORMAT_BGRA, YOLOV3_FACE_THRESHOLD, YOLOV3_FACE_IOU);
            if (status != AILIA_STATUS_SUCCESS) {
                PRINT_ERR("ailiaDetectorCompute failed %d\n", status);
                ailiaDestroyDetector(detector);
                ailiaDestroy(detector_net);
                return -1;
            }
            int status = ailiaDetectorGetObjectCount(detector, &count);
            if (status != AILIA_STATUS_SUCCESS) {
                PRINT_ERR("ailiaDetectorGetObjectCount failed %d\n",status);
                ailiaDestroyDetector(detector);
                ailiaDestroy(detector_net);
                return -1;
            }
        }
        else {
            cv::Mat rsz_img, trs_img, input;
            cv::cvtColor(frame, img, cv::COLOR_BGR2RGB);
            cv::resize(img, rsz_img, cv::Size(BLAZEFACE_INPUT_IMAGE_WIDTH, BLAZEFACE_INPUT_IMAGE_HEIGHT));
            transpose(rsz_img, trs_img, {2, 0, 1});
            normalize_image(trs_img, input, "127.5");

            status = ailiaSetInputBlobData(detector_net, input.data, input_size, in_inds);
            if (status != AILIA_STATUS_SUCCESS) {
                PRINT_ERR("ailiaSetInputBlobData failed %d\n", status);
                ailiaDestroy(detector_net);
                return -1;
            }

            status = ailiaUpdate(detector_net);
            if (status != AILIA_STATUS_SUCCESS) {
                PRINT_ERR("ailiaUpdate failed %d\n", status);
                ailiaDestroy(detector_net);
                return -1;
            }

            status = ailiaGetBlobData(detector_net, preds_ailia0.data, out0_size, out0_inds);
            if (status != AILIA_STATUS_SUCCESS) {
                PRINT_ERR("ailiaGetBlobData failed %d\n", status);
                ailiaDestroy(detector_net);
                return -1;
            }

            status = ailiaGetBlobData(detector_net, preds_ailia1.data, out1_size, out1_inds);
            if (status != AILIA_STATUS_SUCCESS) {
                PRINT_ERR("ailiaGetBlobData failed %d\n", status);
                ailiaDestroy(detector_net);
                return -1;
            }

            detections.clear();
            status = blazeface_postprocess(preds_ailia0, preds_ailia1, detections);
            if (status != AILIA_STATUS_SUCCESS) {
                PRINT_ERR("blazeface_postprocess failed %d\n", status);
                ailiaDestroy(detector_net);
                return -1;
            }

            count = (unsigned int)detections.size();
        }

        for (unsigned int i = 0; i < count; i++) {
            // get detected face
            AILIADetectorObject obj;
            float margin;
            if (face == "yolov3") {
                status = ailiaDetectorGetObject(detector, &obj, i, AILIA_DETECTOR_OBJECT_VERSION);
                if (status != AILIA_STATUS_SUCCESS) {
                    PRINT_ERR("ailiaDetectorGetObjectCount failed %d\n", status);
                    ailiaDestroyDetector(detector);
                    ailiaDestroy(detector_net);
                    return -1;
                }
                margin = 1.0f;
            }
            else {
                float* det_data = (float*)detections[i].data;
                obj.category = 0;
                obj.prob = 1.0f;
                obj.x    = det_data[1];
                obj.y    = det_data[0];
                obj.w    = det_data[3] - det_data[1];
                obj.h    = det_data[2] - det_data[0];
                margin = 1.4f;
            }

            float cx = (obj.x+obj.w/2.0f)*w;
            float cy = (obj.y+obj.h/2.0f)*h;
            float cw = std::max(obj.w*w*margin, obj.h*h*margin);
            float fx = std::max(cx-cw/2.0f, 0.0f);
            float fy = std::max(cy-cw/2.0f, 0.0f);
            float fw = std::min(cw, w-fx);
            float fh = std::min(cw, h-fy);
            cv::Point top_left((int)fx, (int)fy);
            cv::Point bottom_right((int)(fx+fw), (int)(fy+fh));

            // get detected face
            cv::Mat crop_img(img, cv::Rect((int)fx, (int)fy, (int)fw, (int)fh));
            if (crop_img.rows <= 0 || crop_img.cols <= 0) {
                continue;
            }
            cv::Mat resized_frame;
            adjust_frame_size(crop_img, resized_frame, IMAGE_HEIGHT, IMAGE_WIDTH);

            // get matched face
            int id_sim;
            float score_sim;
            status = face_identification(fe_list, net, resized_frame, id_sim, score_sim);
            if (status != AILIA_STATUS_SUCCESS) {
                if (face == "yolov3") ailiaDestroyDetector(detector);
                ailiaDestroy(detector_net);
                return -1;
            }

            // display result
            float fontScale = (float)w / 512.0f;
            int thickness = 2;
            cv::Scalar color = hsv_to_rgb((int)(256*((float)id_sim/16.0f)), 255, 255);
            cv::rectangle(frame, top_left, bottom_right, color, 2);
            cv::Point text_position((int)fx+4, (int)(fy+fh)-8);
            char score[20];
            snprintf(score, sizeof(score), "%d : %5.3f", id_sim, score_sim);
            cv::putText(frame, score, text_position, cv::FONT_HERSHEY_SIMPLEX, fontScale, color, thickness);
        }

        cv::imshow("frame", frame);
    }
    capture.release();
    cv::destroyAllWindows();

    if (face == "yolov3") {
        ailiaDestroyDetector(detector);
    }
    ailiaDestroy(detector_net);

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
        status = compare_video(net);
    }
    else {
        status = compare_images(net);
    }

    ailiaDestroy(net);

    return status;
}
