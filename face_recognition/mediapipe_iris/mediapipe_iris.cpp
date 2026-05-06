/*******************************************************************
*
*    DESCRIPTION:
*      AILIA mediapipe_iris sample
*    AUTHOR:
*
*    DATE:2022/08/19
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
#include "mat_utils.h"
#include "image_utils.h"
#include "detector_utils.h"
#include "webcamera_utils.h"
#include "blazeface_utils.h"


// ======================
// Parameters
// ======================

#define BLAZEFACE_WEIGHT_PATH "blazeface.opt.onnx"
#define BLAZEFACE_MODEL_PATH  "blazeface.opt.onnx.prototxt"
#define FACEMESH_WEIGHT_PATH  "facemesh.opt.onnx"
#define FACEMESH_MODEL_PATH   "facemesh.opt.onnx.prototxt"
#define IRIS_WEIGHT_PATH "iris.opt.onnx"
#define IRIS_MODEL_PATH  "iris.opt.onnx.prototxt"

#define IMAGE_PATH      "man.jpg"
#define SAVE_IMAGE_PATH "output.png"

#if defined(_WIN32) || defined(_WIN64)
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
#else
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif

#define BENCHMARK_ITERS 5

static std::string blazeface_weight(BLAZEFACE_WEIGHT_PATH);
static std::string blazeface_model(BLAZEFACE_MODEL_PATH);
static std::string facemesh_weight(FACEMESH_WEIGHT_PATH);
static std::string facemesh_model(FACEMESH_MODEL_PATH);
static std::string iris_weight(IRIS_WEIGHT_PATH);
static std::string iris_model(IRIS_MODEL_PATH);

static std::string image_path(IMAGE_PATH);
static std::string video_path("0");
static std::string save_image_path(SAVE_IMAGE_PATH);

static bool benchmark  = false;
static bool video_mode = false;
static int args_env_id = -1;
static float resolution = 192.0f;


// ======================
// Argument Parser
// ======================

static void print_usage()
{
    PRINT_OUT("usage: mediapipe_iris [-h] [-i IMAGE] [-v VIDEO] [-s SAVE_IMAGE_PATH] [-b] [-e ENV_ID]\n");
    return;
}


static void print_help()
{
    PRINT_OUT("\n");
    PRINT_OUT("mediapipe_iris model\n");
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
    PRINT_OUT("  -e ENV_ID, --env_id ENV_ID\n");
    PRINT_OUT("                        The backend environment id.\n");
    return;
}


static void print_error(std::string arg)
{
    PRINT_ERR("mediapipe_iris: error: unrecognized arguments: %s\n", arg.c_str());
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
            else if (arg == "-e" || arg == "--env_id") {
                status = 4;
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
            case 4:
                args_env_id = atoi(arg.c_str());
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


static void draw_landmarks(cv::Mat& mat_img, const std::vector<cv::Point2i>& points, const cv::Scalar& color, int size)
{
    for (const cv::Point2i& point : points) {
        cv::circle(mat_img, point, size, color, cv::FILLED);
    }
}


static void draw_eye_iris(cv::Mat& mat_img, const cv::Mat& mat_eyes, const cv::Mat& mat_iris)
{
    static int eye_contour_ordered[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 14, 13, 12, 11, 10, 9};
    static cv::Scalar color_eye(0, 0, 255, 255);
    static cv::Scalar color_iris(255, 0, 0, 255);
    static cv::Scalar color_point(0, 255, 0, 255);
    static int size = 1;

    assert(mat_eyes.size[1] == 16);

    std::vector<cv::Point2i> points_eye(mat_eyes.size[1]);
    std::vector<cv::Point2i> points_iris(mat_iris.size[1]);

    for (int x = 0; x < mat_eyes.size[0]; x++) {
        for (int y = 0; y < mat_eyes.size[1]; y++) {
            points_eye[y] = mat_eyes.at<cv::Point2i>(x, eye_contour_ordered[y]);
        }

        cv::polylines(mat_img, points_eye, true, color_eye, size);

        for (int y = 0; y < mat_iris.size[1]; y++) {
            points_iris[y] = mat_iris.at<cv::Point2i>(x, y);
        }

        cv::Point2i& center = points_iris[0];
        cv::Point2f normal = points_iris[1] - points_iris[0];
        int radius = (int)round(sqrt(normal.x * normal.x + normal.y * normal.y));

        cv::circle(mat_img, center, radius, color_iris, size);

        draw_landmarks(mat_img, points_iris, color_point, size);
    }
}


static void resize_pad(cv::Mat& mat_src, cv::Mat& mat_dst, float& scale, int pad[2])
{
    int h1, w1, padh, padw;
    if (mat_src.rows >= mat_src.cols) {
        h1 = 256;
        w1 = 256 * mat_src.cols / mat_src.rows;
        padh = 0;
        padw = 256 - w1;
        scale = (float)mat_src.cols / (float)w1;
    }
    else {
        h1 = 256 * mat_src.rows / mat_src.cols;
        w1 = 256;
        padh = 256 - h1;
        padw = 0;
        scale = (float)mat_src.rows / (float)h1;
    }

    int padh1 = padh / 2;
    int padh2 = padh / 2 + padh % 2;
    int padw1 = padw / 2;
    int padw2 = padw / 2 + padw % 2;

    cv::Mat mat_rsz;
    cv::resize(mat_src, mat_rsz, cv::Size(w1, h1));

    cv::Mat mat_pad;
    mat_pad.create(padh1 + mat_rsz.rows + padh2, padw1 + mat_rsz.cols + padw2, mat_rsz.type());
    mat_pad.setTo(cv::Scalar::all(0));
    mat_rsz.copyTo(mat_pad(cv::Rect(padw1, padh1, mat_rsz.cols, mat_rsz.rows)));

    cv::resize(mat_pad, mat_dst, cv::Size(128, 128));

    pad[0] = (int)((float)padh1 * scale);
    pad[1] = (int)((float)padw1 * scale);
}


static void denormalize_detections(cv::Mat& mat_detection, float scale, int pad[2])
{
    assert(mat_detection.cols > 4);

    float* data = (float*)mat_detection.data;

    data[0] = data[0] * scale * 256.0f - (float)pad[0];
    data[1] = data[1] * scale * 256.0f - (float)pad[1];
    data[2] = data[2] * scale * 256.0f - (float)pad[0];
    data[3] = data[3] * scale * 256.0f - (float)pad[1];

    for (int i = 4; i < mat_detection.cols; i += 2) {
        data[i] = data[i] * scale * 256.0f - (float)pad[1];
        data[i + 1] = data[i + 1] * scale * 256.0f - (float)pad[0];
    }
}


static void detection2roi(cv::Mat& mat_detection, float& xc, float& yc, float& scale, float& theta)
{
    // compute box center and scale
    // use mediapipe/calculators/util/detections_to_rects_calculator.cc

    static float dy = 0.0f;
    static float dscale = 1.5;
    static int kp1 = 1; // left eye
    static int kp2 = 0; // right eye
    static float theta0 = 0.0f;

    float* data = (float*)mat_detection.data;

    xc = (data[1] + data[3]) / 2.0f;
    yc = (data[0] + data[2]) / 2.0f;
    scale = data[3] - data[1];  // assumes square boxes

    yc += dy * scale;
    scale *= dscale;

    // compute box rotation
    float x0 = data[4 + 2 * kp1];
    float y0 = data[4 + 2 * kp1 + 1];
    float x1 = data[4 + 2 * kp2];
    float y1 = data[4 + 2 * kp2 + 1];
    theta = atan2(y0 - y1, x0 - x1) - theta0;
}


static void extract_roi(const cv::Mat& mat_input, float& xc, float& yc, float& scale, float& theta, cv::Mat& mat_image, cv::Mat& mat_affines)
{
    // take points on unit square and transform them according to the roi

    cv::Mat mat_points;
    {
        float data[] = {-1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f};
        mat_points = cv::Mat(2, 4, CV_32FC1, data).clone().mul(scale / 2.0f);
    }

    cv::Mat mat_r;
    {
        float data[] = {cos(theta), -sin(theta), sin(theta), cos(theta)};
        mat_r = cv::Mat(2, 2, CV_32FC1, data).clone();
    }

    cv::Mat mat_center;
    {
        float data[] = {xc, xc, xc, xc, yc, yc, yc, yc};
        mat_center = cv::Mat(2, 4, CV_32FC1, data).clone();
    }

    mat_points = mat_r * mat_points;
    mat_points = mat_points + mat_center;

    // use the points to compute the affine transform that maps
    // these points back to the output square

    static float res = resolution;

    cv::Mat mat_points1;
    {
        float data[] = {0.0f, 0.0f, 0.0f, res - 1.0f, res - 1.0f, 0.0f};
        mat_points1 = cv::Mat(3, 2, CV_32FC1, data).clone();
    }

    cv::Mat mat_pts;
    mat_pts = mat_points.colRange(0, 3).t();

    cv::Mat mat_m;
    mat_m = cv::getAffineTransform(mat_pts, mat_points1);

    cv::Mat mat_warp1;
    cv::warpAffine(mat_input, mat_warp1, mat_m, cv::Size((int)res, (int)res), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(127.5f));

    cv::Mat mat_warp2;
    normalize_image(mat_warp1, mat_warp2, "127.5");

    reshape_channels_as_dimensions(mat_warp2, mat_image);

    cv::Mat mat_inv1;
    cv::invertAffineTransform(mat_m, mat_inv1);

    cv::Mat mat_inv2;
    mat_inv1.convertTo(mat_inv2, CV_32F);

    expand_dims(mat_inv2, mat_affines, 0);
}


static void estimator_preprocess(const cv::Mat& mat_input, cv::Mat& mat_detection, float scale, int pad[2], cv::Mat& mat_image, cv::Mat& mat_affines)
{
    denormalize_detections(mat_detection, scale, pad);

    float xc, yc, theta;
    detection2roi(mat_detection, xc, yc, scale, theta);
    extract_roi(mat_input, xc, yc, scale, theta, mat_image, mat_affines);
}


static void denormalize_landmarks(const cv::Mat& mat_input, cv::Mat& mat_output, cv::Mat& mat_affines)
{
    assert(mat_input.dims == 3);
    assert(mat_input.size[2] == 3);

    cv::Mat mat_landmarks = mat_input.clone();

    for (int x = 0; x < mat_landmarks.size[0]; x++) {
        for (int y = 0; y < mat_landmarks.size[1]; y++) {
            mat_landmarks.at<float>(x, y, 0) *= resolution;
            mat_landmarks.at<float>(x, y, 1) *= resolution;
        }
    }

    mat_output = mat_input.clone();

    for (int x = 0; x < mat_landmarks.size[0]; x++) {
        cv::Mat mat_landmark;
        {
            cv::Range ranges[] = {cv::Range(x, x + 1), cv::Range::all(), cv::Range(0, 2)};
            int shape[] = {mat_landmarks.size[1], 2};
            mat_landmark = mat_landmarks(ranges).clone().reshape(1, 2, shape).t();
        }

        cv::Mat mat_affine1;
        {
            cv::Range ranges[] = {cv::Range(x, x + 1), cv::Range::all(), cv::Range(0, 2)};
            int shape[] = {mat_affines.size[1], 2};
            mat_affine1 = mat_affines(ranges).clone().reshape(1, 2, shape);
        }

        cv::Mat mat_affine2;
        {
            cv::Range ranges[] = {cv::Range(x, x + 1), cv::Range::all(), cv::Range(2, 3)};
            int shape[] = {mat_affines.size[1], 1};
            mat_affine2 = mat_affines(ranges).clone().reshape(1, 2, shape);
        }

        assert(mat_landmark.size[0] == mat_affine1.size[0]);
        assert(mat_landmark.size[0] == mat_affine2.size[0]);
        assert(mat_landmark.size[1] == mat_output.size[1]);

        mat_landmark = mat_affine1 * mat_landmark;

        for (int y = 0; y < mat_landmark.size[0]; y++) {
            for (int z = 0; z < mat_landmark.size[1]; z++) {
                mat_output.at<float>(x, z, y) = mat_landmark.at<float>(y, z) + mat_affine2.at<float>(y, 0);
            }
        }
    }
}


static void filter_landmarks(const cv::Mat& mat_input, cv::Mat& mat_output, int* points, int count)
{
    mat_output.create(count, 1, CV_32FC2);

    for (int i = 0; i < count; i++) {
        mat_output.at<float>(i, 0) = mat_input.at<float>(0, points[i], 0);
        mat_output.at<float>(i, 1) = mat_input.at<float>(0, points[i], 1);
    }
}


static void iris_preprocess(const cv::Mat& mat_input, const cv::Mat& mat_landmarks, cv::Mat& mat_images, std::vector<cv::Point2f>& vec_origins)
{
    static int eye_left_contour[] = {249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466};
    static int eye_right_contour[] = {7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246};

    cv::Mat mat_input2;
    reshape_dimensions_as_channels(mat_input, mat_input2);

    {
        cv::Mat mat_filter;
        filter_landmarks(mat_landmarks, mat_filter, eye_left_contour, sizeof(eye_left_contour) / sizeof(eye_left_contour[0]));

        cv::Scalar eye_center = cv::mean(mat_filter);
        int x = (int)round(eye_center[0] - 32.0);
        int y = (int)round(eye_center[1] - 32.0);
        vec_origins.push_back(cv::Point2f((float)(x + 63), (float)y));

        cv::Mat mat_image1;
        cv::flip(mat_input2(cv::Rect(x, y, 64, 64)), mat_image1, 1);

        cv::Mat mat_image2;
        reshape_channels_as_dimensions(mat_image1, mat_image2);

        mat_images.push_back(mat_image2);
    }

    {
        cv::Mat mat_filter;
        filter_landmarks(mat_landmarks, mat_filter, eye_right_contour, sizeof(eye_right_contour) / sizeof(eye_right_contour[0]));

        cv::Scalar eye_center = cv::mean(mat_filter);
        int x = (int)round(eye_center[0] - 32.0);
        int y = (int)round(eye_center[1] - 32.0);
        vec_origins.push_back(cv::Point2f((float)x, (float)y));

        cv::Mat mat_image1;
        mat_image1 = mat_input2(cv::Rect(x, y, 64, 64)).clone();

        cv::Mat mat_image2;
        reshape_channels_as_dimensions(mat_image1, mat_image2);

        mat_images.push_back(mat_image2);
    }
}


static void iris_postprocess(cv::Mat& mat_eyes, cv::Mat& mat_iris, cv::Mat& mat_affines, std::vector<cv::Point2f>& vec_origins)
{
    cv::Mat mat_eyes2;
    {
        int shape[] = {(int)mat_eyes.total() / 71 / 3, 71, 3};
        mat_eyes2 = mat_eyes.clone().reshape(1, 3, shape);

        for (int x = 0; x < mat_eyes2.size[0]; x++) {
            for (int y = 0; y < mat_eyes2.size[1]; y++) {
                if (x < 1) {
                    // horizontally flipped left eye processing
                    mat_eyes2.at<float>(x, y, 0) *= -1.0f;
                }

                mat_eyes2.at<float>(x, y, 0) += vec_origins[x % 2].x;
                mat_eyes2.at<float>(x, y, 1) += vec_origins[x % 2].y;
            }
        }
    }

    cv::Mat mat_iris2;
    {
        int shape[] = {(int)mat_iris.total() / 5 / 3, 5, 3};
        mat_iris2 = mat_iris.clone().reshape(1, 3, shape);

        for (int x = 0; x < mat_iris2.size[0]; x++) {
            for (int y = 0; y < mat_iris2.size[1]; y++) {
                if (x < 1) {
                    mat_iris2.at<float>(x, y, 0) *= -1.0f;
                }

                mat_iris2.at<float>(x, y, 0) += vec_origins[x % 2].x;
                mat_iris2.at<float>(x, y, 1) += vec_origins[x % 2].y;
            }
        }
    }

    cv::Mat mat_landmarks1;
    concatenate(mat_eyes2, mat_iris2, mat_landmarks1, 1);

    cv::Mat mat_landmarks2;
    {
        int shape[] = {mat_landmarks1.size[0] / 2, 0, 3};
        shape[1] = (int)mat_landmarks1.total() / shape[0] / shape[2];
        mat_landmarks2 = mat_landmarks1.reshape(1, 3, shape);

        for (int x = 0; x < mat_landmarks2.size[0]; x++) {
            for (int y = 0; y < mat_landmarks2.size[1]; y++) {
                for (int z = 0; z < mat_landmarks2.size[2]; z++) {
                    mat_landmarks2.at<float>(x, y, z) /= resolution;
                }
            }
        }
    }

    cv::Mat mat_landmarks3;
    denormalize_landmarks(mat_landmarks2, mat_landmarks3, mat_affines);

    cv::Mat mat_landmarks4;
    {
        mat_landmarks4.create(mat_landmarks3.dims, mat_landmarks3.size, CV_32SC1);

        for (int x = 0; x < mat_landmarks3.size[0]; x++) {
            for (int y = 0; y < mat_landmarks3.size[1]; y++) {
                for (int z = 0; z < mat_landmarks3.size[2]; z++) {
                    mat_landmarks4.at<int>(x, y, z) = (int)round(mat_landmarks3.at<float>(x, y, z));
                }
            }
        }
    }

    cv::Mat mat_landmarks5;
    {
        int shape[] = {(int)mat_landmarks4.total() / 2 / 76 / 3, 2, 76, 3};
        mat_landmarks5 = mat_landmarks4.reshape(1, 4, shape);
    }

    {
        cv::Range ranges[] = {cv::Range::all(), cv::Range::all(), cv::Range(0, 71), cv::Range::all()};
        mat_eyes = mat_landmarks5(ranges).clone();
    }

    {
        cv::Range ranges[] = {cv::Range::all(), cv::Range::all(), cv::Range(71, mat_landmarks5.size[2]), cv::Range::all()};
        mat_iris = mat_landmarks5(ranges).clone();
    }
}


static int detect_face(AILIANetwork* ailia, const cv::Mat& mat_input, std::vector<cv::Mat>& vec_outputs)
{
    int status = AILIA_STATUS_SUCCESS;

    unsigned input_count;
    status = ailiaGetInputBlobCount(ailia, &input_count);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetInputBlobCount failed %d\n", status);
        return status;
    }

    if (input_count != 1) {
        PRINT_ERR("ailiaGetInputBlobCount returned %u\n", input_count);
        return AILIA_STATUS_OTHER_ERROR;
    }

    unsigned int input_index;
    status = ailiaGetBlobIndexByInputIndex(ailia, &input_index, 0);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetBlobIndexByInputIndex failed %d\n", status);
        return status;
    }

    AILIAShape input_shape;
    status = ailiaGetBlobShape(ailia, &input_shape, input_index, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetBlobShape failed %d\n", status);
        return status;
    }

    int input_size = input_shape.x * input_shape.y * input_shape.z * input_shape.w * sizeof(float);

    assert(input_shape.dim == (unsigned int)mat_input.dims);
    assert(input_shape.w == (unsigned int)mat_input.size[0]);
    assert(input_shape.z == (unsigned int)mat_input.size[1]);
    assert(input_shape.y == (unsigned int)mat_input.size[2]);
    assert(input_shape.x == (unsigned int)mat_input.size[3]);
    assert((int)(mat_input.total() * mat_input.elemSize()) == input_size);

    status = ailiaSetInputBlobData(ailia, mat_input.data, input_size, input_index);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaSetInputBlobData failed %d\n", status);
        return status;
    }

    status = ailiaUpdate(ailia);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaUpdate failed %d\n", status);
        return status;
    }

    unsigned output_count;
    status = ailiaGetOutputBlobCount(ailia, &output_count);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetOutputBlobCount failed %d\n", status);
        return status;
    }

    if (output_count != vec_outputs.size()) {
        PRINT_ERR("ailiaGetOutputBlobCount returned %u\n", output_count);
        return AILIA_STATUS_OTHER_ERROR;
    }

    for (int i = 0; i < (int)output_count; i++) {
        cv::Mat& mat_output = vec_outputs[i];

        unsigned int output_index;
        status = ailiaGetBlobIndexByOutputIndex(ailia, &output_index, i);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaGetBlobIndexByOutputIndex failed %d\n", status);
            return status;
        }

        AILIAShape output_shape;
        status = ailiaGetBlobShape(ailia, &output_shape, output_index, AILIA_SHAPE_VERSION);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaGetBlobShape failed %d\n", status);
            return status;
        }

        int output_size = output_shape.x * output_shape.y * output_shape.z * output_shape.w * sizeof(float);

        assert(output_shape.dim == 3);
        assert(output_shape.z == 1);

        mat_output = cv::Mat((int)output_shape.y, (int)output_shape.x, CV_32FC1);

        assert((int)(mat_output.total() * mat_output.elemSize()) == output_size);

        status = ailiaGetBlobData(ailia, mat_output.data, output_size, output_index);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaSetOutputBlobData failed %d\n", status);
            return status;
        }
    }

    return AILIA_STATUS_SUCCESS;
}


static int estimate_landmarks(AILIANetwork* ailia, const cv::Mat& mat_input, std::vector<cv::Mat>& vec_outputs)
{
    int status = AILIA_STATUS_SUCCESS;

    unsigned input_count;
    status = ailiaGetInputBlobCount(ailia, &input_count);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetInputBlobCount failed %d\n", status);
        return status;
    }

    if (input_count != 1) {
        PRINT_ERR("ailiaGetInputBlobCount returned %u\n", input_count);
        return AILIA_STATUS_OTHER_ERROR;
    }

    unsigned int input_index;
    status = ailiaGetBlobIndexByInputIndex(ailia, &input_index, 0);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetBlobIndexByInputIndex failed %d\n", status);
        return status;
    }

    AILIAShape input_shape;
    status = ailiaGetBlobShape(ailia, &input_shape, input_index, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetBlobShape failed %d\n", status);
        return status;
    }

    int input_size = input_shape.x * input_shape.y * input_shape.z * input_shape.w * sizeof(float);

    assert(input_shape.dim == (unsigned int)mat_input.dims);
    assert(input_shape.w == (unsigned int)mat_input.size[0]);
    assert(input_shape.z == (unsigned int)mat_input.size[1]);
    assert(input_shape.y == (unsigned int)mat_input.size[2]);
    assert(input_shape.x == (unsigned int)mat_input.size[3]);
    assert((int)(mat_input.total() * mat_input.elemSize()) == input_size);

    status = ailiaSetInputBlobData(ailia, mat_input.data, input_size, input_index);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaSetInputBlobData failed %d\n", status);
        return status;
    }

    status = ailiaUpdate(ailia);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaUpdate failed %d\n", status);
        return status;
    }

    unsigned output_count;
    status = ailiaGetOutputBlobCount(ailia, &output_count);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetOutputBlobCount failed %d\n", status);
        return status;
    }

    if (output_count != vec_outputs.size()) {
        PRINT_ERR("ailiaGetOutputBlobCount returned %u\n", output_count);
        return AILIA_STATUS_OTHER_ERROR;
    }

    for (int i = 0; i < (int)output_count; i++) {
        cv::Mat& mat_output = vec_outputs[i];

        unsigned int output_index;
        status = ailiaGetBlobIndexByOutputIndex(ailia, &output_index, i);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaGetBlobIndexByOutputIndex failed %d\n", status);
            return status;
        }

        AILIAShape output_shape;
        status = ailiaGetBlobShape(ailia, &output_shape, output_index, AILIA_SHAPE_VERSION);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaGetBlobShape failed %d\n", status);
            return status;
        }

        int output_size = output_shape.x * output_shape.y * output_shape.z * output_shape.w * sizeof(float);

        assert(output_shape.dim == 2);
        assert(output_shape.w == 1);
        assert(output_shape.z == 1);

        mat_output = cv::Mat((int)output_shape.y, (int)output_shape.x, CV_32FC1);

        assert((int)(mat_output.total() * mat_output.elemSize()) == output_size);

        status = ailiaGetBlobData(ailia, mat_output.data, output_size, output_index);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaSetOutputBlobData failed %d\n", status);
            return status;
        }
    }

    {
        int shape[] = {vec_outputs[0].size[0], vec_outputs[0].size[1] / 3, 3};
        vec_outputs[0] = vec_outputs[0].reshape(1, 3, shape);
    }

    return AILIA_STATUS_SUCCESS;
}


static int estimate_iris(AILIANetwork* ailia, const cv::Mat& mat_input, std::vector<cv::Mat>& vec_outputs)
{
    int status = AILIA_STATUS_SUCCESS;

    unsigned input_count;
    status = ailiaGetInputBlobCount(ailia, &input_count);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetInputBlobCount failed %d\n", status);
        return status;
    }

    if (input_count != 1) {
        PRINT_ERR("ailiaGetInputBlobCount returned %u\n", input_count);
        return AILIA_STATUS_OTHER_ERROR;
    }

    unsigned int input_index;
    status = ailiaGetBlobIndexByInputIndex(ailia, &input_index, 0);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetBlobIndexByInputIndex failed %d\n", status);
        return status;
    }

    assert(mat_input.dims == 4);

    AILIAShape input_shape;
    input_shape.dim = mat_input.dims;
    input_shape.w = mat_input.size[0];
    input_shape.z = mat_input.size[1];
    input_shape.y = mat_input.size[2];
    input_shape.x = mat_input.size[3];

    status = ailiaSetInputBlobShape(ailia, &input_shape, input_index, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaSetInputBlobShape failed %d\n", status);
        return status;
    }

    int input_size = input_shape.x * input_shape.y * input_shape.z * input_shape.w * sizeof(float);

    assert((int)(mat_input.total() * mat_input.elemSize()) == input_size);

    status = ailiaSetInputBlobData(ailia, mat_input.data, input_size, input_index);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaSetInputBlobData failed %d\n", status);
        return status;
    }

    status = ailiaUpdate(ailia);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaUpdate failed %d\n", status);
        return status;
    }

    unsigned output_count;
    status = ailiaGetOutputBlobCount(ailia, &output_count);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetOutputBlobCount failed %d\n", status);
        return status;
    }

    if (output_count != vec_outputs.size()) {
        PRINT_ERR("ailiaGetOutputBlobCount returned %u\n", output_count);
        return AILIA_STATUS_OTHER_ERROR;
    }

    for (int i = 0; i < (int)output_count; i++) {
        cv::Mat& mat_output = vec_outputs[i];

        unsigned int output_index;
        status = ailiaGetBlobIndexByOutputIndex(ailia, &output_index, i);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaGetBlobIndexByOutputIndex failed %d\n", status);
            return status;
        }

        AILIAShape output_shape;
        status = ailiaGetBlobShape(ailia, &output_shape, output_index, AILIA_SHAPE_VERSION);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaGetBlobShape failed %d\n", status);
            return status;
        }

        int output_size = output_shape.x * output_shape.y * output_shape.z * output_shape.w * sizeof(float);

        assert(output_shape.dim == 2);
        assert(output_shape.w == 1);
        assert(output_shape.z == 1);

        mat_output = cv::Mat((int)output_shape.y, (int)output_shape.x, CV_32FC1);

        assert((int)(mat_output.total() * mat_output.elemSize()) == output_size);

        status = ailiaGetBlobData(ailia, mat_output.data, output_size, output_index);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaSetOutputBlobData failed %d\n", status);
            return status;
        }
    }

    return AILIA_STATUS_SUCCESS;
}


// ======================
// Main functions
// ======================

static int recognize(AILIANetwork* ailia_blazeface, AILIANetwork* ailia_facemesh, AILIANetwork* ailia_iris, cv::Mat& mat_img)
{
    int status = AILIA_STATUS_SUCCESS;

    // prepare image
    cv::Mat mat_rgb;
    cv::cvtColor(mat_img, mat_rgb, cv::COLOR_BGRA2RGB);

    cv::Mat mat_data;
    float scale;
    int pad[2];
    {
        cv::Mat mat_data2;
        resize_pad(mat_rgb, mat_data2, scale, pad);

        cv::Mat mat_data3;
        normalize_image(mat_data2, mat_data3, "127.5");

        cv::Mat mat_data4;
        reshape_channels_as_dimensions(mat_data3, mat_data);
    }

    // face detection
    std::vector<cv::Mat> vec_predictions(2);
    status = detect_face(ailia_blazeface, mat_data, vec_predictions);
    if (status != AILIA_STATUS_SUCCESS) {
        return status;
    }

    std::vector<cv::Mat> vec_detections;
    status = blazeface_postprocess(vec_predictions[0], vec_predictions[1], vec_detections);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("blazeface_postprocess failed %d\n", status);
        return status;
    }

    if (vec_detections.size() > 0) {
        // face landmark estimation
        cv::Mat mat_image, mat_affines;
        estimator_preprocess(mat_rgb, vec_detections[0], scale, pad, mat_image, mat_affines);

        std::vector<cv::Mat> vec_estimates1(2);
        status = estimate_landmarks(ailia_facemesh, mat_image, vec_estimates1);
        if (status != AILIA_STATUS_SUCCESS) {
            return status;
        }

        cv::Mat& mat_landmarks = vec_estimates1[0];
        // unused: cv::Mat& mat_confidence = vec_estimates1[1];

        // iris landmark estimation
        cv::Mat mat_images;
        std::vector<cv::Point2f> vec_origins;
        iris_preprocess(mat_image, mat_landmarks, mat_images, vec_origins);

        std::vector<cv::Mat> vec_estimates2(2);
        status = estimate_iris(ailia_iris, mat_images, vec_estimates2);
        if (status != AILIA_STATUS_SUCCESS) {
            return status;
        }

        cv::Mat& mat_eyes1 = vec_estimates2[0];
        cv::Mat& mat_iris1 = vec_estimates2[1];

        iris_postprocess(mat_eyes1, mat_iris1, mat_affines, vec_origins);

        for (int x = 0; x < mat_eyes1.size[0]; x++) {
            cv::Mat mat_eyes2;
            {
                cv::Range ranges[] = {cv::Range(x, x + 1), cv::Range::all(), cv::Range(0, 16), cv::Range(0, 2)};
                int shape[] = {mat_eyes1.size[1], 16};
                mat_eyes2 = mat_eyes1(ranges).clone().reshape(2, 2, shape);
            }

            cv::Mat mat_iris2;
            {
                cv::Range ranges[] = {cv::Range(x, x + 1), cv::Range::all(), cv::Range::all(), cv::Range(0, 2)};
                int shape[] = {mat_iris1.size[1], mat_iris1.size[2]};
                mat_iris2 = mat_iris1(ranges).clone().reshape(2, 2, shape);
            }

            draw_eye_iris(mat_img, mat_eyes2, mat_iris2);
        }
    }

    return AILIA_STATUS_SUCCESS;
}


static int recognize_from_image(AILIANetwork* ailia_blazeface, AILIANetwork* ailia_facemesh, AILIANetwork* ailia_iris)
{
    int status = AILIA_STATUS_SUCCESS;

    // load image
    cv::Mat mat_img;
    status = load_image(mat_img, image_path.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }
    print_shape(mat_img, "input image shape: ");

    // inference
    PRINT_OUT("Start inference...\n");
    if (benchmark) {
        PRINT_OUT("BENCHMARK mode\n");
        for (int i = 0; i < BENCHMARK_ITERS; i++) {
            clock_t start = clock();
            status = recognize(ailia_blazeface, ailia_facemesh, ailia_iris, mat_img);
            if (status != AILIA_STATUS_SUCCESS) {
                return -1;
            }
            clock_t end = clock();
            PRINT_OUT("\tailia processing time %ld ms\n", ((end - start) * 1000) / CLOCKS_PER_SEC);
        }
    }
    else {
        status = recognize(ailia_blazeface, ailia_facemesh, ailia_iris, mat_img);
        if (status != AILIA_STATUS_SUCCESS) {
            return -1;
        }
    }

    PRINT_OUT("saved at : %s\n", save_image_path.c_str());
    cv::imwrite(save_image_path.c_str(), mat_img);

    PRINT_OUT("Program finished successfully.\n");

    return AILIA_STATUS_SUCCESS;
}


static int recognize_from_video(AILIANetwork* ailia_blazeface, AILIANetwork* ailia_facemesh, AILIANetwork* ailia_iris)
{
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
        cv::Mat mat_frame;
        capture >> mat_frame;
        if ((char)cv::waitKey(1) == 'q' || mat_frame.empty()) {
            break;
        }

        int status = recognize(ailia_blazeface, ailia_facemesh, ailia_iris, mat_frame);
        if (status != AILIA_STATUS_SUCCESS) {
            return -1;
        }

        cv::imshow("frame", mat_frame);
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

    // env list
    unsigned int env_count;
    status = ailiaGetEnvironmentCount(&env_count);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetEnvironmentCount failed %d", status);
        return -1;
    }

    int env_id = AILIA_ENVIRONMENT_ID_AUTO;
    for (unsigned int i = 0; i < env_count; i++) {
        AILIAEnvironment* env;
        status = ailiaGetEnvironment(&env, i, AILIA_ENVIRONMENT_VERSION);
        PRINT_OUT("env_id : %d type : %d name : %s\n", env->id, env->type, env->name);
        if (args_env_id == env->id) {
            env_id = env->id;
        }
        if (args_env_id == -1 && env_id == AILIA_ENVIRONMENT_ID_AUTO){
            if (env->type == AILIA_ENVIRONMENT_TYPE_GPU) {
                env_id = env->id;
            }
        }
    }
    if (args_env_id == -1){
        PRINT_OUT("you can select environment using -e option\n");
    }

    // initialize blazeface net
    AILIANetwork *ailia_blazeface;
    {
        status = ailiaCreate(&ailia_blazeface, env_id, AILIA_MULTITHREAD_AUTO);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaCreate failed %d\n", status);
            if (status == AILIA_STATUS_LICENSE_NOT_FOUND || status==AILIA_STATUS_LICENSE_EXPIRED){
                PRINT_OUT("License file not found or expired : please place license file (AILIA.lic)\n");
            }
            return -1;
        }

        AILIAEnvironment *env_ptr = nullptr;
        status = ailiaGetSelectedEnvironment(ailia_blazeface, &env_ptr, AILIA_ENVIRONMENT_VERSION);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaGetSelectedEnvironment failed %d\n", status);
            ailiaDestroy(ailia_blazeface);
            return -1;
        }

        PRINT_OUT("selected env name : %s\n", env_ptr->name);

        status = ailiaOpenStreamFile(ailia_blazeface, blazeface_model.c_str());
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaOpenStreamFile failed %d\n", status);
            PRINT_ERR("ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(ailia_blazeface));
            ailiaDestroy(ailia_blazeface);
            return -1;
        }

        status = ailiaOpenWeightFile(ailia_blazeface, blazeface_weight.c_str());
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
            ailiaDestroy(ailia_blazeface);
            return -1;
        }
    }

    // initialize facemesh net
    AILIANetwork *ailia_facemesh;
    {
        status = ailiaCreate(&ailia_facemesh, env_id, AILIA_MULTITHREAD_AUTO);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaCreate failed %d\n", status);
            if (status == AILIA_STATUS_LICENSE_NOT_FOUND || status==AILIA_STATUS_LICENSE_EXPIRED){
                PRINT_OUT("License file not found or expired : please place license file (AILIA.lic)\n");
            }
            return -1;
        }

        AILIAEnvironment *env_ptr = nullptr;
        status = ailiaGetSelectedEnvironment(ailia_facemesh, &env_ptr, AILIA_ENVIRONMENT_VERSION);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaGetSelectedEnvironment failed %d\n", status);
            ailiaDestroy(ailia_facemesh);
            return -1;
        }

        status = ailiaOpenStreamFile(ailia_facemesh, facemesh_model.c_str());
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaOpenStreamFile failed %d\n", status);
            PRINT_ERR("ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(ailia_facemesh));
            ailiaDestroy(ailia_facemesh);
            return -1;
        }

        status = ailiaOpenWeightFile(ailia_facemesh, facemesh_weight.c_str());
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
            ailiaDestroy(ailia_facemesh);
            return -1;
        }
    }

    // initialize iris net
    AILIANetwork *ailia_iris;
    {
        status = ailiaCreate(&ailia_iris, env_id, AILIA_MULTITHREAD_AUTO);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaCreate failed %d\n", status);
            if (status == AILIA_STATUS_LICENSE_NOT_FOUND || status==AILIA_STATUS_LICENSE_EXPIRED){
                PRINT_OUT("License file not found or expired : please place license file (AILIA.lic)\n");
            }
            return -1;
        }

        AILIAEnvironment *env_ptr = nullptr;
        status = ailiaGetSelectedEnvironment(ailia_iris, &env_ptr, AILIA_ENVIRONMENT_VERSION);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaGetSelectedEnvironment failed %d\n", status);
            ailiaDestroy(ailia_iris);
            return -1;
        }

        status = ailiaOpenStreamFile(ailia_iris, iris_model.c_str());
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaOpenStreamFile failed %d\n", status);
            PRINT_ERR("ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(ailia_iris));
            ailiaDestroy(ailia_iris);
            return -1;
        }

        status = ailiaOpenWeightFile(ailia_iris, iris_weight.c_str());
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
            ailiaDestroy(ailia_iris);
            return -1;
        }
    }

    if (video_mode) {
        status = recognize_from_video(ailia_blazeface, ailia_facemesh, ailia_iris);
    }
    else {
        status = recognize_from_image(ailia_blazeface, ailia_facemesh, ailia_iris);
    }

    ailiaDestroy(ailia_blazeface);
    ailiaDestroy(ailia_facemesh);
    ailiaDestroy(ailia_iris);

    return status;
}
