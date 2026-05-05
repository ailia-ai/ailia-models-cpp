/*******************************************************************
*
*    DESCRIPTION:
*      AILIA retinaface sample
*    AUTHOR:
*
*    DATE:2024/11/01
*
*******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>

#include <opencv2/opencv.hpp>

#undef UNICODE

#include "ailia.h"
#include "ailia_detector.h"
#include "utils.h"
#include "detector_utils.h"
#include "webcamera_utils.h"

using namespace std;


// ======================
// Parameters
// ======================

#define WEIGHT_PATH "retinaface_resnet50.onnx"
#define MODEL_PATH  "retinaface_resnet50.onnx.prototxt"

#define WEIGHT_PATH_MOBILE "retinaface_mobile0.25.onnx"
#define MODEL_PATH_MOBILE  "retinaface_mobile0.25.onnx.prototxt"

#define IMAGE_PATH      "selfie.png"
#define SAVE_IMAGE_PATH "output.png"

#define IMAGE_WIDTH        1280 // for video mode
#define IMAGE_HEIGHT       768 // for video mode

#if defined(_WIN32) || defined(_WIN64)
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
#else
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif

#define BENCHMARK_ITERS 5

static std::string image_path(IMAGE_PATH);
static std::string video_path("0");
static std::string save_image_path(SAVE_IMAGE_PATH);

static bool benchmark  = false;
static bool mobile  = false;
static bool video_mode = false;


// ======================
// Arguemnt Parser
// ======================

static void print_usage()
{
    PRINT_OUT("usage: yolov3-face [-h] [-i IMAGE] [-v VIDEO] [-s SAVE_IMAGE_PATH] [-b]\n");
    return;
}


static void print_help()
{
    PRINT_OUT("\n");
    PRINT_OUT("retinaface model\n");
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
    PRINT_OUT("  -m, --mobile          Use mobile version model.\n");
    return;
}


static void print_error(std::string arg)
{
    PRINT_ERR("retinaface: error: unrecognized arguments: %s\n", arg.c_str());
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
            else if (arg == "-m" || arg == "--mobile") {
                mobile = true;
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
// Pre and Post Processing
// ======================

const int RETINAFACE_DETECTOR_INPUT_CHANNEL_COUNT = 3;
const int RETINAFACE_DETECTOR_INPUT_BATCH_SIZE = 1;
const int BOX_DIM = 4;
const int SCORE_DIM = 2;
const int LANDMARK_DIM = 10;
const int NUM_KEYPOINTS = 5;

int TOP_K = 5000;
int KEEP_TOP_K = 750;
const float CONFIDENCE_THRES = 0.02f;
const float NMS_THRES = 0.4f;
const float VIS_THRES = 0.6f;
float VARIANCE[] = {0.1f, 0.2f};

struct FaceInfo {
    float score;
    pair<float, float> center;
    float width;
    float height;
    vector<pair<float, float>> keypoints;
};

vector<vector<float>> reshape(const vector<float>& array, int columns) {
    int rows = array.size() / columns;
    vector<vector<float>> reshaped(rows, vector<float>(columns));
    for (size_t i = 0; i < array.size(); ++i) {
        int row = i / columns;
        int column = i % columns;
        reshaped[row][column] = array[i];
    }
    return reshaped;
}

vector<vector<float>> prior_box_forward(int image_width, int image_height) {
    vector<vector<int>> minSizes = {{16, 32}, {64, 128}, {256, 512}};
    vector<int> steps = {8, 16, 32};
    vector<int> image_size = {image_height, image_width};
    vector<vector<int>> featureMaps;

    for (int step : steps) {
        featureMaps.push_back({ static_cast<int>(ceil(static_cast<float>(image_size[0]) / step)),
                                static_cast<int>(ceil(static_cast<float>(image_size[1]) / step)) });
    }

    vector<float> anchors;
    for (size_t k = 0; k < featureMaps.size(); ++k) {
        for (int i = 0; i < featureMaps[k][0]; ++i) {
            for (int j = 0; j < featureMaps[k][1]; ++j) {
                for (int minSize : minSizes[k]) {
                    float s_kx = minSize / static_cast<float>(image_size[1]);
                    float s_ky = minSize / static_cast<float>(image_size[0]);
                    float dense_cx = (j + 0.5f) * steps[k] / static_cast<float>(image_size[1]);
                    float dense_cy = (i + 0.5f) * steps[k] / static_cast<float>(image_size[0]);
                    anchors.push_back(dense_cx);
                    anchors.push_back(dense_cy);
                    anchors.push_back(s_kx);
                    anchors.push_back(s_ky);
                }
            }
        }
    }

    return reshape(anchors, 4);
}

vector<vector<float>> decode_box(const vector<vector<float>>& src_box, const vector<vector<float>>& priors) {
    int numBoxes = priors.size();
    vector<vector<float>> dst_box(numBoxes, vector<float>(4));

    for (int i = 0; i < numBoxes; ++i) {
        float center_x = priors[i][0] + src_box[i][0] * VARIANCE[0] * priors[i][2];
        float center_y = priors[i][1] + src_box[i][1] * VARIANCE[0] * priors[i][3];
        float width = priors[i][2] * exp(src_box[i][2] * VARIANCE[1]);
        float height = priors[i][3] * exp(src_box[i][3] * VARIANCE[1]);

        dst_box[i][0] = center_x - width / 2;
        dst_box[i][1] = center_y - height / 2;
        dst_box[i][2] = center_x + width / 2;
        dst_box[i][3] = center_y + height / 2;
    }
    return dst_box;
}

vector<float> decode_score(const vector<vector<float>>& src_score) {
    int numElements = src_score.size();
    vector<float> dst_score(numElements);
    for (int i = 0; i < numElements; ++i) {
        dst_score[i] = src_score[i][1];
    }
    return dst_score;
}

vector<vector<float>> decode_landmark(const vector<vector<float>>& src_landmark, const vector<vector<float>>& priors) {
    int numBoxes = priors.size();
    vector<vector<float>> dst_landmark(numBoxes, vector<float>(10));
    for (int i = 0; i < numBoxes; ++i) {
        for (int j = 0; j < NUM_KEYPOINTS; ++j) {
            dst_landmark[i][2 * j] = priors[i][0] + src_landmark[i][2 * j] * VARIANCE[0] * priors[i][2];
            dst_landmark[i][2 * j + 1] = priors[i][1] + src_landmark[i][2 * j + 1] * VARIANCE[0] * priors[i][3];
        }
    }
    return dst_landmark;
}

vector<vector<float>> scale(const vector<vector<float>>& src, const vector<int>& scale) {
    int src_width = src.size();
    int src_height = src[0].size();
    vector<vector<float>> dst(src_width, vector<float>(src_height));

    for (int i = 0; i < src_width; ++i) {
        for (int j = 0; j < src_height; ++j) {
            dst[i][j] = src[i][j] * scale[j];
        }
    }
    return dst;
}

vector<float> filter(const vector<float>& src, const vector<int>& inds) {
    vector<float> dst(inds.size());
    for (size_t i = 0; i < inds.size(); ++i) {
        dst[i] = src[inds[i]];
    }
    return dst;
}

vector<vector<float>> filter(const vector<vector<float>>& src, const vector<int>& inds) {
    int src_element_num = src[0].size();
    vector<vector<float>> dst(inds.size(), vector<float>(src_element_num));

    for (size_t i = 0; i < inds.size(); ++i) {
        for (int j = 0; j < src_element_num; ++j) {
            dst[i][j] = src[inds[i]][j];
        }
    }

    return dst;
}

vector<float> keep_top_k_before_nms(const vector<float>& src, const vector<int>& order, int top_k) {
    vector<float> dst(top_k);
    for (int i = 0; i < top_k; ++i) {
        dst[i] = src[order[i]];
    }
    return dst;
}

vector<vector<float>> keep_top_k_before_nms(const vector<vector<float>>& src, const vector<int>& order, int top_k) {
    int src_element_num = src[0].size();
    vector<vector<float>> dst(top_k, vector<float>(src_element_num));

    for (int i = 0; i < top_k; ++i) {
        for (int j = 0; j < src_element_num; ++j) {
            dst[i][j] = src[order[i]][j];
        }
    }
    return dst;
}

vector<int> nms(const vector<vector<float>>& boxes, const vector<float>& scores, float thresh) {
    int numBoxes = boxes.size();
    vector<float> x1(numBoxes), y1(numBoxes), x2(numBoxes), y2(numBoxes);
    
    for (int i = 0; i < numBoxes; ++i) {
        x1[i] = boxes[i][0];
        y1[i] = boxes[i][1];
        x2[i] = boxes[i][2];
        y2[i] = boxes[i][3];
    }

    vector<float> areas(numBoxes);
    for (int i = 0; i < numBoxes; ++i) {
        areas[i] = (x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1);
    }

    vector<int> order(numBoxes);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&scores](int a, int b) { return scores[a] > scores[b]; });

    vector<int> keep;
    while (!order.empty()) {
        int i = order[0];
        keep.push_back(i);

        vector<int> newOrder;
        for (size_t j = 1; j < order.size(); ++j) {
            int k = order[j];
            float xx1 = max(x1[i], x1[k]);
            float yy1 = max(y1[i], y1[k]);
            float xx2 = min(x2[i], x2[k]);
            float yy2 = min(y2[i], y2[k]);

            float w = max(0.0f, xx2 - xx1 + 1);
            float h = max(0.0f, yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (areas[i] + areas[k] - inter);
            if (ovr <= thresh) {
                newOrder.push_back(k);
            }
        }
        order = newOrder;
    }
    return keep;
}

bool compare_indices(const std::pair<int, float>& a, const std::pair<int, float>& b) {
    return a.second > b.second;
}

std::vector<int> sort_indices_by_score(const std::vector<float>& filtered_score) {
    std::vector<std::pair<int, float>> indexed_scores;
    for (size_t i = 0; i < filtered_score.size(); ++i) {
        indexed_scores.emplace_back(i, filtered_score[i]);
    }

    std::sort(indexed_scores.begin(), indexed_scores.end(), compare_indices);

    std::vector<int> order;
    for (const auto& pair : indexed_scores) {
        order.push_back(pair.first);
    }

    return order;
}

vector<FaceInfo> post_process(const vector<float>& box_data, const vector<float>& score_data, const vector<float>& landmark_data, int tex_width, int tex_height) {
    vector<FaceInfo> results;

    vector<int> boxShape = {static_cast<int>(box_data.size() / BOX_DIM), BOX_DIM};
    vector<vector<float>> reshaped_box_data = reshape(box_data, boxShape[1]);
    vector<vector<float>> priors = prior_box_forward(tex_width, tex_height);
    vector<vector<float>> boxes = decode_box(reshaped_box_data, priors);
    vector<int> box_scale = {tex_width, tex_height, tex_width, tex_height};
    vector<vector<float>> scaled_boxes = scale(boxes, box_scale);

    vector<int> scoreShape = {static_cast<int>(score_data.size() / SCORE_DIM), SCORE_DIM};
    vector<vector<float>> reshaped_score_data = reshape(score_data, scoreShape[1]);
    vector<float> scores = decode_score(reshaped_score_data);

    vector<int> landmarkShape = {static_cast<int>(landmark_data.size() / LANDMARK_DIM), LANDMARK_DIM};
    vector<vector<float>> reshaped_landmark_data = reshape(landmark_data, landmarkShape[1]);
    vector<vector<float>> landmarks = decode_landmark(reshaped_landmark_data, priors);
    vector<int> landmark_scale = {tex_width, tex_height, tex_width, tex_height, tex_width, tex_height, tex_width, tex_height, tex_width, tex_height};
    vector<vector<float>> scaled_landmarks = scale(landmarks, landmark_scale);

    vector<int> inds;
    for (size_t i = 0; i < scores.size(); i++) {
        if (scores[i] > CONFIDENCE_THRES) {
            inds.push_back(i);
        }
    }

    vector<vector<float>> filtered_boxes = filter(scaled_boxes, inds);
    vector<float> filtered_scores = filter(scores, inds);
    vector<vector<float>> filtered_landmarks = filter(scaled_landmarks, inds);

    if (filtered_scores.size() == 0){
        return results;
    }

    int top_k = min(TOP_K, static_cast<int>(filtered_scores.size()));
    vector<int> order = sort_indices_by_score(filtered_scores);
    vector<vector<float>> top_k_boxes = keep_top_k_before_nms(filtered_boxes, order, top_k);
    vector<float> top_k_scores = keep_top_k_before_nms(filtered_scores, order, top_k);
    vector<vector<float>> top_k_landmarks = keep_top_k_before_nms(filtered_landmarks, order, top_k);

    vector<int> keep = nms(top_k_boxes, top_k_scores, NMS_THRES);

    int row = keep.size();
    vector<vector<float>> nms_boxes(row, vector<float>(BOX_DIM));
    vector<float> nms_scores(row);
    vector<vector<float>> nms_landmarks(row, vector<float>(LANDMARK_DIM));
    for (int i = 0; i < row; i++) {
        int index = keep[i];
        nms_boxes[i] = top_k_boxes[index];
        nms_scores[i] = top_k_scores[index];
        nms_landmarks[i] = top_k_landmarks[index];
    }

    int keep_top_k = min(KEEP_TOP_K, static_cast<int>(nms_scores.size()));
    for (int i = 0; i < keep_top_k; i++) {
        FaceInfo faceInfo;
        faceInfo.score = nms_scores[i];
        faceInfo.center = {(nms_boxes[i][0] + nms_boxes[i][2]) / 2, (nms_boxes[i][1] + nms_boxes[i][3]) / 2};
        faceInfo.width = nms_boxes[i][2] - nms_boxes[i][0];
        faceInfo.height = nms_boxes[i][3] - nms_boxes[i][1];
        for (int j = 0; j < NUM_KEYPOINTS; j++) {
            faceInfo.keypoints.emplace_back(nms_landmarks[i][2 * j], nms_landmarks[i][2 * j + 1]);
        }
        results.push_back(faceInfo);
    }

    return results;
}

int set_input_shape(AILIANetwork *ailia, int tex_width, int tex_height){
    AILIAShape shape;
    shape.x = tex_width;
    shape.y = tex_height;
    shape.z = 3;
    shape.w = 1;
    shape.dim = 4;

    unsigned int input_idx = 0;
    int status = ailiaGetBlobIndexByInputIndex(ailia, &input_idx, 0);
    if (status != AILIA_STATUS_SUCCESS){
        PRINT_ERR("ailiaGetBlobIndexByInputIndex failed %d\n", status);
        return status;
    }

    status = ailiaSetInputBlobShape(ailia, &shape, input_idx, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS){
        PRINT_ERR("ailiaSetInputBlobShape failed %d\n", status);
        return status;
    }

    return AILIA_STATUS_SUCCESS;
}

vector<FaceInfo> detection(AILIANetwork *ailia, std::vector<float> &work, const unsigned char* camera, int tex_width, int tex_height, int channels) {
    vector<FaceInfo> detections;

    // Prepare input data
    work.resize(tex_width * tex_height * 3 * 1);

    for (int y = 0; y < tex_height; y++) {
        for (int x = 0; x < tex_width; x++) {
            int idx = (y * tex_width + x) * channels;
            work[(y * tex_width + x) + 2 * tex_width * tex_height] = static_cast<float>(camera[idx + 2]) - 123.0f; //R
            work[(y * tex_width + x) + 1 * tex_width * tex_height] = static_cast<float>(camera[idx + 1]) - 117.0f; //G
            work[(y * tex_width + x)                             ] = static_cast<float>(camera[idx + 0]) - 104.0f; //B
        }
    }

    // Inference
    unsigned int input_idx = 0;
    int status = ailiaGetBlobIndexByInputIndex(ailia, &input_idx, 0);
    if (status != AILIA_STATUS_SUCCESS){
        PRINT_ERR("ailiaGetBlobIndexByInputIndex failed %d\n", status);
        return detections;
    }

    status = ailiaSetInputBlobData(ailia, &work[0], (unsigned int)(work.size() * sizeof(float)), input_idx);
    if (status != AILIA_STATUS_SUCCESS){
        PRINT_ERR("ailiaSetInputBlobData failed %d\n", status);
        return detections;
    }

    status = ailiaUpdate(ailia);
    if (status != AILIA_STATUS_SUCCESS){
        PRINT_ERR("ailiaUpdate failed %d\n", status);
        return detections;
    }

    AILIAShape box_shape;
    AILIAShape score_shape;
    AILIAShape landmark_shape;

    unsigned int box_idx = 0;
    unsigned int score_idx = 0;
    unsigned int landmark_idx = 0;

    status = ailiaGetBlobIndexByOutputIndex(ailia, &box_idx, 0);
    if (status != AILIA_STATUS_SUCCESS){
        PRINT_ERR("ailiaGetBlobIndexByOutputIndex failed %d\n", status);
        return detections;
    }
    status = ailiaGetBlobIndexByOutputIndex(ailia, &score_idx, 1);
    if (status != AILIA_STATUS_SUCCESS){
        PRINT_ERR("ailiaGetBlobIndexByOutputIndex failed %d\n", status);
        return detections;
    }
    status = ailiaGetBlobIndexByOutputIndex(ailia, &landmark_idx, 2);
    if (status != AILIA_STATUS_SUCCESS){
        PRINT_ERR("ailiaGetBlobIndexByOutputIndex failed %d\n", status);
        return detections;
    }

    status = ailiaGetBlobShape(ailia, &box_shape, box_idx, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS){
        PRINT_ERR("ailiaGetBlobShape failed %d\n", status);
        return detections;
    }
    status = ailiaGetBlobShape(ailia, &score_shape, score_idx, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS){
        PRINT_ERR("ailiaGetBlobShape failed %d\n", status);
        return detections;
    }
    status = ailiaGetBlobShape(ailia, &landmark_shape, landmark_idx, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS){
        PRINT_ERR("ailiaGetBlobShape failed %d\n", status);
        return detections;
    }

    vector<float> box_data(box_shape.x * box_shape.y * box_shape.z * box_shape.w);
    vector<float> score_data(score_shape.x * score_shape.y * score_shape.z * score_shape.w);
    vector<float> landmark_data(landmark_shape.x * landmark_shape.y * landmark_shape.z * landmark_shape.w);

    //PRINT_OUT("box %d %d %d %d\n", box_shape.x, box_shape.y, box_shape.z, box_shape.w);
    //PRINT_OUT("score %d %d %d %d\n", score_shape.x, score_shape.y, score_shape.z, score_shape.w);
    //PRINT_OUT("landmark %d %d %d %d\n", landmark_shape.x, landmark_shape.y, landmark_shape.z, landmark_shape.w);

    status = ailiaGetBlobData(ailia, &box_data[0], (unsigned int)(box_data.size() * sizeof(float)), box_idx);
    if (status != AILIA_STATUS_SUCCESS){
        PRINT_ERR("ailiaGetBlobData failed %d\n", status);
        return detections;
    }
    status = ailiaGetBlobData(ailia, &score_data[0], (unsigned int)(score_data.size() * sizeof(float)), score_idx);
    if (status != AILIA_STATUS_SUCCESS){
        PRINT_ERR("ailiaGetBlobData failed %d\n", status);
        return detections;
    }
    status = ailiaGetBlobData(ailia, &landmark_data[0], (unsigned int)(landmark_data.size() * sizeof(float)), landmark_idx);
    if (status != AILIA_STATUS_SUCCESS){
        PRINT_ERR("ailiaGetBlobData failed %d\n", status);
        return detections;
    }

    // Post-processing
    detections = post_process(box_data, score_data, landmark_data, tex_width, tex_height);

    return detections;
}


int plot_result_retinaface(std::vector<FaceInfo> info, cv::Mat& img, bool logging)
{
    if (logging) {
        PRINT_OUT("object_count=%d\n", (int)info.size());
    }

    for (size_t i = 0; i < info.size(); i++) {
        FaceInfo obj = info[i];
        if (obj.score < VIS_THRES){
            continue;
        }
        if (logging){
            PRINT_OUT("+ idx=%zu\n  score=%.15f\n  x=%.15f\n  y=%.15f\n  w=%.15f\n  h=%.15f\n",
                        i, obj.score, obj.center.first, obj.center.second, obj.width, obj.height);
        }

        cv::Point top_left((int)((obj.center.first - obj.width / 2)), (int)((obj.center.second - obj.height / 2)));
        cv::Point bottom_right((int)((obj.center.first + obj.width / 2)), (int)((obj.center.second + obj.height / 2)));
        cv::Point text_position((int)((obj.center.first - obj.width / 2))+4, (int)((obj.center.second - obj.height / 2)+4));

        // update image
        cv::Scalar color = hsv_to_rgb(256*((float)i/(float)info.size()), 255, 255);
        // unused: float fontScale = (float)img.cols / 512.0f;
        cv::rectangle(img, top_left, bottom_right, color, 4);
    }

    return 0;
}

// ======================
// Main functions
// ======================

static int recognize_from_image(AILIANetwork* ailia)
{
    // prepare input data
    cv::Mat img;
    int status = load_image(img, image_path.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        return status;
    }
    PRINT_OUT("input image shape: (%d, %d, %d)\n",
              img.cols, img.rows, img.channels());
    
    status = set_input_shape(ailia, img.cols, img.rows);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("set_input_shape failed %d\n", status);
        return status;
    }

    // inference
    PRINT_OUT("Start inference...\n");
    std::vector<float> work;
    vector<FaceInfo> results;
    if (benchmark) {
        PRINT_OUT("BENCHMARK mode\n");
        for (int i = 0; i < BENCHMARK_ITERS; i++) {
            clock_t start = clock();
            results = detection(ailia, work, img.data, img.cols, img.rows, img.channels());
            clock_t end = clock();
            if (status != AILIA_STATUS_SUCCESS) {
                PRINT_ERR("ailiaDetectorCompute failed %d\n", status);
                return status;
            }
            PRINT_OUT("\tailia processing time %ld ms\n", ((end-start)*1000)/CLOCKS_PER_SEC);
        }
    }
    else {
        results = detection(ailia, work, img.data, img.cols, img.rows, img.channels());
    }

    status = plot_result_retinaface(results, img, true);
    if (status != AILIA_STATUS_SUCCESS) {
        return status;
    }

    cv::imwrite(save_image_path.c_str(), img);

    PRINT_OUT("Program finished successfully.\n");

    return AILIA_STATUS_SUCCESS;
}


static int recognize_from_video(AILIANetwork* ailia)
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

    int status = set_input_shape(ailia, IMAGE_WIDTH, IMAGE_HEIGHT);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("set_input_shape failed %d\n", status);
        return status;
    }

    std::vector<float> work;

    while (1) {
        cv::Mat frame;
        capture >> frame;
        if ((char)cv::waitKey(1) == 'q' || frame.empty()) {
            break;
        }
        cv::Mat resized_img, img;
        adjust_frame_size(frame, resized_img, IMAGE_WIDTH, IMAGE_HEIGHT);
        cv::cvtColor(resized_img, img, cv::COLOR_BGR2BGRA);

        vector<FaceInfo> results;
        results = detection(ailia, work, img.data, img.cols, img.rows, 4);
        
        status = plot_result_retinaface(results, resized_img, false);
        if (status != AILIA_STATUS_SUCCESS) {
            return status;
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

    static std::string weight(WEIGHT_PATH);
    static std::string model(MODEL_PATH);
    if (mobile){
        weight = WEIGHT_PATH_MOBILE;
        model = MODEL_PATH_MOBILE;
    }

    // net initialize
    AILIANetwork *ailia;
    int env_id = AILIA_ENVIRONMENT_ID_AUTO;
    status = ailiaCreate(&ailia, env_id, AILIA_MULTITHREAD_AUTO);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaCreate failed %d\n", status);
        return -1;
    }

    status = ailiaSetMemoryMode(ailia, AILIA_MEMORY_OPTIMAIZE_DEFAULT | AILIA_MEMORY_REUSE_INTERSTAGE);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaSetMemoryMode failed %d\n", status);
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
    // unused: const unsigned int flags = AILIA_DETECTOR_FLAG_NORMAL;

    if (video_mode) {
        status = recognize_from_video(ailia);
    }
    else {
        status = recognize_from_image(ailia);
    }

    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("recognize failed %d\n", status);
    }

    ailiaDestroy(ailia);

    return status;
}
