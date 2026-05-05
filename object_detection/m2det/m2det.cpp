/*******************************************************************
*
*    DESCRIPTION:
*      AILIA m2det sample
*    AUTHOR:
*
*    DATE:2020/08/11
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


// ======================
// Parameters
// ======================

#define WEIGHT_PATH "m2det.onnx"
#define MODEL_PATH  "m2det.onnx.prototxt"

#define IMAGE_PATH      "couple.jpg"
#define SAVE_IMAGE_PATH "output.png"

#define THRESHOLD      0.4f
#define IOU            0.45f
#define KEEP_PER_CLASS 10

#if defined(_WIN32) || defined(_WIN64)
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
#else
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif

#define BENCHMARK_ITERS 5

typedef struct {
    unsigned int in;
    unsigned int out0;
    unsigned int out1;
} ioIndices;

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
} box;

static std::string weight(WEIGHT_PATH);
static std::string model(MODEL_PATH);

static std::string image_path(IMAGE_PATH);
static std::string video_path("0");
static std::string save_image_path(SAVE_IMAGE_PATH);

static const std::vector<const char*> COCO_CATEGORY = {
    "__background__",
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
    PRINT_OUT("usage: m2det [-h] [-i IMAGE] [-v VIDEO] [-s SAVE_IMAGE_PATH] [-b]\n");
    return;
}


static void print_help()
{
    PRINT_OUT("\n");
    PRINT_OUT("m2det model\n");
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
    PRINT_ERR("m2det: error: unrecognized arguments: %s\n", arg.c_str());
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
// Secondary functions
// ======================

static void nms(std::vector<float>& c_scores, std::vector<box>& c_boxes,
                std::vector<int>& keep, float thresh)
{
    // calculate area of the boxes
    std::vector<float> areas(c_scores.size());
    for (size_t i = 0; i < c_scores.size(); i++) {
        areas[i] = (c_boxes[i].x2-c_boxes[i].x1+1.0f)*(c_boxes[i].y2-c_boxes[i].y1+1.0f);
    }

    // get sorted indices of the score
    cv::Mat mat_scores = cv::Mat_<float>(1, c_scores.size(), &c_scores[0]);
    cv::Mat mat_order  = cv::Mat_<int>(1, c_scores.size());
    cv::sortIdx(mat_scores, mat_order, cv::SORT_EVERY_ROW|cv::SORT_DESCENDING);

    std::vector<int> order; 
    order.insert(order.end(), (int*)mat_order.data, (int*)mat_order.data+mat_order.cols);

    while (order.size() > 0) {
        std::vector<int> inds;
        int i = order[0];
        keep.push_back(i);
        for (size_t j = 1; j < order.size(); j++) {
            int k = order[j];
            float xx1 = std::max(c_boxes[i].x1, c_boxes[k].x1);
            float yy1 = std::max(c_boxes[i].y1, c_boxes[k].y1);
            float xx2 = std::min(c_boxes[i].x2, c_boxes[k].x2);
            float yy2 = std::min(c_boxes[i].y2, c_boxes[k].y2);

            float w = std::max(0.0f, xx2 - xx1 + 1.0f);
            float h = std::max(0.0f, yy2 - yy1 + 1.0f);
            float inter = w * h;
            float ovr = inter / (areas[i] + areas[k] - inter);
            if (ovr <= thresh) {
                inds.push_back(order[j]);
            }
        }

        order = inds;
    }

    return;
}


static void preprocess(const cv::Mat& simg, cv::Mat& dimg, int resize = 512,
                       std::vector<float> rgb_means = {104, 117, 123},
                       std::vector<int> swap = {2, 0, 1})
{
    cv::Mat mimg;
    cv::resize(simg, mimg, cv::Size(resize, resize), 0, 0);

    std::vector<int> size0 = {mimg.rows, mimg.cols, mimg.channels()};
    std::vector<int> size1 = {size0[swap[0]], size0[swap[1]], size0[swap[2]]};
    dimg = cv::Mat_<float>(size1.size(), &size1[0]); // 3D array

    unsigned char* mdata = (unsigned char*)mimg.data;
    float*         ddata = (float*)dimg.data;
    int sd[3] = {0, 0, 0};
    for (int d0 = 0; d0 < size1[0]; d0++) {
        sd[swap[0]] = d0;
        for (int d1 = 0; d1 < size1[1] ; d1++) {
            sd[swap[1]] = d1;
            for (int d2 = 0; d2 < size1[2]; d2++) {
                sd[swap[2]] = d2;
                float col = mdata[sd[0]*size0[1]*size0[2]+sd[1]*size0[2]+sd[2]];
                ddata[d0*size1[1]*size1[2]+d1*size1[2]+d2] = col - rgb_means[sd[2]];
            }
        }
    }

    return;
}


static std::vector<cv::Scalar> COLORS;

static void gen_colors()
{
    int base = std::ceil(std::pow(COCO_CATEGORY.size(), 1.0f/3.0f));
    int base2 = base * base;

    for (size_t indx = 0; indx < COCO_CATEGORY.size(); indx++) {
        float b = 2.0f - (float)indx / (float)base2;
        float g = 2.0f - (float)(indx % base2) / (float)base;
        float r = 2.0f - (indx % base2) % base;
        COLORS.push_back(cv::Scalar(b*127.0f, g*127.0f, r*127.0f));
    }
}


static void draw_detection(const cv::Mat& img, cv::Mat& imgcv,
                           std::vector<box> boxes,
                           std::vector<float> scores,
                           std::vector<int> cls_inds)
{
   imgcv = img.clone(); 
   int imgw = img.cols;
   int imgh = img.rows;

   int thick = (float)(imgw+imgh) / 300.0f;

   for (size_t i = 0; i < boxes.size(); i++) {
       int cls_indx = cls_inds[i];
       cv::rectangle(imgcv,
                     cv::Point(boxes[i].x1, boxes[i].y1),
                     cv::Point(boxes[i].x2, boxes[i].y2),
                     COLORS[cls_indx], thick);
       char str[30];
       snprintf(str, sizeof(str), "%s: %.3f", COCO_CATEGORY[cls_indx], scores[i]);
       std::string mess(str);
       cv::putText(imgcv, mess,
                   cv::Point(boxes[i].x1, boxes[i].y1-7),
                   0, imgh * 1e-3, COLORS[cls_indx], thick);
   }
}


// ======================
// Main functions
// ======================

static int detect_objects(const cv::Mat& img, AILIANetwork *detector, ioIndices io_inds,
                          std::vector<box>& boxes, std::vector<float>& scores, std::vector<int>& cls_inds)
{
    int status;

    // get input/output shape
    AILIAShape in_shape;
    status = ailiaGetBlobShape(detector, &in_shape, io_inds.in, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetBlobShape failed %d\n", status);
        return -1;
    }

    AILIAShape out0_shape;
    status = ailiaGetBlobShape(detector, &out0_shape, io_inds.out0, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetBlobShape failed %d\n", status);
        return -1;
    }

    AILIAShape out1_shape;
    status = ailiaGetBlobShape(detector, &out1_shape, io_inds.out1, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetBlobShape failed %d\n", status);
        return -1;
    }

    // get sizes for posterior rescaling
    int imgw = img.cols;
    int imgh = img.rows;

    // initial process for source image
    cv::Mat src;
    preprocess(img, src, in_shape.y);

    float* srcp = (float*)src.data;
    status = ailiaSetInputBlobData(detector, srcp,
                                   in_shape.x*in_shape.y*in_shape.z*sizeof(float),
                                   io_inds.in);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaSetInputBlobData failed %d\n", status);
        return -1;
    }

    // feedforward
    std::vector<float> dst0(out0_shape.x*out0_shape.y*out0_shape.z*out0_shape.w);
    std::vector<float> dst1(out1_shape.x*out1_shape.y*out1_shape.z*out1_shape.w);

    status = ailiaUpdate(detector);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaUpdate failed %d\n", status);
        return -1;
    }

    status = ailiaGetBlobData(detector, &dst0[0], dst0.size()*sizeof(float), io_inds.out0);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetBlobData failed %d\n", status);
        return -1;
    }

    status = ailiaGetBlobData(detector, &dst1[0], dst1.size()*sizeof(float), io_inds.out1);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetBlobData failed %d\n", status);
        return -1;
    }

    // filter boxes for every class
    for (size_t cls = 1; cls < out1_shape.x; cls++) {
        std::vector<float> c_scores;
        std::vector<box>   c_boxes;
        for (size_t obj = 0; obj < out1_shape.y; obj++) {
            float score = dst1[obj*out1_shape.x+cls];
            if (score >= THRESHOLD) {
                box c_box;
                c_box.x1 = dst0[obj*out0_shape.x+0];
                c_box.y1 = dst0[obj*out0_shape.x+1];
                c_box.x2 = dst0[obj*out0_shape.x+2];
                c_box.y2 = dst0[obj*out0_shape.x+3];
                c_scores.push_back(score);
                c_boxes.push_back(c_box);
            }
        }
        if (c_scores.empty()) {
            continue;
        }

        std::vector<int> keep;
        nms(c_scores, c_boxes, keep, IOU);
        if (keep.size() > KEEP_PER_CLASS) {
            keep.resize(KEEP_PER_CLASS);
        }

        for (size_t i = 0; i < keep.size(); i++) {
            int j = keep[i];
            box c_box;
            c_box.x1 = c_boxes[j].x1*imgw;
            c_box.y1 = c_boxes[j].y1*imgh;
            c_box.x2 = c_boxes[j].x2*imgw;
            c_box.y2 = c_boxes[j].y2*imgh;
            boxes.push_back(c_box);
            scores.push_back(c_scores[j]);
            cls_inds.push_back(cls);
        }
    }

    return AILIA_STATUS_SUCCESS;
}


static int recognize_from_image(AILIANetwork *detector, ioIndices io_inds)
{
    // load image
    cv::Mat img = cv::imread(image_path.c_str(), -1);
    if (img.empty()) {
        PRINT_ERR("\'%s\' image not found\n", image_path.c_str());
        return -1;
    }
//    PRINT_OUT("input image shape: (%d, %d, %d)\n",
//              img.cols, img.rows, img.channels());

    // inference
    int status;
    std::vector<box>   boxes;
    std::vector<float> scores;
    std::vector<int>   cls_inds;
    PRINT_OUT("Start inference...\n");
    if (benchmark) {
        PRINT_OUT("BENCHMARK mode\n");
        for (int i = 0; i < BENCHMARK_ITERS; i++) {
            boxes.clear();
            scores.clear();
            cls_inds.clear();
            clock_t start = clock();
            status = detect_objects(img, detector, io_inds, boxes, scores, cls_inds);
            clock_t end = clock();
            if (status != AILIA_STATUS_SUCCESS) {
                return -1;
            }
            PRINT_OUT("\tailia processing time %ld ms\n", ((end-start)*1000)/CLOCKS_PER_SEC);
        }
    }
    else {
        status = detect_objects(img, detector, io_inds, boxes, scores, cls_inds);
        if (status != AILIA_STATUS_SUCCESS) {
            return -1;
        }
    }

    for (size_t i = 0; i < boxes.size(); i++) {
        PRINT_OUT("pos:(%.1f,%.1f,%.1f,%.1f), ids:%s, score:%.3f\n",
                  boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2,
                  COCO_CATEGORY[cls_inds[i]], scores[i]);
    }

    cv::Mat im2show;
    draw_detection(img, im2show, boxes, scores, cls_inds);
    cv::imwrite(save_image_path.c_str(), im2show);

    PRINT_OUT("Program finished successfully.\n");

    return AILIA_STATUS_SUCCESS;
}


static int recognize_from_video(AILIANetwork *detector, ioIndices io_inds)
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
        int status;
        std::vector<box>   boxes;
        std::vector<float> scores;
        std::vector<int>   cls_inds;

        cv::Mat frame;
        capture >> frame;
        if ((char)cv::waitKey(1) == 'q' || frame.empty()) {
            break;
        }

        status = detect_objects(frame, detector, io_inds, boxes, scores, cls_inds);
        if (status != AILIA_STATUS_SUCCESS) {
            return -1;
        }

        cv::Mat frame2show;
        draw_detection(frame, frame2show, boxes, scores, cls_inds);

        cv::imshow("frame", frame2show);
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
    AILIANetwork *detector;
    int env_id = AILIA_ENVIRONMENT_ID_AUTO;
    status = ailiaCreate(&detector, env_id, AILIA_MULTITHREAD_AUTO);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaCreate failed %d\n", status);
        return -1;
    }

    AILIAEnvironment *env_ptr = nullptr;
    status = ailiaGetSelectedEnvironment(detector, &env_ptr, AILIA_ENVIRONMENT_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetSelectedEnvironment failed %d\n", status);
        ailiaDestroy(detector);
        return -1;
    }

//    PRINT_OUT("env_id: %d\n", env_ptr->id);
    PRINT_OUT("env_name: %s\n", env_ptr->name);

    status = ailiaOpenStreamFile(detector, model.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaOpenStreamFile failed %d\n", status);
        PRINT_ERR("ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(detector));
        ailiaDestroy(detector);
        return -1;
    }

    status = ailiaOpenWeightFile(detector, weight.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
        ailiaDestroy(detector);
        return -1;
    }

    ioIndices io_inds = {0, 0, 0};
    status = ailiaFindBlobIndexByName(detector, &io_inds.in, "input.1");
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaFindBlobIndexByName failed %d\n", status);
        return -1;
    }

    status = ailiaFindBlobIndexByName(detector, &io_inds.out0, "2369");
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaFindBlobIndexByName failed %d\n", status);
        return -1;
    }

    status = ailiaFindBlobIndexByName(detector, &io_inds.out1, "2370");
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaFindBlobIndexByName failed %d\n", status);
        return -1;
    }

    gen_colors();

    if (video_mode) {
        recognize_from_video(detector, io_inds);
    }
    else {
        recognize_from_image(detector, io_inds);
    }

    ailiaDestroy(detector);

    return 0;
}
