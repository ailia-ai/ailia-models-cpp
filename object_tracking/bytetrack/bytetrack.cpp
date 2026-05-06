/*******************************************************************
 *
 *    DESCRIPTION:
 *      ailia tracker (ByteTrack) sample
 *    AUTHOR:
 *      ailia Inc.
 *    DATE:2026/05/05
 *
 *******************************************************************/

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <vector>
#include <map>

#undef UNICODE

#include "ailia.h"
#include "ailia_detector.h"
#include "ailia_tracker.h"
#include "detector_utils.h"
#include "utils.h"
#include "webcamera_utils.h"

// ======================
// Parameters
// ======================

#define WEIGHT_PATH "yolox_s.opt.onnx"
#define MODEL_PATH "yolox_s.opt.onnx.prototxt"

#define MODEL_INPUT_WIDTH 640
#define MODEL_INPUT_HEIGHT 640
#define IMAGE_WIDTH 640  // for video mode
#define IMAGE_HEIGHT 640 // for video mode

#define TARGET_CATEGORY 0   // person
#define THRESHOLD 0.4f
#define IOU 0.45f
#define IS_VERTICAL_THRESHOLD 1.6

#define RECTANGLE_BORDER_SIZE 2
#define TEXT_COLOR cv::Scalar(0, 255, 0)
#define TEXT_SIZE 1.0
#define TEXT_BORDER_SIZE 1
#define TEXT_FONT cv::FONT_HERSHEY_SIMPLEX

#if defined(_WIN32) || defined(_WIN64)
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
#else
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif

static std::string weight(WEIGHT_PATH);
static std::string model(MODEL_PATH);

static bool useWebCamera(false);
static bool saveOutputVideo(false);
static std::string inputVideoPath;
static std::string outputVideoPath;

static const std::vector<const char *> COCO_CATEGORY = {
    "person",        "bicycle",      "car",
    "motorcycle",    "airplane",     "bus",
    "train",         "truck",        "boat",
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",        "bird",
    "cat",           "dog",          "horse",
    "sheep",         "cow",          "elephant",
    "bear",          "zebra",        "giraffe",
    "backpack",      "umbrella",     "handbag",
    "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports ball",
    "kite",          "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket",
    "bottle",        "wine glass",   "cup",
    "fork",          "knife",        "spoon",
    "bowl",          "banana",       "apple",
    "sandwich",      "orange",       "broccoli",
    "carrot",        "hot dog",      "pizza",
    "donut",         "cake",         "chair",
    "couch",         "potted plant", "bed",
    "dining table",  "toilet",       "tv",
    "laptop",        "mouse",        "remote",
    "keyboard",      "cell phone",   "microwave",
    "oven",          "toaster",      "sink",
    "refrigerator",  "book",         "clock",
    "vase",          "scissors",     "teddy bear",
    "hair drier",    "toothbrush"};

static int args_env_id = -1;

// ======================
// Argument Parser
// ======================

static void print_usage() {
    PRINT_OUT("usage: bytetrack [-h] VIDEO [-w WEB_CAMERA] [-o OUTPUT_VIDEO_PATH] "
              "[-e ENV_ID]\n");
    return;
}

static void print_help() {
    PRINT_OUT("\n");
    PRINT_OUT("bytetrack model\n");
    PRINT_OUT("\n");
    PRINT_OUT("arguments:\n");
    PRINT_OUT("  VIDEO                 The input video path\n");
    PRINT_OUT("\n");
    PRINT_OUT("optional arguments:\n");
    PRINT_OUT("  -h, --help            show this help message and exit\n");
    PRINT_OUT("  -w WEB_CAMERA, --use_web_camera\n");
    PRINT_OUT("                        use web camera\n");
    PRINT_OUT("  -o OUTPUT_VIDEO_PATH, --output_video_path OUTPUT_VIDEO_PATH\n");
    PRINT_OUT("                        Save path for the output video (video format must be MP4).\n");
    PRINT_OUT("  -e ENV_ID, --env_id ENV_ID\n");
    PRINT_OUT("                        The backend environment id.\n");
    return;
}

static void print_error(std::string arg) {
    PRINT_ERR("bytetrack: error: unrecognized arguments: %s\n", arg.c_str());
    return;
}

static int argument_parser(int argc, char **argv) {
    int status = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (i == 1) {
            if (arg == "-h" || arg == "--help") {
                print_usage();
                print_help();
                return -1;
            }
            inputVideoPath = arg;
            continue;
        }
        if (status == 0) {
            if (arg == "-o" || arg == "--output_video_path") {
                status = 1;
                saveOutputVideo = true;
            } else if (arg == "-e" || arg == "--env_id") {
                status = 2;
            } else if (arg == "-w" || arg == "--use_web_camera") {
                useWebCamera = true;
            } else {
                print_usage();
                print_error(arg);
                return -1;
            }
        } else if (arg[0] != '-') {
            switch (status) {
            case 1: outputVideoPath = arg; break;
            case 2: args_env_id = atoi(arg.c_str()); break;
            default:
                print_usage();
                print_error(arg);
                return -1;
            }
            status = 0;
        } else {
            print_usage();
            print_error(arg);
            return -1;
        }
    }

    if (inputVideoPath.empty()) {
        print_usage();
        PRINT_ERR("bytetrack: error: VIDEO argument is required\n");
        return -1;
    }

    return AILIA_STATUS_SUCCESS;
}

static int recognize_from_video(AILIADetector *detector) {
    srand((unsigned)time(NULL));
    int status;
    AILIATracker *ailiaTracker = nullptr;
    AILIATrackerSettings settings;
    settings.score_threshold = 0.1f;
    settings.nms_threshold = 0.7f;
    settings.track_threshold = 0.5f;
    settings.track_buffer = 30;
    settings.match_threshold = 0.8f;
    status = ailiaTrackerCreate(&ailiaTracker,
                                AILIA_TRACKER_ALGORITHM_BYTE_TRACK, &settings,
                                AILIA_TRACKER_SETTINGS_VERSION, AILIA_TRACKER_FLAG_NONE);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaTrackerCreate failed %d\n", status);
        return -1;
    }

    cv::VideoCapture capture;
    if (useWebCamera) {
        PRINT_OUT("[INFO] webcamera mode is activated\n");
        capture = cv::VideoCapture(atoi(inputVideoPath.c_str()));
        if (!capture.isOpened()) {
            PRINT_ERR("[ERROR] webcamera not found\n");
            ailiaTrackerDestroy(ailiaTracker);
            return -1;
        }
    } else {
        if (check_file_existance(inputVideoPath.c_str())) {
            capture = cv::VideoCapture(inputVideoPath.c_str());
        } else {
            PRINT_ERR("[ERROR] \"%s\" not found\n", inputVideoPath.c_str());
            ailiaTrackerDestroy(ailiaTracker);
            return -1;
        }
    }

    cv::VideoWriter writer;
    if (saveOutputVideo) {
        int fourcc = cv::VideoWriter::fourcc('M', 'P', '4', 'V');
        writer = cv::VideoWriter(
            outputVideoPath.c_str(), fourcc, capture.get(cv::CAP_PROP_FPS),
            cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
    }

    std::map<unsigned int, cv::Scalar> id2Color;

    while (1) {
        cv::Mat frame, resized_img, img;
        capture >> frame;
        if ((char)cv::waitKey(1) == 'q' || frame.empty()) {
            break;
        }
        adjust_frame_size(frame, resized_img, IMAGE_WIDTH, IMAGE_HEIGHT);
        cv::cvtColor(resized_img, img, cv::COLOR_BGR2BGRA);
        status = ailiaDetectorCompute(detector, img.data, MODEL_INPUT_WIDTH * 4,
                                      MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT,
                                      AILIA_IMAGE_FORMAT_BGRA, THRESHOLD, IOU);

        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaDetectorCompute failed %d\n", status);
            ailiaTrackerDestroy(ailiaTracker);
            return -1;
        }

        unsigned int objCounts;
        ailiaDetectorGetObjectCount(detector, &objCounts);
        AILIADetectorObject *ailiaDetectorObject = new AILIADetectorObject[objCounts];
        for (unsigned int i = 0; i < objCounts; i++) {
            status = ailiaDetectorGetObject(detector, &ailiaDetectorObject[i],
                                            i, AILIA_DETECTOR_OBJECT_VERSION);
            if (status != AILIA_STATUS_SUCCESS) {
                PRINT_ERR("ailiaDetectorGetObject failed %d\n", status);
                delete[] ailiaDetectorObject;
                ailiaTrackerDestroy(ailiaTracker);
                return -1;
            }
        }
        for (unsigned int i = 0; i < objCounts; i++) {
            status = ailiaTrackerAddTarget(ailiaTracker, &ailiaDetectorObject[i],
                                           AILIA_DETECTOR_OBJECT_VERSION);
            if (status != AILIA_STATUS_SUCCESS) {
                PRINT_ERR("ailiaTrackerAddTarget failed %d\n", status);
                delete[] ailiaDetectorObject;
                ailiaTrackerDestroy(ailiaTracker);
                return -1;
            }
        }
        delete[] ailiaDetectorObject;
        status = ailiaTrackerCompute(ailiaTracker);

        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaTrackerCompute failed %d\n", status);
            ailiaTrackerDestroy(ailiaTracker);
            return -1;
        }

        unsigned int onlineSize;
        status = ailiaTrackerGetObjectCount(ailiaTracker, &onlineSize);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaTrackerGetObjectCount failed %d\n", status);
            ailiaTrackerDestroy(ailiaTracker);
            return -1;
        }
        AILIATrackerObject *ailiaTrackerObject = new AILIATrackerObject[onlineSize];

        for (unsigned int i = 0; i < onlineSize; i++) {
            status = ailiaTrackerGetObject(ailiaTracker, &ailiaTrackerObject[i],
                                           i, AILIA_TRACKER_OBJECT_VERSION);
            if (status != AILIA_STATUS_SUCCESS) {
                PRINT_ERR("ailiaTrackerGetObject failed %d\n", status);
                delete[] ailiaTrackerObject;
                ailiaTrackerDestroy(ailiaTracker);
                return -1;
            }
        }

        cv::Point leftUpperPoint, rightBottomPoint;
        cv::Scalar color;
        for (unsigned int i = 0; i < onlineSize; i++) {
            AILIATrackerObject obj = ailiaTrackerObject[i];

            if (obj.category != TARGET_CATEGORY) {
                continue;
            }
            const unsigned int id = obj.id;
            if (id2Color.find(id) != id2Color.end()) {
                color = id2Color[id];
            } else {
                int b = rand() % 256;
                int g = rand() % 256;
                int r = rand() % 256;
                color = cv::Scalar(b, g, r);
                id2Color.insert(std::make_pair(id, color));
            }
            const unsigned int x = static_cast<unsigned int>(obj.x * IMAGE_WIDTH);
            const unsigned int y = static_cast<unsigned int>(obj.y * IMAGE_HEIGHT);
            const unsigned int width = static_cast<unsigned int>(obj.w * IMAGE_WIDTH);
            const unsigned int height = static_cast<unsigned int>(obj.h * IMAGE_HEIGHT);
            leftUpperPoint = cv::Point(x, y);
            rightBottomPoint = cv::Point(x + width, y + height);
            cv::rectangle(resized_img, leftUpperPoint, rightBottomPoint, color,
                          RECTANGLE_BORDER_SIZE);
            cv::putText(resized_img, std::to_string(obj.id), leftUpperPoint,
                        TEXT_FONT, TEXT_SIZE, TEXT_COLOR, TEXT_BORDER_SIZE);
        }
        delete[] ailiaTrackerObject;
        cv::imshow("result frame", resized_img);
        if (saveOutputVideo) {
            writer.write(resized_img);
        }
    }

    capture.release();
    if (saveOutputVideo) {
        writer.release();
    }
    cv::destroyAllWindows();
    ailiaTrackerDestroy(ailiaTracker);

    PRINT_OUT("Program finished successfully.\n");

    return AILIA_STATUS_SUCCESS;
}

int main(int argc, char **argv) {
    int status = argument_parser(argc, argv);
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }

    // env list
    unsigned int env_count;
    status = ailiaGetEnvironmentCount(&env_count);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetEnvironmentCount Failed %d", status);
        return -1;
    }

    int env_id = AILIA_ENVIRONMENT_ID_AUTO;
    for (unsigned int i = 0; i < env_count; i++) {
        AILIAEnvironment *env;
        status = ailiaGetEnvironment(&env, i, AILIA_ENVIRONMENT_VERSION);
        PRINT_OUT("env_id : %d type : %d name : %s\n", env->id, env->type,
                  env->name);
        if (args_env_id == env->id) {
            env_id = env->id;
        }
        if (args_env_id == -1 && env_id == AILIA_ENVIRONMENT_ID_AUTO) {
            if (env->type == AILIA_ENVIRONMENT_TYPE_GPU) {
                env_id = env->id;
            }
        }
    }
    if (args_env_id == -1) {
        PRINT_OUT("you can select environment using -e option\n");
    }

    // net initialize
    AILIANetwork *ailia;
    status = ailiaCreate(&ailia, env_id, AILIA_MULTITHREAD_AUTO);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaCreate failed %d\n", status);
        if (status == AILIA_STATUS_LICENSE_NOT_FOUND ||
            status == AILIA_STATUS_LICENSE_EXPIRED) {
            PRINT_OUT("License file not found or expired : please place "
                      "license file (AILIA.lic)\n");
        }
        return -1;
    }

    AILIAEnvironment *env_ptr = nullptr;
    status =
        ailiaGetSelectedEnvironment(ailia, &env_ptr, AILIA_ENVIRONMENT_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetSelectedEnvironment failed %d\n", status);
        ailiaDestroy(ailia);
        return -1;
    }

    PRINT_OUT("env_id: %d\n", env_ptr->id);
    PRINT_OUT("selected env name : %s\n", env_ptr->name);

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
    status = ailiaCreateDetector(
        &detector, ailia, AILIA_NETWORK_IMAGE_FORMAT_BGR,
        AILIA_NETWORK_IMAGE_CHANNEL_FIRST,
        AILIA_NETWORK_IMAGE_RANGE_UNSIGNED_INT8, AILIA_DETECTOR_ALGORITHM_YOLOX,
        (unsigned int)COCO_CATEGORY.size(), flags);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaCreateDetector failed %d\n", status);
        ailiaDestroy(ailia);
        return -1;
    }

    status = ailiaDetectorSetInputShape(detector, MODEL_INPUT_WIDTH,
                                        MODEL_INPUT_HEIGHT);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaDetectorSetInputShape(w=%u, h=%u) failed %d\n",
                  MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, status);
        ailiaDestroyDetector(detector);
        ailiaDestroy(ailia);
        return -1;
    }

    status = recognize_from_video(detector);
    ailiaDestroyDetector(detector);
    ailiaDestroy(ailia);
    return status;
}
