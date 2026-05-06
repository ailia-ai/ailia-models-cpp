#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

#include "mat_utils.h"
#include "image_utils.h"

#if defined(_WIN32) || defined(_WIN64)
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
#else
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif


int adjust_frame_size(const cv::Mat& sframe, cv::Mat& dframe, int d_width, int d_height)
{
    int s_width  = sframe.cols;
    int s_height = sframe.rows;

    float scale = std::max<float>((float)s_width/(float)d_width, (float)s_height/(float)d_height);

    // padding base
    cv::Mat dframe0 = cv::Mat((int)std::round(scale*(float)d_height), (int)std::round(scale*(float)d_width), sframe.type(), cv::Scalar(0, 0, 0, 255));

    int start_x = (dframe0.cols - s_width)  / 2;
    int start_y = (dframe0.rows - s_height) / 2;
    cv::Rect roi(start_x, start_y, s_width, s_height);

    cv::Mat dframe0_roi = dframe0(roi);
    sframe.copyTo(dframe0_roi);

    cv::resize(dframe0, dframe, cv::Size(d_width, d_height), 0, 0);

    return 0;
}


int adjust_frame_size(const cv::Mat& sframe, cv::Mat& dframe0, cv::Mat& dframe, int d_width, int d_height)
{
    int s_width  = sframe.cols;
    int s_height = sframe.rows;

    float scale = std::max<float>((float)s_width/(float)d_width, (float)s_height/(float)d_height);

    // padding base
    dframe0 = cv::Mat((int)std::round(scale*(float)d_height), (int)std::round(scale*(float)d_width), sframe.type(), cv::Scalar(0, 0, 0, 255));

    int start_x = (dframe0.cols - s_width)  / 2;
    int start_y = (dframe0.rows - s_height) / 2;
    cv::Rect roi(start_x, start_y, s_width, s_height);

    cv::Mat dframe0_roi = dframe0(roi);
    sframe.copyTo(dframe0_roi);

    cv::resize(dframe0, dframe, cv::Size(d_width, d_height), 0, 0);

    return 0;
}


int preprocess_frame(const cv::Mat& sframe, cv::Mat& dframe0, cv::Mat& dframe, int d_width, int d_height,
                     bool rgb, std::string normalize_type)
{
    cv::Mat resized_img0;
    adjust_frame_size(sframe, dframe0, resized_img0, d_width, d_height);

    cv::Mat resized_img1;
    if (rgb) {
        cv::cvtColor(resized_img0, resized_img1, cv::COLOR_BGR2RGB);
    }
    else {
        resized_img0.copyTo(resized_img1);
    }

    cv::Mat data;
    normalize_image(resized_img1, data, normalize_type);
    if (rgb) {
        transpose(data, dframe, {2, 0, 1});
    }
    else {
        cv::cvtColor(data, dframe, cv::COLOR_BGR2GRAY);
    }

    return 0;
}


int get_writer(cv::VideoWriter& writer, const char* path, cv::Size size, bool rgb)
{
    const std::string spath = path;
    writer = cv::VideoWriter(spath, cv::VideoWriter::fourcc('M','P','E','G'), 20, size, rgb);

    if (!writer.isOpened()) {
        PRINT_ERR("\'%s\' open failed\n", path);
        return -1;
    }

    return 0;
}
