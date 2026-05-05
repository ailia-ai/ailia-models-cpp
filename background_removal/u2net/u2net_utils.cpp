#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "mat_utils.h"

#if defined(_WIN32) || defined(_WIN64)
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
#else
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif


void transform(const cv::Mat& simg, cv::Mat& dimg, cv::Size scaled_size)
{
    cv::Mat mimg0;
    cv::resize(simg, mimg0, scaled_size, 0, 0);

    int w = mimg0.cols;
    int h = mimg0.rows;
    cv::Mat mimg1(h, w, CV_32FC3);
    unsigned char* data0 = (unsigned char*)mimg0.data;
    float*         data1 = (float*)mimg1.data;
    if (mimg0.channels() == 1) {
        for (int i = 0; i < w*h; i++) {
            float col = data0[i];
            col = (col / 255.0f - 0.485f) / 0.229f;
            data1[i*3+0] = data1[i*3+1] = data1[i*3+2] = col;
        }
    }
    else {
        float mean[] = {0.485f, 0.456f, 0.406f};
        float std[]  = {0.229f, 0.224f, 0.225f};
        for (int i = 0; i < w*h; i++) {
            for (int c = 0; c < 3; c++) {
                float col = data0[i*3+c];
                col = (col / 255.0f - mean[c]) / std[c];
                data1[i*3+c] = col;
            }
        }
    }

    transpose(mimg1, dimg, {2, 0, 1});

    return;
}


int load_image(cv::Mat& image, cv::Size& src_size, const char* path, cv::Size scaled_size)
{
    cv::Mat oimg = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (oimg.empty()) {
        PRINT_ERR("\'%s\' not found\n", path);
        return -1;
    }

    src_size = cv::Size(oimg.cols, oimg.rows);
    transform(oimg, image, scaled_size);

    return 0;
}


static void normalize(const cv::Mat& simg, cv::Mat& dimg)
{
    float min = FLT_MAX;
    float max = FLT_MIN;
    float* sdata = (float*)simg.data;
    int size = simg.rows*simg.cols*simg.channels();

    for (int i = 0; i < size; i++) {
        min = std::min(min, sdata[i]);
        max = std::max(max, sdata[i]);
    }

    dimg = cv::Mat(simg.rows, simg.cols, CV_8UC1);
    unsigned char* ddata = (unsigned char*)dimg.data;
    for (int i = 0; i < size; i++) {
        ddata[i] = (unsigned char)(((sdata[i]-min) / (max-min))*255.0f);
    }

    return;
}


int save_result(const cv::Mat& pred, const char* path, cv::Size src_size)
{
    cv::Mat norm, outimg;
    normalize(pred, norm);
    cv::resize(norm, outimg, src_size, 0, 0);

    cv::imwrite(path, outimg);

    return 0;
}
