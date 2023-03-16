//
// Created by simon on 28.01.23.
//

#ifndef ZED_INFERENCE_ZED_INFERENCE_H
#define ZED_INFERENCE_ZED_INFERENCE_H

// OpenCV includes
#include <opencv2/opencv.hpp>
// OpenCV dep
#include <opencv2/cvconfig.h>
// ZED
#include <sl/Camera.hpp>
// Object Detector
#include "object_detector.h"

class ZedInference {
public:
    int run();
    ZedInference();
    bool visualize = true;
private:
    void grab_rgb_image();
    std::vector<std::vector<float>> inference_rgb_image(cv::Mat rgb_image); // returns boxes. Every vector in the vector is one Box
    std::vector<std::vector<float>> calculateDepth(const std::vector<std::vector<float>>& bboxes, sl::Mat& point_cloud);
    void visualizeDetections(const cv::Mat& inputImage, std::vector<std::vector<float>> bboxes, std::vector<std::vector<float>> distances);
    bool running;
    sl::InitParameters init_params;
    sl::Mat rgb_image;
    sl::Mat depth_image;
    sl::Camera zed;
    ObjectDetector Detector{"/home/simon/LionsRacing_workspace/Zed_FS_Inference/model/saved_model.onnx"};
    const char* svo_path = "/home/simon/CLionProjects/test/misc/zed_raw_jetta.svo";

    const cv::Scalar BLUE = {180, 128, 0};
    const cv::Scalar YELLOW = {77, 220, 255};
    const cv::Scalar ORANGE = {0, 110, 250};
    const cv::Scalar BIGORANGE = {60, 30, 190};
    const std::vector<cv::Scalar> COLORS = {BLUE, YELLOW, ORANGE, BIGORANGE};
};


#endif //ZED_INFERENCE_ZED_INFERENCE_H
