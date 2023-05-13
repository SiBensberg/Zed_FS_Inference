//
// Created by: https://github.com/SiBensberg
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
//
#include <filesystem>
#include <stdexcept>


class ZedInference {
public:
    ZedInference();
    int run();
    bool visualize = true;
private:
    bool running;
    const ObjectDetector Detector;
    const std::string svo_path = "../misc/zed_raw_jetta.svo";
    sl::InitParameters init_params;
    sl::Mat rgb_image;
    sl::Mat depth_image;

    int num_cameras;
    std::vector<sl::Camera> zeds; // vector with camera instances.
    sl::Resolution image_size;

    void grabRgbImage();
    void visualizeDetections(const cv::Mat& inputImage, const std::vector<std::vector<float>> &bboxes, const std::vector<std::vector<float>> &distances, const std::string &cam);
    std::vector<std::vector<std::vector<float>>> inferenceRgbImage(std::vector<cv::Mat> &rgb_cv_images); // returns boxes. Every vector in the vector is one Box
    std::vector<std::vector<float>> calculateDepth(const std::vector<std::vector<float>>& bboxes, const sl::Mat &point_cloud);

    // Colors for visualisation
    const cv::Scalar BLUE = {180, 128, 0};
    const cv::Scalar YELLOW = {77, 220, 255};
    const cv::Scalar ORANGE = {0, 110, 250};
    const cv::Scalar BIGORANGE = {60, 30, 190};
    const std::vector<cv::Scalar> COLORS = {BLUE, YELLOW, ORANGE, BIGORANGE};
};


#endif //ZED_INFERENCE_ZED_INFERENCE_H
