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
private:
    void grab_rgb_image();
    void grab_depth_image();
    void inference_rgb_image(cv::Mat rgb_image);
    bool running;
    sl::Mat rgb_image;
    sl::Mat depth_image;
    sl::Camera zed;
    ObjectDetector Detector{"/home/simon/CLionProjects/test/model/fsoco_efficientdet_d0.onnx"};
    const char* svo_path = "/home/simon/CLionProjects/test/misc/zed_raw_jetta.svo";
    //const std::string svo_path1 = "/home/simon/CLionProjects/test/misc/zed_raw_jetta.svo";
};


#endif //ZED_INFERENCE_ZED_INFERENCE_H
