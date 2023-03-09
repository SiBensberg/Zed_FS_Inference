//
// Created by simon on 04.02.23.
//

#ifndef ZED_INFERENCE_OBJECT_DETECTOR_H
#define ZED_INFERENCE_OBJECT_DETECTOR_H

#include <memory>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

using clock_time = std::chrono::system_clock;
using sec = std::chrono::duration<double>;

class ObjectDetector {
public:
    ObjectDetector(const std::string& modelPath);
    int Inference(cv::Mat imageBGR);
    bool hwc = true; // whether input to model is HWC or CHW
private:
    // ORT Environment
    std::shared_ptr<Ort::Env> mEnv;
    // Session
    std::shared_ptr<Ort::Session> mSession;


    // Inputs
    char* mInputName;
    std::vector<int64_t> mInputDims; // b x h x w x c
    // Outputs
    char* mOutputName;
    std::vector<int64_t> mOutputDims; // b x h x w x c
    std::vector<int64_t> cameraInputDims; // h x w


    void CreateTensorFromImage(const cv::Mat& img,
                               std::vector<uint8_t>& inputTensorValues);

    void CreateInferenceImage(Ort::Value *outputTensor, cv::Mat inputImage);
};



#endif //ZED_INFERENCE_OBJECT_DETECTOR_H
