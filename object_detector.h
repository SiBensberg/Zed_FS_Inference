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
    explicit ObjectDetector(const std::string& modelPath);
    std::vector<std::vector<float>> inference(const cv::Mat &imageBGR) const;
    bool hwc = true; // whether input to model is HWC or CHW
private:
    // ORT Environment
    Ort::Env mEnv;
    // Session
    mutable Ort::Session mSession = Ort::Session(nullptr);
    // Memory Info
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // Inputs
    char* mInputName;
    std::vector<int64_t> mInputDims; // b x h x w x c
    static inline std::vector<int64_t> mDefaultInputDims = {1, 512, 512, 3};
    // Outputs
    char* mOutputName;
    std::vector<int64_t> mOutputDims; // b x h x w x c

    // Camera input
    mutable std::vector<int64_t> cameraInputDims; // h x w

    void createTensorFromImage(const cv::Mat& img,
                               std::vector<uint8_t>& inputTensorValues) const;

    std::vector<std::vector<float>> calculateBoxes(const Ort::Value &outputTensor) const;
};



#endif //ZED_INFERENCE_OBJECT_DETECTOR_H
