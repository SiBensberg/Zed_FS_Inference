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
    std::vector<std::vector<float>> Inference(const cv::Mat& imageBGR);
    bool hwc = true; // whether input to model is HWC or CHW
private:
    // ORT Environment
    //std::shared_ptr<Ort::Env> mEnv;
    Ort::Env mEnv;
    // Session
    //std::shared_ptr<Ort::Session> mSession;
    Ort::Session mSession = Ort::Session(nullptr);

    // Inputs
    char* mInputName;
    std::vector<int64_t> mInputDims; // b x h x w x c
    // Outputs
    char* mOutputName;
    std::vector<int64_t> mOutputDims; // b x h x w x c
    std::vector<int64_t> cameraInputDims; // h x w

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    void CreateTensorFromImage(const cv::Mat& img,
                               std::vector<uint8_t>& inputTensorValues);

    std::vector<std::vector<float>> CreateInferenceImage(Ort::Value *outputTensor);
};



#endif //ZED_INFERENCE_OBJECT_DETECTOR_H
