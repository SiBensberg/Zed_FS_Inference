//
// Created by simon on 04.02.23.
//
#include "object_detector.h"



ObjectDetector::ObjectDetector(const std::string &modelPath) {
    std::cout << "Initiating ObjectDetector: " << std::endl;
    // Create Environment:
    std::string instance_Name{"Object Detector"};
    // make shared for creating new shared pointer
    mEnv = std::make_shared<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instance_Name.c_str());

    // Ort Session
    Ort::SessionOptions sessionOptions;
    // Enable Cuda
    sessionOptions.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // Load model
    mSession = std::make_shared<Ort::Session>(*mEnv, modelPath.c_str(), sessionOptions);

    // Allocator
    Ort::AllocatorWithDefaultOptions allocator;

    // Extract input info:
    size_t numInputNodes = mSession->GetInputCount();
    //mInputName = mSession->GetInputName(0, allocator);
    mInputName = "image_arrays:0";

    // Input type:
    Ort::TypeInfo inputTypeInfo = mSession->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();

    mInputDims = inputTensorInfo.GetShape();
    // Check if input dim is variable:
    bool variabel = false;
    for (int64_t i: mInputDims){
        if (i <= 0){
            variabel = true;
        }
    }
    if (variabel){
        std::cout << "Input size of exported ONNX model is variable. For this reason it has to be predefined." << std::endl << "Setting it to 512x512" << std::endl;
        mInputDims = {1, 3, 512, 512};
    }


    size_t numOutputNodes = mSession->GetOutputCount();
    //mOutputName = mSession -> GetOutputNameAllocated(0, allocator);
    mOutputName = "detections:0";

    Ort::TypeInfo outputTypeInfo = mSession->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    mOutputDims = outputTensorInfo.GetShape();

    std::cout << "Input Type: " << inputType << std::endl;
    std::cout << "Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Input Dimension: ";
    for (int64_t i: mInputDims){
        std::cout << i << ' ';
    }
    std::cout << std::endl;
    std::cout << "Output Type: " << outputType << std::endl;
    std::cout << "Output Nodes: " << numOutputNodes << std::endl;
    std::cout << "Output Dimension: ";
    for (int64_t i: mOutputDims){
        std::cout << i << ' ';
    }
    std::cout << std::endl;

}

int ObjectDetector::Inference(const cv::Mat imageBGR) {
    std::cout << std::endl << "Starting Preprocessing:" << std::endl;
    // for time measuring
    const auto start = clock_time::now();

    // Calculate flat tensor input size:
    int inputTensorSize = 1;
    for (const auto& e: mInputDims){
        inputTensorSize *= e;
    }

    std::vector<float> inputTensorValues(inputTensorSize);
    CreateTensorFromImage(imageBGR, inputTensorValues);

    //debug prints:
    std::cout << "inputTensorValues.data: " << *inputTensorValues.data() << std::endl;
    std::cout << "inputTensorSize: " << inputTensorSize << std::endl;
    std::cout << "mInputDims.data: " << *mInputDims.data() << std::endl;
    std::cout << "mInputDims.size: " << mInputDims.size() << std::endl;

    //Assign memory
    std::vector<Ort::Value> inputTensors;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu( //maybe GPU allocator?
            OrtAllocatorType::OrtArenaAllocator,
            OrtMemType::OrtMemTypeDefault
            );
    // creating float input tensor. But I think we need uint8
    //inputTensors.push_back(Ort::Value::CreateTensor<float>(
    //        memoryInfo,
    //        inputTensorValues.data(),
    //        inputTensorSize,
    //        mInputDims.data(),
    //        mInputDims.size()
    //        ));

    inputTensors.push_back(Ort::Value::CreateTensor(
            memoryInfo,
            inputTensorValues.data(),
            inputTensorSize,
            mInputDims.data(),
            mInputDims.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
    ));

    //Create output tensor
    size_t outputTensorSize = 1;
    for (const auto& e: mOutputDims){
        outputTensorSize *= e;
    }
    std::vector<float> outputTensorValues(outputTensorSize);

    std::vector<Ort::Value> outputTensors;

    //debug prints:
    std::cout << "outputTensorValues.data: " << *outputTensorValues.data() << std::endl;
    std::cout << "outputTensorSize: " << outputTensorSize << std::endl;
    std::cout << "mOutputDims.data: ";
    for (int64_t i: mOutputDims){
        std::cout << i << " ";
    }
    std::cout << std::endl;
    // << *mOutputDims.data() << std::endl;
    std::cout << "mOutputDims.size: " << 4 * mOutputDims.size() << std::endl;

    outputTensors.push_back(
            //Ort::Value::CreateTensor<float>(
            //    memoryInfo,
            //    outputTensorValues.data(),
            //    outputTensorSize,
            //    mOutputDims.data(),
            //    mOutputDims.size())
            //    );
            Ort::Value::CreateTensor(
                    memoryInfo,
                    outputTensorValues.data(),
                    4 * outputTensorSize,
                    mOutputDims.data(),
                    mOutputDims.size(), //*4 for Float32. https://github.com/triton-inference-server/server/issues/4478
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                    ));

    const sec preprocessing_time = clock_time::now() - start;
    std::cout << "The preprocessing takes " << preprocessing_time.count() << "s"
              << std::endl;


    // Inference:
    const auto before_inference = std::chrono::system_clock::now();

    std::vector<const char*> inputNames{mInputName};
    std::vector<const char*> outputNames{mOutputName};

    std::cout << "\nStarting Inferencing:" << std::endl;
    mSession->Run(Ort::RunOptions{nullptr},
                  inputNames.data(),
                  inputTensors.data(),
                  1,
                  outputNames.data(),
                  outputTensors.data(),
                  1);

    const sec inference_time = clock_time::now() - start;
    std::cout << "The preprocessing takes " << inference_time.count() << "s"
              << std::endl;


    auto after = std::chrono::system_clock::now();
}


// Create a tensor from the input image
void ObjectDetector::CreateTensorFromImage(
        const cv::Mat& img, std::vector<float>& inputTensorValues) {
    cv::Mat imageRGB, scaledImage, preprocessedImage;

    auto type = img.type();

    //std::string type="uint8";

    /******* Preprocessing *******/
    // Scale image pixels from [0 255] to [-1, 1]
    //img.convertTo(scaledImage, CV_32F, 2.0f / 255.0f, -1.0f);
    if (type != 24){
        img.convertTo(scaledImage, CV_8U, 2.0f / 255.0f, -1.0f);
    }
    else {
        img.convertTo(scaledImage, CV_8U, 1.0f, 0.0f);
    }
    // Convert HWC to CHW
    cv::dnn::blobFromImage(scaledImage, preprocessedImage);

    // Assign the input image to the input tensor
    inputTensorValues.assign(preprocessedImage.begin<float>(),
                             preprocessedImage.end<float>());
}
