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

    // print available providers
    std::cout << "Avaiable providers: ";
    for (std::string i : Ort::GetAvailableProviders()) {
        std::cout << " " << i << " ";
    };
    std::cout << std::endl;

    OrtCUDAProviderOptions cuda_opts = OrtCUDAProviderOptions();
    cuda_opts.device_id = 0;

    // Ort Session
    Ort::SessionOptions sessionOptions;
    // Enable Cuda
    sessionOptions.AppendExecutionProvider_CUDA(cuda_opts);
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
        mInputDims = {1, 512, 512, 3};
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

    std::vector<uint8_t> inputTensorValues(inputTensorSize);
    CreateTensorFromImage(imageBGR, inputTensorValues);

    // inputTensorValues is flattened array with chw format.
    // inputTensorValues should be reordererd to hwc format

    //debug prints:
    std::cout << "  inputTensorValues.data: " << *inputTensorValues.data() << std::endl;
    std::cout << "  inputTensorSize: " << inputTensorSize << std::endl;
    std::cout << "  mInputDims.data: " << *mInputDims.data() << std::endl;
    std::cout << "  mInputDims.size: " << mInputDims.size() << std::endl;

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
    std::cout << "\n  outputTensorValues.data: " << *outputTensorValues.data() << std::endl;
    std::cout << "  outputTensorSize: " << outputTensorSize << std::endl;
    std::cout << "  mOutputDims.data: ";
    for (int64_t i: mOutputDims){
        std::cout << i << " ";
    }
    std::cout << std::endl;
    // << *mOutputDims.data() << std::endl;
    std::cout << "  mOutputDims.size: " << 4 * mOutputDims.size() << std::endl;

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
    std::cout << "The inference takes " << inference_time.count() << "s"
              << std::endl;

    // debug: try to show image
    this->CreateInferenceImage(&outputTensors.back(), imageBGR);

    auto after = std::chrono::system_clock::now();
}


// Create a tensor from the input image
void ObjectDetector::CreateTensorFromImage(
        const cv::Mat& img, std::vector<uint8_t>& inputTensorValues) {
    auto type = img.type();
    auto input_height = mInputDims.at(1);
    auto input_width = mInputDims.at(2);
    int nativeRows = img.rows;
    int nativeCols = img.cols;

    // Init new Images todo: can probably be simplified and made more memory efficient
    // also todo: shift to gpu memory maybe helpful.
    cv::Mat scaledImage(nativeRows, nativeCols, CV_8UC3);
    cv::Mat preprocessedImage(input_height, input_width, CV_8UC3);

    //cv::cuda::GpuMat
    //img.convertTo(img, cv::COLOR_BGRA2RGBA);

    std::vector<cv::Mat> rgbchannel;
    cv::split(img, rgbchannel);
    rgbchannel.erase(rgbchannel.begin() + 3);

    cv::merge(rgbchannel, scaledImage);

    std::cout << "img channels: " << img.channels() << std::endl;
    std::cout << "scaled_img channels: " << scaledImage.channels() << std::endl;
    std::cout << "prepro_img channels: " << preprocessedImage.channels() << std::endl;

    /******* Preprocessing *******/
    // Scale image pixels from [0 255] to [-1, 1]
    //img.convertTo(scaledImage, CV_32F, 2.0f / 255.0f, -1.0f);
    if (type != 24){
        scaledImage.convertTo(scaledImage, CV_8U, 2.0f / 255.0f, -1.0f);
    }
    else {
        scaledImage.convertTo(scaledImage, CV_8U, 1.0f, 0.0f);
    }
    // Convert HWC to CHW
    if (!this->hwc){
        std::cout << "ATTENTION BLOB" << std::endl;
        cv::dnn::blobFromImage(scaledImage, preprocessedImage);
    }
    else {
        //cv::COLOR_RGB2BGR todo: find out
        //cv::COLOR_BGR2RGB
        //cv::cvtColor(scaledImage, preprocessedImage, cv::COLOR_RGB2BGR);
        //scaledImage.convertTo(preprocessedImage, cv::COLOR_RGBA2BGRA);
        cv::resize(scaledImage,
                   preprocessedImage,
                   cv::Size(input_width,input_height),
                   cv::INTER_LINEAR);
        cv::cvtColor(preprocessedImage, preprocessedImage, cv::COLOR_RGB2BGR);
    }

    std::cout << "prepro_img channels: " << preprocessedImage.channels() << std::endl;

    std::cout << "Tensorsize: " << inputTensorValues.size() << std::endl;
    std::cout << "MAT size: " << preprocessedImage.size().height * preprocessedImage.size().width * preprocessedImage.channels() << std::endl;

    // Assign MAT values to flat vector
    // this is from here: https://stackoverflow.com/a/26685567
    inputTensorValues.assign(preprocessedImage.data, preprocessedImage.data + (preprocessedImage.total() * preprocessedImage.channels()));
}

void ObjectDetector::CreateInferenceImage(
        Ort::Value *outputTensor,
        cv::Mat inputImage
        ) {
    // todo: add box data and original opencvmat
    std::cout << "\nShowing image:" << std::endl;


    // const float *data = outputTensor->GetTensorData<float>();

    auto shape = outputTensor->GetTensorTypeAndShapeInfo().GetShape();

    // Shape should be 100 Boxes with 7 values:
    // Box coordinates are 1:5
    // class id's is 6
    // scores is 5
    std::cout << "  Tensor_shape: ";
    for (long i : shape) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    float* floatarr = outputTensor->GetTensorMutableData<float>();

    // todo: confidences seem to be pretty low. Somethings off maybe bgr<->rgb?
    for(int row=0; row<shape[1]; row++){
        // init indexes for easy access of flattened array.
        int row_index = row * 7; // index of forst value for row. Because of flattened array.
        int confidence_index = row_index + 5; // confidence value is on the 5th place of the row
        int class_index = row_index + 6;
        std::vector<float> box_coordinates;

        // fills vector with coordinates for box. Should be ymin, xmin, ymax and xmax.
        // They still need to be upscaled, because they are respective to input size.
        for(int i=1; i<5; i++) {
            box_coordinates.push_back(floatarr[row_index + i]);
        }
        int class_val = floatarr[class_index];
        float confidence = floatarr[confidence_index];

        // debug: print detected boxes
        std::cout << "Box" << row << ":";
        std::cout << "  conf: " << confidence
                << " class: " << class_val
                << " box_coordinates: ";
        for (float coord: box_coordinates) {
            std::cout << coord << " ";
        }
        std::cout << std::endl;
    };



}

