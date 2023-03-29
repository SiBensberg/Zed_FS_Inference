//
// Created by simon on 04.02.23.
//
#include "object_detector.h"

// Colors for class ID 0,1,2,3 respective:
const cv::Scalar BLUE = {180, 128, 0};
const cv::Scalar YELLOW = {77, 220, 255};
const cv::Scalar ORANGE = {0, 110, 250};
const cv::Scalar BIGORANGE = {60, 30, 190};
const std::vector<cv::Scalar> COLORS = {BLUE, YELLOW, ORANGE, BIGORANGE};


ObjectDetector::ObjectDetector(const std::string &modelPath) {
    std::cout<< "Initiating ObjectDetector: " << std::endl;
    // Create Environment:
    std::string instance_Name{"Object Detector"};
    // make shared for creating new shared pointer
    mEnv = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instance_Name.c_str());

    // print available providers
    std::cout<< "Avaiable providers: ";
    for (std::string i : Ort::GetAvailableProviders()) {
        std::cout<< " " << i << " ";
    };
    std::cout<< std::endl;

    OrtCUDAProviderOptions cuda_opts = OrtCUDAProviderOptions();
    cuda_opts.device_id = 0;

    // Ort Session
    Ort::SessionOptions sessionOptions;
    // Enable Cuda
    sessionOptions.AppendExecutionProvider_CUDA(cuda_opts);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // Load model
    Ort::OrtRelease(mSession);
    mSession = Ort::Session(mEnv, modelPath.c_str(), sessionOptions);

    // Allocator
    //Ort::AllocatorWithDefaultOptions allocator;

    // Extract input info:
    size_t numInputNodes = mSession.GetInputCount();
    //mInputName = mSession->GetInputName(0, allocator);
    mInputName = "image_arrays:0"; //todo: initialize this more general or with constructor parameter

    // Input type:
    Ort::TypeInfo inputTypeInfo = mSession.GetInputTypeInfo(0);
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
        // std::cout<< "Input size of exported ONNX model is variable. For this reason it has to be predefined." << std::endl << "Setting it to 512x512" << std::endl;
        mInputDims = {1, 512, 512, 3};
    }


    size_t numOutputNodes = mSession.GetOutputCount();
    //mOutputName = mSession -> GetOutputNameAllocated(0, allocator);
    mOutputName = "detections:0";

    Ort::TypeInfo outputTypeInfo = mSession.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    mOutputDims = outputTensorInfo.GetShape();

    std::cout<< "Input Type: " << inputType << std::endl;
    std::cout<< "Input Nodes: " << numInputNodes << std::endl;
    std::cout<< "Input Dimension: ";
    for (int64_t i: mInputDims){
        std::cout<< i << ' ';
    }
    std::cout<< std::endl;
    std::cout<< "Output Type: " << outputType << std::endl;
    std::cout<< "Output Nodes: " << numOutputNodes << std::endl;
    std::cout<< "Output Dimension: ";
    for (int64_t i: mOutputDims){
        std::cout<< i << ' ';
    }
    std::cout<< std::endl;

}

std::vector<std::vector<float>> ObjectDetector::Inference(const cv::Mat& imageBGR) {
    // std::cout<< std::endl << "Starting Preprocessing:" << std::endl;
    // for time measuring
    const auto start = clock_time::now();

    // Calculate flat tensor input size:
    long inputTensorSize = 1;
    for (const auto& e: mInputDims){
        inputTensorSize *= e;
    }

    std::vector<uint8_t> inputTensorValues(inputTensorSize);
    CreateTensorFromImage(imageBGR, inputTensorValues);

    // inputTensorValues is flattened array with chw format.
    // inputTensorValues should be reordererd to hwc format

    //debug prints:
    // std::cout<< "  inputTensorValues.data: " << *inputTensorValues.data() << std::endl;
    // std::cout<< "  inputTensorSize: " << inputTensorSize << std::endl;
    // std::cout<< "  mInputDims.data: " << *mInputDims.data() << std::endl;
    // std::cout<< "  mInputDims.size: " << mInputDims.size() << std::endl;

    //Assign memory
    std::vector<Ort::Value> inputTensors;

    // Memory
    // Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu( //maybe GPU allocator?
    //         //OrtAllocatorType::OrtArenaAllocator,
    //         OrtAllocatorType::OrtDeviceAllocator,
    //         OrtMemType::OrtMemTypeDefault
    // );

    //Ort::MemoryInfo memoryInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    //auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    //Ort::Allocator cuda_allocator(cuda_mem_info);

    // Create input tensor
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
    // std::cout<< "\n  outputTensorValues.data: " << *outputTensorValues.data() << std::endl;
    // std::cout<< "  outputTensorSize: " << outputTensorSize << std::endl;
    // std::cout<< "  mOutputDims.data: ";
    for (int64_t i: mOutputDims){
        // std::cout<< i << " ";
    }
    // std::cout<< std::endl;
    // << *mOutputDims.data() << std::endl;
    // std::cout<< "  mOutputDims.size: " << 4 * mOutputDims.size() << std::endl;

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
                    4 * outputTensorSize, //*4 for Float32. https://github.com/triton-inference-server/server/issues/4478
                    mOutputDims.data(),
                    mOutputDims.size(),
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            ));

    const sec preprocessing_time = clock_time::now() - start;
    // std::cout<< "The preprocessing takes " << preprocessing_time.count() << "s"
    //          << std::endl;


    // Inference:
    const auto before_inference = std::chrono::system_clock::now();

    std::vector<const char*> inputNames{mInputName};
    std::vector<const char*> outputNames{mOutputName};

    // std::cout<< "\nStarting Inferencing:" << std::endl;
    mSession.Run(Ort::RunOptions{nullptr},
                  inputNames.data(),
                  inputTensors.data(),
                  1,
                  outputNames.data(),
                  outputTensors.data(),
                  1);

    const sec inference_time = clock_time::now() - start;
    // std::cout<< "The inference takes " << inference_time.count() << "s" << std::endl;

    // debug: try to show image
    std::vector<std::vector<float>> outputBoxes = this->CreateInferenceImage(&outputTensors.back());

    const sec after = clock_time::now() - start;

    std::cout<< "Image Precessing and Inference taking a overall: " << after.count() << "s" << std::endl;

    return outputBoxes;
}


// Create a tensor from the input image
void ObjectDetector::CreateTensorFromImage(
        const cv::Mat& img, std::vector<uint8_t>& inputTensorValues) {
    auto type = img.type();
    auto input_height = mInputDims.at(1);
    auto input_width = mInputDims.at(2);
    int nativeRows = img.rows;
    int nativeCols = img.cols;

    this->cameraInputDims = {nativeRows, nativeCols};

    // Init new Images todo: can probably be simplified and made more memory efficient
    // also todo: shift to gpu memory maybe helpful.
    cv::Mat scaledImage(nativeRows, nativeCols, CV_8UC3);
    cv::Mat preprocessedImage(input_height, input_width, CV_8UC3);

    std::vector<cv::Mat> rgbchannel;
    cv::split(img, rgbchannel);
    rgbchannel.erase(rgbchannel.begin() + 3);

    cv::merge(rgbchannel, scaledImage);

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
    // Tensorflow models do mostly need input image in NHWC.
    // PyTorch usually needs NCHW.
    if (!this->hwc){
        // std::cout<< "ATTENTION BLOB" << std::endl;
        cv::dnn::blobFromImage(scaledImage, preprocessedImage);
    }
    else {
        cv::resize(scaledImage,
                   preprocessedImage,
                   cv::Size(input_width,input_height),
                   cv::INTER_LINEAR);
        cv::cvtColor(preprocessedImage, preprocessedImage, cv::COLOR_RGB2BGR);
    }

    // Assign MAT values to flat vector
    // this is from here: https://stackoverflow.com/a/26685567
    inputTensorValues.assign(preprocessedImage.data, preprocessedImage.data + (preprocessedImage.total() * preprocessedImage.channels()));
}

std::vector<std::vector<float>> ObjectDetector::CreateInferenceImage(
        Ort::Value *outputTensor
        ) {
    // Calculate Factors for later upscaling of boxes with very sexy casts
    auto width_factor = (float) cameraInputDims[1] / (float)  mInputDims.at(2);
    auto height_factor = (float) cameraInputDims[0] / (float) mInputDims.at(1);
    auto shape = outputTensor->GetTensorTypeAndShapeInfo().GetShape();

    // Get data from tensor:
    auto* floatarr = outputTensor->GetTensorMutableData<float>();

    std::vector<std::vector<float>> outputBoxes;

    // for every of the 100 boxes:
    for(int row=0; row<shape[1]; row++){
        // init vector to fill:
        std::vector<float> box_data;

        // init indexes for easy access of flattened array.
        int row_index = row * 7; // index of forst value for row. Because of flattened array.
        int confidence_index = row_index + 5; // confidence value is on the 5th place of the row
        int class_index = row_index + 6;
        std::vector<int32_t> box_coordinates;

        // fills vector with coordinates for box. Should be ymin, xmin, ymax and xmax.
        // They still need to be upscaled, because they are respective to input size.
        for(int i=1; i<5; i++) {
            box_coordinates.push_back(floatarr[row_index + i]);
        }
        // Upscale boxes
        box_coordinates[0] *= height_factor;
        box_coordinates[1] *= width_factor;
        box_coordinates[2] *= height_factor;
        box_coordinates[3] *= width_factor;

        int class_id = floatarr[class_index];
        float confidence = floatarr[confidence_index];


        if (confidence >= 0.09) {
            // fill box data
            // class, confidence, ymin, xmin, ymax and xmax
            box_data.push_back(class_id);
            box_data.push_back(confidence);
            for (float n: box_coordinates) {
                box_data.push_back(n);
            }
            // append box to return array:
            outputBoxes.push_back(box_data);
        }

        // debug: print detected boxes
        // std::cout<< "Box" << row << ":";
        // std::cout<< "  conf: " << confidence
        //        << " class: " << class_id
        //        << " box_coordinates: ";
        //for (float coord: box_coordinates) {
            // std::cout<< coord << " ";
        //}
        // std::cout<< std::endl;

    };
    return outputBoxes;
}

