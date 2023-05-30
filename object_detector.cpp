//
// Some code is lent from: https://github.com/freshtechyy/ONNX-Runtime-GPU-image-classifciation-example at the point of writing licensed with the MIT license.
// The rest is written by me: https://github.com/SiBensberg
//
#include "object_detector.h"

// Colors for class ID 0,1,2,3 respective:
const cv::Scalar BLUE = {180, 128, 0};
const cv::Scalar YELLOW = {77, 220, 255};
const cv::Scalar ORANGE = {0, 110, 250};
const cv::Scalar BIGORANGE = {60, 30, 190};
const std::vector<cv::Scalar> COLORS = {BLUE, YELLOW, ORANGE, BIGORANGE};


ObjectDetector::ObjectDetector(const std::string &modelPath) {
    /**
     * Init an object detector class taht loads a ONNX model from model path and uses it to infer bounding boxes
     *
     * @param modelPath path to ONNX model.
     */
    std::cout << "Initiating ObjectDetector: " << std::endl;
    // Create Environment:
    std::string instance_Name = "Object Detector";
    mEnv = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instance_Name.c_str());

    // print available providers
    std::cout << "   - Available providers: ";
    for (const std::string &prov: Ort::GetAvailableProviders()) {
        std::cout << " " << prov << " ";
    }
    std::cout << std::endl;

    OrtCUDAProviderOptions cuda_opts; // todo: what happens without cuda?
    cuda_opts.device_id = 0;

    // Ort Session
    Ort::SessionOptions sessionOptions;
    // Enable Cuda
    sessionOptions.AppendExecutionProvider_CUDA(cuda_opts);
    sessionOptions.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED); // other optimization levels ara avaiable
    // Load model
    Ort::OrtRelease(mSession); // release nullptr initialized session object to make a new one
    mSession = Ort::Session(mEnv, modelPath.c_str(), sessionOptions); // model gets loaded

    // Extract input info:
    auto numInputNodes = mSession.GetInputCount();
    //mInputName = mSession->GetInputName(0, allocator);
    mInputName = "image_arrays:0"; //todo: initialize this more general or with constructor parameter

    // Input type:
    auto inputTypeInfo = mSession.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    auto inputType = inputTensorInfo.GetElementType();

    mInputDims = inputTensorInfo.GetShape();
    // Check if any input dim is variable (-1):
    bool variable = false;
    for (const auto &i: mInputDims) {
        variable |= i <= 0;
    }
    if (variable) {
        std::cout
                << "Input size of exported ONNX model is variable. For this reason it has to be predefined.\nSetting it to 512x512"
                << std::endl;
        mInputDims = mDefaultInputDims;
    }

    auto numOutputNodes = mSession.GetOutputCount();
    //mOutputName = mSession -> GetOutputNameAllocated(0, allocator);
    mOutputName = "detections:0";

    auto outputTypeInfo = mSession.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    auto outputType = outputTensorInfo.GetElementType();
    mOutputDims = outputTensorInfo.GetShape();

    std::cout << "Input Type: " << inputType << std::endl;
    std::cout << "Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Input Dimension: ";
    for (const auto &i: mInputDims) {
        std::cout << i << ' ';
    }
    std::cout << std::endl;
    std::cout << "Output Type: " << outputType << std::endl;
    std::cout << "Output Nodes: " << numOutputNodes << std::endl;
    std::cout << "Output Dimension: ";
    for (const auto &i: mOutputDims) {
        std::cout << i << ' ';
    }
    std::cout << std::endl;

}

std::vector<std::vector<std::vector<float>>> ObjectDetector::inference(const std::vector<cv::Mat> &imagesBGR) const {
    /**
     * Inferences bounding boxes on the given images.
     * Input is vector of n images which will be batch inferred.
     * If batch size doesn't fit the input dimensions an error will be thrown.
     *
     * @param imagesBGR vector with all images to infer. cv:Mat
     * @return returns vector with detected bounding boxes above confidence 0.09
     */
    // for time measuring
    const auto start = clock_time::now();
    auto num_images = imagesBGR.size();

    if (num_images > mDefaultInputDims[0]) {
        throw std::domain_error("More Camera images then the network can inference. "
                                "Adjust network input dimensions or lower number of cameras.");
    }

    // Calculate flat tensor input size:
    long inputTensorSize = 1;
    for (const auto &e: mInputDims) {
        inputTensorSize *= e;
    }
    // todo: assertion for the following?
    long input_image_size = inputTensorSize / num_images;

    // inputTensorValues is flattened array with chw format.
    // inputTensorValues must be reordered to hwc format
    // vector of input tensor values:
    std::vector<uint8_t> inputTensorValues;
    for (int i=0; i<num_images; ++i) {
        std::vector<uint8_t> input_image_values(input_image_size);
        createTensorFromImage(imagesBGR[i], input_image_values);
        inputTensorValues.insert(inputTensorValues.end(), input_image_values.begin(), input_image_values.end());

        //input_tensor_values_vector[i] = createTensorFromImage(imagesBGR[i]);
    }

    //Assign memory
    std::vector<Ort::Value> inputTensors;

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
    for (const auto &e: mOutputDims) {
        outputTensorSize *= e;
    }
    std::vector<float> outputTensorValues(outputTensorSize);

    std::vector<Ort::Value> outputTensors;
    outputTensors.push_back(
            Ort::Value::CreateTensor(
                    memoryInfo,
                    outputTensorValues.data(),
                    4 *
                    outputTensorSize, //*4 for Float32. https://github.com/triton-inference-server/server/issues/4478
                    mOutputDims.data(),
                    mOutputDims.size(),
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            ));

    const sec preprocessing_time = clock_time::now() - start;

    // inference:
    const auto before_inference = std::chrono::system_clock::now();

    std::vector<const char *> inputNames{mInputName};
    std::vector<const char *> outputNames{mOutputName};

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

    auto outputBoxes = this->calculateBoxes(outputTensors.back());

    const sec after = clock_time::now() - start;

    // std::cout << "Image Precessing and inference taking an overall: " << after.count() << "s" << std::endl;

    return outputBoxes;
}


// Create a tensor from the input image
void ObjectDetector::createTensorFromImage(
        const cv::Mat &img, std::vector<uint8_t> &inputTensorValues) const {
    /**
     * Creates a ONNX tensor for the session. Takes the cv:Mat as input and writes the to inputTensorValues.
     * @param img Reference of cv:Mat image to be inferred.
     * @param inputTensorValues Flat uint8 vector with all the values from the input image.
     */
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
    if (type != 24) {
        scaledImage.convertTo(scaledImage, CV_8U, 2.0f / 255.0f, -1.0f);
    } else {
        scaledImage.convertTo(scaledImage, CV_8U, 1.0f, 0.0f);
    }
    // Convert HWC to CHW
    // Tensorflow models do mostly need input image in NHWC.
    // PyTorch usually needs NCHW.
    if (!this->hwc) {
        // std::cout<< "ATTENTION BLOB" << std::endl;
        cv::dnn::blobFromImage(scaledImage, preprocessedImage);
    } else {
        cv::resize(scaledImage,
                   preprocessedImage,
                   cv::Size(input_width, input_height),
                   cv::INTER_LINEAR);
        cv::cvtColor(preprocessedImage, preprocessedImage, cv::COLOR_RGB2BGR);
    }

    // Assign MAT values to flat vector
    // this is from here: https://stackoverflow.com/a/26685567
    inputTensorValues.assign(preprocessedImage.data,
                             preprocessedImage.data + (preprocessedImage.total() * preprocessedImage.channels()));
}

std::vector<std::vector<std::vector<float>>> ObjectDetector::calculateBoxes(const Ort::Value &outputTensor) const {
    /**
     * Extract the output boxes data from the flat output vector.
     * Also scales them back to initial image size.
     * Filters out every box with confidence score <= 0.09.
     *
     * @param outputTensor flat output tensor from ONNX session
     * @return Scaled output boxes in vector. First vector is for the image second for each box
     */
    // Calculate Factors for later upscaling of boxes with very sexy casts
    auto width_factor = (float) cameraInputDims[1] / (float) mInputDims.at(2);
    auto height_factor = (float) cameraInputDims[0] / (float) mInputDims.at(1);
    auto shape = outputTensor.GetTensorTypeAndShapeInfo().GetShape();

    // Get data from tensor:
    const auto data = outputTensor.GetTensorData<float>();

    std::vector<std::vector<std::vector<float>>> outputBoxes(shape[0]); //one vector for each box, for each image

    // for every image
    for (int img = 0; img < shape[0]; ++img) {
        // for every of the 100 boxes:
        for (int row = 0; row < shape[1]; ++row) {
            // init indexes for easy access of flattened array.
            const auto confidence = *(data + (row * 7 + 5)); // confidence value is on the 5th place of the row
            const auto class_id = *(data + (row * 7 + 6));

            if (confidence >= 0.09) {
                std::vector<float> box_data{class_id, confidence, //test1, test2, test3, test4};
                                            *(data + (row * 7 + 1)) * height_factor,
                                            *(data + (row * 7 + 2)) * width_factor,
                                            *(data + (row * 7 + 3)) * height_factor,
                                            *(data + (row * 7 + 4)) * width_factor};
                outputBoxes[img].push_back(box_data);
            }
        }
    }


    return outputBoxes;
}

