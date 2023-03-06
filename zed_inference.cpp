//
// Created by simon on 28.01.23.
//

#include "zed_inference.h"

// General imports
#include <unistd.h>

// ZED includes
#include <sl/Camera.hpp>

// OpenCV includes
#include <opencv2/opencv.hpp>
// OpenCV dep
#include <opencv2/cvconfig.h>

// Local imports
#include "slmat_to_cvmat.h" // Functions to show images

// onnx includes
//#include "onnx/onnx_pb.h"
//#include "onnx/proto_utils.h"
#include <onnxruntime_cxx_api.h>



ZedInference::ZedInference() {
    std::cout << "Created ZedInference Class" << std::endl;
    this->running = false;
}

void ZedInference::grab_rgb_image() {

    sl::Resolution image_size = zed.getCameraInformation().camera_resolution;

    sl::Mat image_zed(image_size.width, image_size.height, sl::MAT_TYPE::U8_C4);
    //sl::Mat image_zed(zed.getResolution(), MAT_TYPE::U8_C4);
    cv::Mat image_ocv = slMat2cvMat(image_zed);

    if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
        std::cout << "grab image" << std::endl;
        // Retrieve the left image in sl::Mat
        // The cv::Mat is automatically updated
        zed.retrieveImage(image_zed, sl::VIEW::LEFT);

        // Inference image
        ZedInference::inference_rgb_image(image_ocv);

        //debug
        sleep(10);

        // Display the left image from the cv::Mat object
        cv::imshow("Image", image_ocv);
        cv::waitKey(25);
    } else {
        std::cout << "Could not grab image.\nWaiting for 5 seconds" << std::endl;
        sleep(5);
    }
}

int ZedInference::run(){
    running = true;
    bool camera_open = false;

    sl::InitParameters init_params;
    init_params.camera_resolution = sl::RESOLUTION::HD1080;
    init_params.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_params.coordinate_units = sl::UNIT::METER;

    // Open the camera
    std::cout << "Opening Camera..." << std::endl;
    if (camera_open == false) {
        sl::ERROR_CODE err = zed.open(init_params);
        camera_open = true;

        if (err != sl::ERROR_CODE::SUCCESS) {
            printf("%s\n", toString(err).c_str());
            // zed.close();
            camera_open = false; // Quit if an error occurred
            // return 0;
        }
    }

    // If Camera could not be opened try with svo
    if (camera_open == false){
        std::cout << "\nOpening SVO..." << std::endl;
        sl::String input_svo_path(this->svo_path);
        init_params.input.setFromSVOFile(input_svo_path);
        sl::ERROR_CODE err = zed.open(init_params);

        if (err != sl::ERROR_CODE::SUCCESS){
            printf("%s\n", toString(err).c_str());
            zed.close();
            camera_open = false;
            return 0;
        }
        else {
            camera_open = true;
            std::cout << "Successfully opened: " << this->svo_path << std::endl;
        }
    }

    // run camera
    if (running && camera_open){
        std::cout << "\nRun camera: " << std::endl;
    }
    while (running && camera_open){
        ZedInference::grab_rgb_image();
    }

}

void ZedInference::grab_depth_image() {

}

void ZedInference::inference_rgb_image(cv::Mat rgb_image) {
    ZedInference::Detector.Inference(rgb_image);

}
