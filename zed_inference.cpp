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

    // Set configuration parameters
    init_params.depth_mode = sl::DEPTH_MODE::ULTRA; // Use ULTRA depth mode
    init_params.coordinate_units = sl::UNIT::METER; // Use millimeter units (for depth measurements)
}

void ZedInference::grab_rgb_image() {

    sl::Resolution image_size = zed.getCameraInformation().camera_resolution;

    sl::Mat image_zed(image_size.width, image_size.height, sl::MAT_TYPE::U8_C4);
    sl::Mat point_cloud;
    //sl::Mat image_zed(zed.getResolution(), MAT_TYPE::U8_C4);
    cv::Mat image_ocv = slMat2cvMat(image_zed);

    std::vector<std::vector<float>> bboxes;
    std::vector<std::vector<float>> distances;

    if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
        // std::cout << "grab image" << std::endl;
        // Retrieve the left image in sl::Mat
        // The cv::Mat is automatically updated
        zed.retrieveImage(image_zed, sl::VIEW::LEFT);
        zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA); // Retrieve depth

        // Inference image
        bboxes = ZedInference::inference_rgb_image(image_ocv);

        // Calculate Depth:
        distances = ZedInference::calculateDepth(bboxes, point_cloud);

        if (visualize) {
            ZedInference::visualizeDetections(image_ocv, bboxes, distances);
        }

        // Display the left image from the cv::Mat object
        //cv::imshow("Image", image_ocv);
        //cv::waitKey(1);
    } else {
        std::cout << "Could not grab image.\nWaiting for 5 seconds" << std::endl;
        sleep(5);
    }
}

int ZedInference::run(){
    running = true;
    bool camera_open = false;

    sl::InitParameters init_params;
    init_params.camera_resolution = sl::RESOLUTION::HD720;
    init_params.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_params.camera_fps = 100;
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

std::vector<std::vector<float>> ZedInference::calculateDepth(
                                                                const std::vector<std::vector<float>>& bboxes,
                                                                sl::Mat& point_cloud) {
    // Init vector that needs to be filled and is later returned. Vector with vectors for every detected object in it.
    std::vector<std::vector<float>> coneDistances;

    // Extract depth from pointcloud with given box coordinates
    // Check if boxes are not empty:
    if (!bboxes.empty()) {
        for (auto box: bboxes) {
            int i, j;

            // take the lowest point in the mid of the bbox
            auto xmid = (box[3] + (box[5] - box[3]) / 2);
            i = (int) xmid;
            j = (int) box[4]; // simply bottom of box

            // Get the 3D point cloud values for pixel (i,j)
            sl::float4 point3D;
            point_cloud.getValue(i,j,&point3D);
            float x = point3D.x;
            float y = point3D.y;
            float z = point3D.z;
            //float color = point3D.w;

            // fill box class, confidence, x, y, z
            std::vector<float> boxDistance{box[0], box[1], x, y, z};
            coneDistances.push_back(boxDistance);

            // debug:
            //std::cout << "x: " << x << " y: " << y << " z: " << z << std::endl;

        }
    }

    return coneDistances;
}

std::vector<std::vector<float>> ZedInference::inference_rgb_image(cv::Mat rgb_image) {
    std::vector<std::vector<float>> Boxes;
    Boxes = ZedInference::Detector.Inference(rgb_image);

    return Boxes;
}

void ZedInference::visualizeDetections(const cv::Mat& inputImage, std::vector<std::vector<float>> bboxes, std::vector<std::vector<float>> distances) {
    int boxIndex = 0;

    for(std::vector<float> box: bboxes) {
        // Get depth data from distances:
        std::vector<float> boxDistance = distances[boxIndex];
        boxIndex += 1;

        // Get box coordinates
        std::vector<int> coordinates;
        for (int i=2; i < box.size(); ++i) {
            coordinates.push_back((int) box[i]);
        }
        int class_id = (int) box[0];
        float confidence = box[1];

        // print boxes around detected objects:
        cv::Point min = cv::Point(coordinates[1], coordinates[0]); // ymin and xmin
        cv::Point max = cv::Point(coordinates[3],  coordinates[2]); //ymax and xmax
        cv::rectangle(inputImage, min, max, this->COLORS[class_id - 1], 1.5); // -1 because there is a background id 0

        // print coordinates next to boxes:
        float x = boxDistance[2];
        float y = boxDistance[3];
        float z = boxDistance[4];

        std::string xText = "x: " + std::to_string(x) + "m";
        std::string yText = "y: " + std::to_string(y) + "m";
        std::string zText = "z: " + std::to_string(z) + "m";
        std::string confText = std::to_string(confidence);

        float fontScale = 0.5;
        int thickness = 1;
        int baseline = 0;
        int textSpacing = 2;

        cv::Size text_size = cv::getTextSize(xText, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
        baseline += thickness;

        cv::Point xTextPoint = cv::Point(coordinates[3]+5, coordinates[0]+(text_size.height+textSpacing));
        cv::Point yTextPoint = cv::Point(coordinates[3]+5, coordinates[0]+(text_size.height+textSpacing)*2);
        cv::Point zTextPoint = cv::Point(coordinates[3]+5, coordinates[0]+(text_size.height+textSpacing)*3);
        cv::Point confTextPoint = cv::Point(coordinates[3]+5, coordinates[0]+(text_size.height+textSpacing)*4);

        cv::putText(inputImage, xText, xTextPoint, cv::FONT_HERSHEY_SIMPLEX, fontScale, this->COLORS[class_id-1], thickness);
        cv::putText(inputImage, yText, yTextPoint, cv::FONT_HERSHEY_SIMPLEX, fontScale, this->COLORS[class_id-1], thickness);
        cv::putText(inputImage, zText, zTextPoint, cv::FONT_HERSHEY_SIMPLEX, fontScale, this->COLORS[class_id-1], thickness);
        cv::putText(inputImage, confText, confTextPoint, cv::FONT_HERSHEY_SIMPLEX, fontScale, this->COLORS[class_id-1], thickness);
    }
    cv::imshow("Image", inputImage);
    cv::waitKey(1);


    }
