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



ZedInference::ZedInference(): Detector("/home/carsten/Downloads/saved_model.onnx") {
    std::cout << "Created ZedInference Class" << std::endl;
    this->running = false;

    // Set configuration parameters
    init_params.depth_mode = sl::DEPTH_MODE::ULTRA; // Use ULTRA depth mode
    init_params.coordinate_units = sl::UNIT::METER; // Use millimeter units (for depth measurements)
}

void ZedInference::grabRgbImage() {
    for (int i = 0; i <= 1; i++) {
        printf("Grabbing from Camera %d...", i + 1);
        sl::Camera &zed = this->zed1;
        if (i == 0) {
            zed = this->zed1;
        } else {
            zed = this->zed2;
        }

        sl::Resolution image_size = zed.getCameraInformation().camera_resolution;

        sl::Mat image_zed(image_size.width, image_size.height, sl::MAT_TYPE::U8_C4);
        sl::Mat point_cloud;
        //sl::Mat image_zed(zed1.getResolution(), MAT_TYPE::U8_C4);
        cv::Mat image_ocv = slMat2cvMat(image_zed);

        std::vector<std::vector<float>> bboxes;
        std::vector<std::vector<float>> distances;

        if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
            // std::cout << "grab image" << std::endl;
            // Retrieve the left image in sl::Mat
            // The cv::Mat is automatically updated
            zed.retrieveImage(image_zed, sl::VIEW::LEFT);
            zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA); // Retrieve pointcloud

            // inference image
            bboxes = ZedInference::inferenceRgbImage(image_ocv);

            // Calculate Depth:
            distances = ZedInference::calculateDepth(bboxes, point_cloud);

            if (visualize and i == 0) {
                ZedInference::visualizeDetections(image_ocv, bboxes, distances, "ZED A");
            } else if (visualize and i == 1) {
                ZedInference::visualizeDetections(image_ocv, bboxes, distances, "ZED B");
            }
            // publish distances here

        } else {
            std::cout << "Could not grab image.\nWaiting for 5 seconds" << std::endl;
            sleep(5);
        }
    }

}
int ZedInference::run() {
    running = true;
    bool camera_open = false;

    sl::InitParameters init_params;
    init_params.camera_resolution = sl::RESOLUTION::HD720;
    init_params.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_params.camera_fps = 100;
    init_params.coordinate_units = sl::UNIT::METER;

    bool open = true;
    // Open the camera
    for (int i = 0; i <= 1; i++) {
        printf("Opening Camera %d...\n", i+1);
        if(i==0) {
            init_params.input.setFromSerialNumber(39833514);
            open &= zed1.open(init_params) == sl::ERROR_CODE::SUCCESS;
        }
        else{
            init_params.input.setFromSerialNumber(32593281);
            open &= zed2.open(init_params) == sl::ERROR_CODE::SUCCESS;
        }

        if (not open) {
            printf("Unable to open cameras\n");
        }
    }

/*    if (camera_open == false){
        std::cout << "\nOpening SVO..." << std::endl;
        sl::String input_svo_path(this->svo_path.c_str());
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
    }*/

    // run camera
    if (running && open){
        std::cout << "\nRun camera: " << std::endl;
    }
    while (running && open){
        ZedInference::grabRgbImage();
    }

}

std::vector<std::vector<float>> ZedInference::calculateDepth(const std::vector<std::vector<float>>& bboxes,
                                                             const sl::Mat &point_cloud) {
    // Init vector that needs to be filled and is later returned. Vector with vectors for every detected object in it.
    std::vector<std::vector<float>> coneDistances;

    // Extract depth from pointcloud with given box coordinates
    // Check if boxes are not empty:
    if (not bboxes.empty()) {
        for (const auto &box: bboxes) {
            int i, j;

            // take the lowest point in the mid of the bbox
            auto xmid = (box[3] + (box[5] - box[3]) / 2);
            i = (int) xmid;
            j = (int) box[4]; // simply bottom of box

            // Get the 3D point cloud values for pixel (i,j)
            sl::float4 point3D;
            point_cloud.getValue(i,j,&point3D);
            //float color = point3D.w;

            // fill box class, confidence, x, y, z
            coneDistances.push_back({box[0], box[1], point3D.x, point3D.y, point3D.z});

        }
    }

    return coneDistances;
}

std::vector<std::vector<float>> ZedInference::inferenceRgbImage(const cv::Mat &rgb_image) {
    std::vector<std::vector<float>> Boxes;
    Boxes = ZedInference::Detector.inference(rgb_image);

    return Boxes;
}

//als return type cv::Mat, return inputImage
void ZedInference::visualizeDetections(const cv::Mat& inputImage, std::vector<std::vector<float>> bboxes, std::vector<std::vector<float>> distances, const std::string &cam) {
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
