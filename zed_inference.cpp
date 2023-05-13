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



ZedInference::ZedInference(): Detector("../model/saved_model_b2.onnx") {
    std::cout << "Created ZedInference Class" << std::endl;
    this->running = false;

    // Set configuration parameters
    init_params.depth_mode = sl::DEPTH_MODE::ULTRA; // Use ULTRA depth mode
    init_params.coordinate_units = sl::UNIT::METER; // Use millimeter units (for depth measurements)
}

void ZedInference::grabRgbImage() {
    std::vector<sl::Mat> images_zed;
    std::vector<cv::Mat> images_cv_zed;
    std::vector<sl::Mat> pointclouds;

    for (int i=0; i<num_cameras; ++i) {
        sl::Mat image_zed(image_size.width, image_size.height, sl::MAT_TYPE::U8_C4);
        sl::Mat point_cloud;

        if (zeds[i].grab() == sl::ERROR_CODE::SUCCESS) {
            zeds[i].retrieveImage(image_zed, sl::VIEW::LEFT);
            images_zed.push_back(image_zed);
            zeds[i].retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA);
            pointclouds.push_back(point_cloud);
        }

        // I hope this works
        //cv::Mat image_ocv = slMat2cvMat(images_zed[i]);
        images_cv_zed.push_back(slMat2cvMat(image_zed));
    }

    //sl::Mat image_zed(image_size.width, image_size.height, sl::MAT_TYPE::U8_C4);
    //sl::Mat point_cloud;
    // //sl::Mat image_zed(zed1.getResolution(), MAT_TYPE::U8_C4);
    //cv::Mat image_ocv = slMat2cvMat(image_zed);


    std::vector<std::vector<std::vector<float>>> bboxes; // one vector for each bbox, one for each image.
    std::vector<std::vector<std::vector<float>>> distances;
    // inference image
    bboxes = ZedInference::inferenceRgbImage(images_cv_zed);
    // Calculate Depth:
    for (int i=0; i<num_cameras; ++i) {
        distances.push_back(ZedInference::calculateDepth(bboxes[i], pointclouds[i]));
        std::string camera_name = "ZED_%d" + std::to_string(i);
        ZedInference::visualizeDetections(images_cv_zed[i], bboxes[i], distances[i], camera_name);
    }
    //distances = ZedInference::calculateDepth(bboxes, pointclouds);
    //ZedInference::visualizeDetections(image_ocv, bboxes, distances, "ZED A");

    // publish distances here
}


int ZedInference::run() {
    running = true;
    bool camera_open = false;

    // todo: there is also a class member. fix this
    sl::InitParameters init_params;
    init_params.camera_resolution = sl::RESOLUTION::HD720;
    init_params.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_params.camera_fps = 100;
    init_params.coordinate_units = sl::UNIT::METER;

    // How many cameras are detected?
    std::vector<sl::DeviceProperties> devList = sl::Camera::getDeviceList();
    num_cameras = devList.size(); // todo: throw execption if not 2?
    std::vector<std::thread> thread_pool(num_cameras); // compute threads


    for (int i = 0; i < num_cameras; ++i) {
        std::cout << "ID : " << devList[i].id << " ,model : " << devList[i].camera_model << " , S/N : " << devList[i].serial_number << " , state : "<<devList[i].camera_state<<std::endl;
    }

    this->zeds = std::vector<sl::Camera>(num_cameras);

    //bool open = true;
    // Open the camera

    for (int i = 0; i < num_cameras; ++i) {
        init_params.input.setFromCameraID(i);
        printf("Opening Camera %d...\n", i+1);
        //init_params.input.setFromSerialNumber(39833514);
        sl::ERROR_CODE err = zeds[i].open(init_params);
        if (err == sl::ERROR_CODE::SUCCESS) {
            auto cam_info = zeds[i].getCameraInformation();
            std::cout << cam_info.camera_model << ", ID: " << i << ", SN: " << cam_info.serial_number << " Opened" << std::endl;
            camera_open = true;
        } else {
            std::cout << "ZED ID:" << i << " Error: " << err << std::endl;
            zeds[i].close();
        }
    }

    // todo: open 2 svos for simulating 2 cameras
    if (camera_open == false){
        std::cout << "\nOpening SVO..." << std::endl;

        // Init 2 fake cameras
        this->num_cameras = 1;
        this->zeds = std::vector<sl::Camera>(num_cameras);

        sl::String input_svo_path(this->svo_path.c_str());
        init_params.input.setFromSVOFile(input_svo_path);
        //sl::ERROR_CODE err = zed.open(init_params);

        for (int i=0; i < num_cameras; ++i) {
            //init_params.input.setFromCameraID(i);
            auto err = zeds[i].open(init_params);
            if (err != sl::ERROR_CODE::SUCCESS){
                printf("%s\n", toString(err).c_str());
                zeds[i].close();
                camera_open = false;
                return 0;
            }
            else {
                camera_open = true;
                std::cout << "Successfully opened: " << this->svo_path << std::endl;
            }
        }
    }

    // extract camera resolutions and throw exception if different. (Should not happen because of same input parameters)
    this->image_size = zeds[0].getCameraInformation().camera_resolution;

    if (this->num_cameras != 1) {
        sl::Resolution image_size2 = zeds[1].getCameraInformation().camera_resolution;
        if (image_size.width != image_size2.width or image_size.height != image_size2.height) {
            throw std::domain_error("ZED Cameras have different resolutions.");
        }
    }

    // ##### RUN Inference #####
    if (running && camera_open){
        std::cout << "\nRun camera: " << std::endl;
    }
    while (running && camera_open){
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

std::vector<std::vector<std::vector<float>>> ZedInference::inferenceRgbImage(std::vector<cv::Mat> &rgb_cv_images) {
    // std::vector<std::vector<float>> Boxes;
    auto Boxes = ZedInference::Detector.inference(rgb_cv_images);

    return Boxes;
}

//als return type cv::Mat, return inputImage
void ZedInference::visualizeDetections(const cv::Mat& inputImage, const std::vector<std::vector<float>> &bboxes, const std::vector<std::vector<float>> &distances, const std::string &cam) {
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
        //float x = boxDistance[2];
        //float y = boxDistance[3];
        //float z = boxDistance[4];

        std::vector<std::string> coords_to_string; // x,y,z

        // convert distances to objects to string with precision 2
        for(int n=2; n<=4; ++n) {
            std::stringstream stream;
            stream << std::fixed << std::setprecision(2) << boxDistance[n];
            coords_to_string.push_back(stream.str());
        }

        std::string xText = "x: " + coords_to_string[0] + "m";
        std::string yText = "y: " + coords_to_string[1] + "m";
        std::string zText = "z: " + coords_to_string[2] + "m";
        std::stringstream conf_stream;
        conf_stream << std::fixed << std::setprecision(2) << confidence;
        std::string confText = "P: " + conf_stream.str();

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
    cv::imshow(cam, inputImage);
    cv::waitKey(1);


}
