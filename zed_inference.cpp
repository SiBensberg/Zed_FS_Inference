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
    /**
     * This class is the main class for the project.
     */
    std::cout << "Created ZedInference Class" << std::endl;
    this->running = false;

    // Set configuration parameters
    init_params.depth_mode = sl::DEPTH_MODE::ULTRA; // Use ULTRA depth mode
    init_params.coordinate_units = sl::UNIT::METER; // Use millimeter units (for depth measurements)
    init_params.camera_resolution = sl::RESOLUTION::HD720;
    init_params.camera_fps = 100;
}

void ZedInference::grabRgbImage() {
    /**
     * Opens Threads for each camera to grab images and then batch infers the images of the cameras.
     * Later visualize or publish detected cones.
     *
     * Name might be misleading todo: better name!
     */
    // allocate space for images, pcl and timestamps
    std::vector<sl::Mat> images_zed(num_cameras);
    std::vector<cv::Mat> images_cv_zed(num_cameras);
    std::vector<sl::Mat> pointclouds(num_cameras);
    std::vector<sl::Timestamp> timestamps(num_cameras);

    // Open a thread for every camera that runs continously and grabs a new image.
    // This runs *async* and overwrites the last image for the inference to always be executed on the latest image.
    for (int z = 0; z < num_cameras; ++z) {
        if (zeds[z].isOpened()) {
            // camera acquisition thread
            thread_pool[z] = std::thread(zed_acquisition, std::ref(zeds[z]), std::ref(images_zed[z]),
                                         std::ref(pointclouds[z]), std::ref(images_cv_zed[z]), std::ref(timestamps[z]), std::ref(this->running));
        }
    }

    // Sleep is necessary. Else the first images will be empty.
    // Opening the threads is async, thus the inference will be called too early with all references empty.
    // todo: is this the correct way or is there a safer way?
    sl::sleep_ms(1000); // sleep for a second

    // inference image
    while (running) {
        // todo: Maybe allocate space outside loop to avoid reallocating it every split second? How?
        std::vector<std::vector<std::vector<float>>> bboxes; // first vector is data for each box. Next is for each image all boxes.
        std::vector<std::vector<std::vector<float>>> distances; // same as above

        // Copy for visualising. Else the images in "images_cv_zed" will already be overwritten by the next ones by the async thread.
        // Visualization will look off.
        // This is probably really, really, really slow and should not be used in "production".
        std::vector<cv::Mat> images_cv_zed_copy(num_cameras);
        if (visualize) {
            for (int i=0; i<num_cameras; ++i) {
                images_cv_zed[i].copyTo(images_cv_zed_copy[i]);
            }
        }

        bboxes = ZedInference::inferenceRgbImage(images_cv_zed);
        // Calculate Depth:
        for (int i = 0; i < num_cameras; ++i) {
            distances.push_back(ZedInference::calculateDepth(bboxes[i], pointclouds[i]));
            std::string camera_name = "ZED_%d" + std::to_string(i);

            if (visualize) {
                ZedInference::visualizeDetections(images_cv_zed_copy[i], bboxes[i], distances[i], camera_name);
            }

            // publish distances here
        }
    }
}


int ZedInference::run() {
    /**
    * Starts the inference process.
    * First opens the cameras or if none found, the SVO's.
    * Then runs the inference continuously on the newest images.
    *
    * @return
    */
    running = true;
    bool camera_open = false;

    // Detect the number of cameras.
    std::vector<sl::DeviceProperties> devList = sl::Camera::getDeviceList();
    num_cameras = (int) devList.size();
    this->zeds = std::vector<sl::Camera>(num_cameras);
    for (int i = 0; i < num_cameras; ++i) {
        std::cout << "ID : " << devList[i].id << " ,model : " << devList[i].camera_model << " , S/N : " << devList[i].serial_number << " , state : "<<devList[i].camera_state<<std::endl;
    }

    // Init threads for every camera
    this->thread_pool = std::vector<std::thread>(num_cameras); // compute threads

    // Open the cameras.
    if (this->num_cameras != 0) {
        for (int i = 0; i < num_cameras; ++i) {
            init_params.input.setFromCameraID(i);
            printf("Opening Camera %d...\n", i + 1);
            //init_params.input.setFromSerialNumber(39833514);
            sl::ERROR_CODE err = zeds[i].open(init_params);
            if (err == sl::ERROR_CODE::SUCCESS) {
                auto cam_info = zeds[i].getCameraInformation();
                std::cout << cam_info.camera_model << ", ID: " << i << ", SN: " << cam_info.serial_number << " Opened"
                          << std::endl;
                camera_open = true;
            } else {
                std::cout << "ZED ID:" << i << " Error: " << err << std::endl;
                zeds[i].close();

                // throw error
                std::stringstream error_message;
                error_message << "Could not open Camera " << i;
                throw std::domain_error(error_message.str());
            }
        }
    }

    // If no cameras are found open SVO files.
    if (!camera_open && this->num_cameras == 0){
        // Init 2 fake cameras
        this->num_cameras = this->fake_cameras;
        std::cout << "\nOpening " << this->num_cameras << " SVOs to simulate cameras." << std::endl;

        this->zeds = std::vector<sl::Camera>(num_cameras);
        this->thread_pool = std::vector<std::thread>(num_cameras); // Thread pool will be empty if no real cameras detected

        sl::String input_svo_path(this->svo_path.c_str());
        init_params.input.setFromSVOFile(input_svo_path);

        // Open n-times the SVO to simulate n cameras.
        for (int i=0; i < num_cameras; ++i) {
            auto err = zeds[i].open(init_params);
            if (err != sl::ERROR_CODE::SUCCESS){
                printf("%s\n", toString(err).c_str());
                zeds[i].close();
                camera_open = false;

                // throw error
                std::stringstream error_message;
                error_message << "Could not open SVO " << i;
                throw std::domain_error(error_message.str());
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
        for (int i=0; i < num_cameras; ++i) {
            sl::Resolution image_size2 = zeds[i].getCameraInformation().camera_resolution;
            if (image_size.width != image_size2.width or image_size.height != image_size2.height) {
                std::stringstream error_message;
                error_message << "ZED Cameras have different resolutions:\nCamera 0: "
                                << image_size.width << "x" << image_size.height << "\n"
                                << "Camera " << i << ":" << image_size2.width << "x" << image_size2.height;
                throw std::domain_error(error_message.str());
            }
        }
    }

    // ##### RUN Inference #####
    if (running && camera_open){
        std::cout << "\nRun camera: " << std::endl;
        ZedInference::grabRgbImage();
    }
}

std::vector<std::vector<float>> ZedInference::calculateDepth(const std::vector<std::vector<float>>& bboxes,
                                                             const sl::Mat &point_cloud) {
    /**
     * Take calculated bounding boxes and the point-cloud to extract depth to each cone.
     * For this th midpoint at the bottom line of the box is calculated and
     * distances are taken from the corresponding point in the pcl.
     *
     * @param bboxes vector with bounding boxes. Each box is one vector
     * @param point_cloud ZED point-cloud
     */
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
    /**
     * Takes vector of cv:Mat images and inferences them with ONNX.
     *
     * @param rgb_cv_images vector of cv::Mat to infer
     * @return vector with bounding boxes.
     */
    auto Boxes = ZedInference::Detector.inference(rgb_cv_images);

    return Boxes;
}

void ZedInference::zed_acquisition(sl::Camera &zed, sl::Mat &img, sl::Mat &pcl, cv::Mat &cv_img, sl::Timestamp &ts, bool &running) {
    /**
    * Function for grabbing images for each camera. This can be run in a thread.
    * Takes a bunch of references and fills them with data. If run in a thread this will happen async.
    *
    * This code heavily leans on official Zed code:
    * https://github.com/stereolabs/zed-sdk/blob/master/depth%20sensing/multi%20camera/cpp/src/main.cpp#L119
    * at the time of writing under the MIT License
    */

    std::cout << "Camera thread opened" << std::endl;
    while (running) {
        // grab current images and compute depth
        if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
            // Rettieve image
            zed.retrieveImage(img, sl::VIEW::LEFT, sl::MEM::CPU);
            // to cv Mat
            cv_img = slMat2cvMat(img);
            // retrieve Pointcloud
            zed.retrieveMeasure(pcl, sl::MEASURE::XYZRGBA);
            ts = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE);
        }
        else {
            std::cout << "Could not grab image" << std::endl;
        }
        sl::sleep_ms(2); // This comes from the official ZED code. Why sleep?
    }
    zed.close();
}


void ZedInference::visualizeDetections(const cv::Mat& inputImage, const std::vector<std::vector<float>> &bboxes, const std::vector<std::vector<float>> &distances, const std::string &cam) {
    /**
     * Visualizing the detected bounding boxes.
     * "Paint" them on the cv:Mat and then dispalying it.
     *
     * @param inputImage Input image cv::Mat
     * @param bboxes vector of bboxes.
     * @param distances calculated distances for each bbox.
     * @param cam camera name to visualize detections with different window names.
     */
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
