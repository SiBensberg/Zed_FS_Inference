//
// This code is from stereolabs github https://github.com/stereolabs/zed-opencv
// and at the point of writing under the MIT License
//
#ifndef ZED_INFERENCE_SLMAT_TO_CVMAT_H
#define ZED_INFERENCE_SLMAT_TO_CVMAT_H

#include <sl/Camera.hpp>
#include <opencv2/core/mat.hpp>

int getOCVtype(sl::MAT_TYPE type);

cv::Mat slMat2cvMat(sl::Mat& input);

#endif //ZED_INFERENCE_SLMAT_TO_CVMAT_H
