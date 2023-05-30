#include <iostream>

#include "zed_inference.h"

int main() {
    std::cout << "Starting ZED inference: \n \n" << std::endl;

    // Create ZedInference Class. Inference Session will be automatically initialized.
    ZedInference zed_inf;

    // Run cameras and inference
    zed_inf.run();


    return 0;
}
