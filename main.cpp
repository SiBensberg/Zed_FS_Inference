#include <iostream>

#include "zed_inference.h"

int main() {
    std::cout << "Starting ZED inference: \n \n" << std::endl;

    ZedInference zed_inf;

    zed_inf.run();


    return 0;
}
