# Formula Student Driverless inference with ZED and ONNX

## Installation

### Install Opencv

-`sudo apt install -y g++ cmake make git libgtk2.0-dev pkg-config`
-Download Source code from here https://github.com/opencv/opencv/releases
-`mkdir -p build && cd build`
-`cmake ../opencv`
-`make -j16`
-`sudo make install`

### Install ZED SDK

### Install CUDA & CUDNN
- https://developer.nvidia.com/cuda-toolkit
- https://developer.nvidia.com/cudnn

### Build ONNX Runtime

- Install CUDA and CUDNN
- clone onnxruntime https://github.com/microsoft/onnxruntime
  - Important: clone from release commit. E.g. `https://github.com/microsoft/onnxruntime/tree/v1.13.1`
- build with:
- ```./build.sh --use_cuda --cudnn_home <CUDNN HOME PATH> --cuda_home <CUDA PATH> --parallel --build_shared_lib --config=Release```
  - cudnn_home is probably something like: `/usr/lib/x86_64-linux-gnu`
  - CUDA path is probably something like: `/usr/local/cuda-11.7`
  - parallel makes compiling faster. If OOM occurs maybe exclude it.

For further information you can look here: https://onnxruntime.ai/docs/build/eps.html#cuda



