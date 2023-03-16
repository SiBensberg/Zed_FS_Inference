cmake_minimum_required(VERSION 3.24)
project(ZED_inference)

set(CMAKE_CXX_STANDARD 14)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS -O0)
else()
    set(CMAKE_CXX_FLAGS -O0)
endif()

add_executable(ZED_inference main.cpp zed_inference.cpp zed_inference.h slmat_to_cvmat.cpp slmat_to_cvmat.h object_detector.h object_detector.cpp)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
#add_executable( DisplayImage DisplayImage.cpp )
#target_link_libraries( DisplayImage ${OpenCV_LIBS} )

find_package(ZED 3 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)

IF(NOT WIN32)
    add_definitions(-Wno-format-extra-args)
    SET(SPECIAL_OS_LIBS "pthread" "X11")
ENDIF()

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${ZED_LIBRARY_DIR})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

FILE(GLOB_RECURSE SRC_FILES src/*.cpp)
FILE(GLOB_RECURSE HDR_FILES include/*.hpp)



#ADD_EXECUTABLE(${PROJECT_NAME} ${HDR_FILES} ${SRC_FILES})

#target_link_libraries(ZED_inference stdc++fs -lstdc++fs)

#TARGET_LINK_LIBRARIES(ZED_inference ${ZED_LIBRARIES})


if (LINK_SHARED_ZED)
    SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
else()
    SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()

# ONNX
set(ONNXRUNTIME_ROOT_PATH /mnt/Ubuntu_01/onnxruntime-1.13.1)
set(ONNXRUNTIME_INCLUDE_DIRS
        ${ONNXRUNTIME_ROOT_PATH}/onnxruntime/include/onnxruntime
        ${ONNXRUNTIME_ROOT_PATH}/onnxruntime
        ${ONNXRUNTIME_ROOT_PATH}/onnxruntime/include/onnxruntime/core/session
        )

set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/onnxruntime/build/Linux/Release/libonnxruntime.so)
#/mnt/Ubuntu_01/onnxruntime-1.13.1/onnxruntime/build/Linux/Release/libonnxruntime.so

target_include_directories(ZED_inference PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
target_link_libraries(ZED_inference PRIVATE ${ONNXRUNTIME_LIB} ${ZED_LIBRARIES} stdc++fs -lstdc++fs ${ZED_LIBS} ${OpenCV_LIBRARIES})

#TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${ZED_LIBS} ${OpenCV_LIBRARIES})
