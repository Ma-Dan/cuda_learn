cmake_minimum_required(VERSION 3.16.0)

project(Cuda_learn)

set(CMAKE_CXX_STANDARD 17)

# user-defined configuration
find_package(Python 3.10 REQUIRED)
find_package(CUDA QUIET REQUIRED)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)

IF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    SET(ARCH aarch64)
ELSEIF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
    SET(ARCH x86_64)
ENDIF ()

message("CMAKE_CURRENT_SOURCE_DIR is ${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

execute_process(COMMAND python3 -m pybind11 --includes
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYBIND11_INC
)
string(REPLACE " " ";" PYBIND11_INC ${PYBIND11_INC})

execute_process(COMMAND python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_PATH
)
message("TORCH_PATH is ${TORCH_PATH}")

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_PATH}/lib")

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -g -O3 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86
    )

# Specify include directories
include_directories(
    kernels
    utility
    )

cuda_add_library(pybind11_lib SHARED pybind11.cpp matmul.cu)

target_include_directories(pybind11_lib PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${TORCH_PATH}/include
  ${TORCH_PATH}/include/torch/csrc/api/include
)

target_compile_options(pybind11_lib PRIVATE
  ${PYBIND11_INC}
#  -D_GLIBCXX_USE_CXX11_ABI=0
#  -O2
#  -std=c++17
)

set(PYBIND11_SUFFIX ".cpython-310-x86_64-linux-gnu.so")
set_target_properties(pybind11_lib PROPERTIES
  OUTPUT_NAME cuda_learn${PYBIND11_SUFFIX}
  PREFIX "" SUFFIX ""
)

target_link_directories(pybind11_lib PRIVATE
  ${TORCH_PATH}/lib
  /usr/local/cuda/lib64 /opt/cuda/lib64 $(CUDA_PATH)/targets/x86_64-linux/lib
)

target_link_libraries(pybind11_lib
    ${TORCH_PYTHON_LIBRARY}
    cublas culibos cudart cublasLt
)

install(TARGETS pybind11_lib DESTINATION ${CMAKE_INSTALL_PREFIX})
