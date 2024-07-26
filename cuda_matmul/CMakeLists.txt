cmake_minimum_required(VERSION 3.16.0)

project(Cuda_learn)

set(CMAKE_CXX_STANDARD 17)

# user-defined configuration
find_package(Python 3.10 REQUIRED)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)

IF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    SET(ARCH aarch64)
ELSEIF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
    SET(ARCH x86_64)
ENDIF ()

if(USE_CXX11_ABI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1 -fPIC -o2 -std=c++17")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -o2 -std=c++17")
endif()


add_library(pybind11_lib SHARED pybind11.cpp)

execute_process(COMMAND python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_PATH
)
message("TORCH_PATH is ${TORCH_PATH}")

target_link_directories(pybind11_lib PRIVATE
  ${TORCH_PATH}/lib
)

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_PATH}/lib")

target_link_libraries(pybind11_lib PRIVATE
    ${TORCH_PYTHON_LIBRARY}
)

message("CMAKE_CURRENT_SOURCE_DIR is ${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(pybind11_lib PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${TORCH_PATH}/include
  ${TORCH_PATH}/include/torch/csrc/api/include
)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

execute_process(COMMAND python3 -m pybind11 --includes
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYBIND11_INC
)
string(REPLACE " " ";" PYBIND11_INC ${PYBIND11_INC})
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

install(TARGETS pybind11_lib DESTINATION ${CMAKE_INSTALL_PREFIX})
