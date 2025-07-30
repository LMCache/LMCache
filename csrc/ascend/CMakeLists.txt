# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.16.0)
# project information
project(c_ops)

set(CMAKE_CXX_STANDARD 17)
include(${CMAKE_CURRENT_LIST_DIR}/utils.cmake)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
set(PYTHON_SUPPORTED_VERSIONS "3.9" "3.10" "3.11")
find_package(pybind11 REQUIRED)

append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
set(LMC_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}")

set(SOC_VERSION ${SOC_VERSION})
set(ARCH ${ARCH})

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRINGS "Build type Release/Debug (default Release)" FORCE)
endif()

set(ASCEND_HOME_PATH ${ASCEND_CANN_PACKAGE_PATH}) 
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# ${KERNEL_FILES} are used to compile library, push files written by ascendc in ${KERNEL_FILES}.
# ref to cmake/npu.cmake ascendc_library, cmake/cpu.cmake add_library
file(GLOB KERNEL_FILES 
${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cpp)

message(STATUS "kernel files: ${KERNEL_FILES}")

# ascendc_library use to add kernel file to generate ascendc library
ascendc_library(ascend_kernels SHARED
    ${KERNEL_FILES}
)

message("TORCH_NPU_PATH is ${TORCH_NPU_PATH}")

file(GLOB SRC_FILES
${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

set(PYBIND11_SOURCE_FILE 
${CMAKE_CURRENT_SOURCE_DIR}/mem_kernels.cpp
${CMAKE_CURRENT_SOURCE_DIR}/managed_mem.cpp
${CMAKE_CURRENT_SOURCE_DIR}/../pybind.cpp
)

find_package(Torch REQUIRED)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${pybind11_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_PATH}
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_PATH}/include
  ${ASCEND_HOME_PATH}/include
  ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include
  ${ASCEND_HOME_PATH}/aarch64-linux/include/experiment/platform
  ${ASCEND_HOME_PATH}/aarch64-linux/include/experiment/ascend_hal
  ${ASCEND_HOME_PATH}/x86_64-linux/include/experiment/platform
  ${ASCEND_HOME_PATH}/x86_64-linux/include/experiment/ascend_hal
)


set(
  INCLUDES
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_PATH}/include
  ${ASCEND_HOME_PATH}/include
  ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include
  ${ASCEND_HOME_PATH}/aarch64-linux/include/experiment/platform
  ${ASCEND_HOME_PATH}/aarch64-linux/include/experiment/ascend_hal
)

set(PYMODULE_FILES 
    ${SRC_FILES}
    ${PYBIND11_SOURCE_FILE}
)

pybind11_add_module(c_ops ${PYMODULE_FILES})

message(STATUS "CMake: Adding -DUSE_ASCEND compile definition.")
target_compile_definitions(c_ops PRIVATE USE_ASCEND)

set(TORCH_NPU_LIBS_DIR "${TORCH_NPU_PATH}/lib")
set(ASCEND_CANN_LIBS_DIR "${ASCEND_HOME_PATH}/lib64")
set(TORCH_LIBS_DIR "${TORCH_PATH}/lib")


target_link_options(c_ops PRIVATE
  "-Wl,-rpath,$ORIGIN:$ORIGIN/lib"
  "-Wl,-rpath,${LMC_INSTALL_PATH}" 
)

target_link_directories(
  c_ops
  PRIVATE
  ${TORCH_LIBS_DIR}
  ${TORCH_NPU_PATH}/lib/
  ${ASCEND_HOME_PATH}/lib64
  ${ASCEND_DRIVER_PATH}/lib64/driver
)

target_link_libraries(
  c_ops
  PUBLIC
  ${TORCH_LIBRARIES}
  libtorch_npu.so
  ascend_kernels
  ascendcl
  platform
  ascend_hal
  tiling_api
)


install(TARGETS c_ops ascend_kernels DESTINATION ${LMC_INSTALL_PATH})
