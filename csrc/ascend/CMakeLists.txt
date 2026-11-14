# SPDX-License-Identifier: Apache-2.0
#
# Ascend NPU MP native extension.
#
# Builds ``lmcache.npu_ops`` (host-side pybind module) together with
# ``libcache_kernels.so`` (Ascend C block-transfer kernels).  This project is
# driven by ``setup_extensions/ascend_cmake.py``; it is not part of the
# setuptools/torch extension flow used by the other backends.

cmake_minimum_required(VERSION 3.16.0)
project(npu_ops)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LMC_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}")
if (NOT GLIBCXX_USE_CXX11_ABI)
  set(GLIBCXX_USE_CXX11_ABI 0)
endif()
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI})

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")

# Device-kernel compilation inputs.
set(SOC_VERSION ${SOC_VERSION})
set(ARCH ${ARCH})
set(ASCEND_AICORE_ARCH ${ASCEND_AICORE_ARCH})

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Build type Release/Debug (default Release)" FORCE)
endif()

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ARCH_SUBDIR "aarch64-linux")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(ARCH_SUBDIR "x86_64-linux")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

add_subdirectory(kernels)

include(utils.cmake)
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LMCACHE_ROOT}/csrc
  ${pybind11_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_PATH}
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_PATH}/include
  ${ASCEND_CANN_PACKAGE_PATH}/include
  ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/ascendc/include
  ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/include
  ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/pkg_inc
  ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/include/experiment/platform
  ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/include/experiment/ascend_hal
)

set(NPU_OPS_SOURCES
  pybind.cpp
  mp_mem_kernels.cpp
  completion_recorder.cpp
  event_recorder.cpp
  mem_alloc.cpp
  managed_mem.cpp
)

pybind11_add_module(npu_ops ${NPU_OPS_SOURCES})

set(TORCH_LIBS_DIR "${TORCH_PATH}/lib")

target_link_options(npu_ops PRIVATE
  "-Wl,-rpath,$ORIGIN:$ORIGIN/lib"
  "-Wl,-rpath,${LMC_INSTALL_PATH}"
)
target_link_directories(
  npu_ops
  PRIVATE
  ${TORCH_LIBS_DIR}
  ${TORCH_NPU_PATH}/lib/
  ${ASCEND_CANN_PACKAGE_PATH}/lib64
  ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/devlib
)
target_link_libraries(
  npu_ops
  PUBLIC
  ${TORCH_LIBRARIES}
  libtorch_npu.so
  cache_kernels
  ascendcl
  platform
  ascend_hal
  tiling_api
  numa
)

install(TARGETS npu_ops cache_kernels DESTINATION ${LMC_INSTALL_PATH})
