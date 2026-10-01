# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22)
project(lmcache_compressed_record_probe LANGUAGES CXX)

set(LMCACHE_COMPAT_VENDOR "" CACHE STRING "NVIDIA or AMD")
set_property(CACHE LMCACHE_COMPAT_VENDOR PROPERTY STRINGS NVIDIA AMD)

add_executable(lmcache_compressed_record_probe record_probe.cpp)

if(LMCACHE_COMPAT_VENDOR STREQUAL "NVIDIA")
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  find_package(nvcomp 5.3 CONFIG REQUIRED COMPONENTS nvcomp)
  set_source_files_properties(record_probe.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(
    lmcache_compressed_record_probe
    PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  target_compile_definitions(lmcache_compressed_record_probe
                             PRIVATE LMCACHE_COMPAT_NVIDIA=1)
  target_link_libraries(lmcache_compressed_record_probe PRIVATE nvcomp::nvcomp
                                                                  CUDA::cudart)
elseif(LMCACHE_COMPAT_VENDOR STREQUAL "AMD")
  enable_language(HIP)
  find_package(hip CONFIG REQUIRED)
  find_package(hipcomp 2.3 CONFIG REQUIRED)
  set_source_files_properties(record_probe.cpp PROPERTIES LANGUAGE HIP)
  set_target_properties(
    lmcache_compressed_record_probe
    PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
  target_compile_definitions(lmcache_compressed_record_probe
                             PRIVATE LMCACHE_COMPAT_AMD=1)
  target_link_libraries(lmcache_compressed_record_probe PRIVATE hipcomp::hipcomp
                                                                  hip::host)
else()
  message(FATAL_ERROR "Set LMCACHE_COMPAT_VENDOR to NVIDIA or AMD")
endif()

include(CTest)
add_test(
  NAME lmcache_compressed_record_probe
  COMMAND
    lmcache_compressed_record_probe
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/raw-deflate-v1.hex
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/gzip-v1.hex)
