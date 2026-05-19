# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.17)

project(lmcache_mp_cpp LANGUAGES CXX)

option(LMCACHE_BUILD_NATIVE_MP "Build the native LMCache MP server" ON)
option(LMCACHE_ENABLE_NIXL "Enable NIXL-backed native MP L2 adapter" OFF)
option(LMCACHE_ENABLE_CUDA "Enable CUDA-aware native MP transfers" OFF)
option(LMCACHE_ENABLE_TESTS "Build native MP C++ tests" OFF)
option(LMCACHE_ENABLE_TSAN "Build native MP C++ targets with ThreadSanitizer" OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(LMCACHE_ENABLE_TSAN)
  if(MSVC)
    message(FATAL_ERROR "LMCACHE_ENABLE_TSAN is not supported with MSVC")
  endif()
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_compile_options(-fno-pie)
    add_link_options(-no-pie)
  endif()
endif()

find_program(LLVM_CONFIG_EXECUTABLE NAMES llvm-config-18 llvm-config)
if(LLVM_CONFIG_EXECUTABLE)
  execute_process(
    COMMAND "${LLVM_CONFIG_EXECUTABLE}" --includedir
    OUTPUT_VARIABLE LLVM_CONFIG_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(
    COMMAND "${LLVM_CONFIG_EXECUTABLE}" --libdir
    OUTPUT_VARIABLE LLVM_CONFIG_LIBRARY_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

find_path(
  LLVM_BLAKE3_INCLUDE_DIR
  NAMES llvm-c-18/llvm-c/blake3.h llvm-c/blake3.h
  HINTS
    "${LLVM_CONFIG_INCLUDE_DIR}"
    "/usr/include"
    "/usr/local/include")
find_library(
  LLVM_BLAKE3_LIBRARY
  NAMES LLVM-18 LLVM
  HINTS
    "${LLVM_CONFIG_LIBRARY_DIR}"
    "/usr/lib/llvm-18/lib"
    "/usr/lib/x86_64-linux-gnu"
    "/usr/lib64")

if(NOT LLVM_BLAKE3_INCLUDE_DIR OR NOT LLVM_BLAKE3_LIBRARY)
  message(FATAL_ERROR "Could not find LLVM BLAKE3 headers/library")
endif()

set(
  LMCACHE_MP_CPP_CACHE_SOURCES
  src/cuda_layout.cpp
  src/cuda_metadata.cpp
  src/ipc_key.cpp
  src/kv_metadata.cpp
  src/l2_adapter.cpp
  src/native_http_utils.cpp
  src/native_status.cpp
  src/tiered_cache.cpp
  src/key_compat.cpp
  src/protocol_compat.cpp)

add_library(lmcache_mp_cpp_cache STATIC ${LMCACHE_MP_CPP_CACHE_SOURCES})
target_include_directories(lmcache_mp_cpp_cache PUBLIC include)
target_include_directories(lmcache_mp_cpp_cache PRIVATE "${LLVM_BLAKE3_INCLUDE_DIR}")
target_link_libraries(lmcache_mp_cpp_cache PUBLIC "${LLVM_BLAKE3_LIBRARY}")

add_library(lmcache_mp_cpp SHARED ${LMCACHE_MP_CPP_CACHE_SOURCES})
target_include_directories(lmcache_mp_cpp PUBLIC include)
target_include_directories(lmcache_mp_cpp PRIVATE "${LLVM_BLAKE3_INCLUDE_DIR}")
target_link_libraries(lmcache_mp_cpp PUBLIC "${LLVM_BLAKE3_LIBRARY}")

if(LMCACHE_BUILD_NATIVE_MP)
  find_path(
    ZMQ_INCLUDE_DIR
    NAMES zmq.h
    HINTS
      "$ENV{CONDA_PREFIX}/include"
      "/home/cxlsol/miniconda3/pkgs/zeromq-4.3.5-hb0a5e54_1/include"
      "/usr/include"
      "/usr/local/include")
  find_library(
    ZMQ_LIBRARY
    NAMES zmq libzmq.so.5
    HINTS
      "/usr/lib/x86_64-linux-gnu"
      "/usr/local/lib")

  if(NOT ZMQ_INCLUDE_DIR OR NOT ZMQ_LIBRARY)
    message(FATAL_ERROR "Could not find libzmq headers/library")
  endif()

  if(LMCACHE_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
      set(CMAKE_CUDA_ARCHITECTURES 80 90)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()

  add_executable(
    lmcache-mp-server-native
    src/cuda_runtime_state.cpp
    src/cuda_transfer.cpp
    src/http_server.cpp
    src/main.cpp
    src/msgpack_lite.cpp
    src/native_server_http_status.cpp
    src/native_server_requests.cpp
    src/native_server.cpp)
  if(LMCACHE_ENABLE_CUDA)
    target_sources(lmcache-mp-server-native PRIVATE src/native_transfer_kernel.cu)
  endif()
  target_include_directories(
    lmcache-mp-server-native
    PRIVATE include "${ZMQ_INCLUDE_DIR}")
  target_link_libraries(
    lmcache-mp-server-native
    PRIVATE lmcache_mp_cpp_cache "${ZMQ_LIBRARY}" pthread)
  if(LMCACHE_ENABLE_CUDA)
    target_link_libraries(lmcache-mp-server-native PRIVATE CUDA::cudart)
  endif()
  target_compile_definitions(
    lmcache-mp-server-native
    PRIVATE
      LMCACHE_ENABLE_NIXL=$<BOOL:${LMCACHE_ENABLE_NIXL}>
      LMCACHE_ENABLE_CUDA=$<BOOL:${LMCACHE_ENABLE_CUDA}>)
  install(TARGETS lmcache-mp-server-native RUNTIME DESTINATION bin)
endif()

if(LMCACHE_ENABLE_TESTS)
  enable_testing()
endif()
