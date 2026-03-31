cmake_minimum_required(VERSION 3.20)

# Allow FetchContent projects with old cmake_minimum_required
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "")

project(lmcache-server LANGUAGES C CXX CUDA ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures: Ampere(80,86), Ada(89), Hopper(90)
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
set(TORCH_CUDA_ARCH_LIST "8.0 8.6 8.9 9.0" CACHE STRING "" FORCE)

# ============================================================================
# Dependencies
# ============================================================================

# 1. CUDAToolkit
find_package(CUDAToolkit REQUIRED)

# 2. Torch (libtorch) — found via Python-installed torch
execute_process(
  COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _torch_result
)
if(NOT _torch_result EQUAL 0)
  message(FATAL_ERROR "Could not determine torch cmake prefix path. Is PyTorch installed?")
endif()
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)

# 3. ZeroMQ — use the system shared library + FetchContent for headers
include(FetchContent)

# We have libzmq.so.5 at runtime but no dev headers.
# Fetch cppzmq (header-only C++ binding) which also provides zmq.h.
# Link against the system libzmq.so.5 directly.
FetchContent_Declare(
  libzmq_headers
  GIT_REPOSITORY https://github.com/zeromq/libzmq.git
  GIT_TAG        v4.3.5
  GIT_SHALLOW    TRUE
)
# We only need the include directory, don't build libzmq
FetchContent_GetProperties(libzmq_headers)
if(NOT libzmq_headers_POPULATED)
  FetchContent_Populate(libzmq_headers)
endif()

FetchContent_Declare(
  cppzmq
  GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
  GIT_TAG        v4.10.0
  GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(cppzmq)
if(NOT cppzmq_POPULATED)
  FetchContent_Populate(cppzmq)
endif()

# Create an imported target for the system libzmq
add_library(libzmq_system SHARED IMPORTED)
set_target_properties(libzmq_system PROPERTIES
  IMPORTED_LOCATION "/usr/lib/x86_64-linux-gnu/libzmq.so.5"
  INTERFACE_INCLUDE_DIRECTORIES "${libzmq_headers_SOURCE_DIR}/include"
)

# 4. msgpack-cxx (header-only, C++ only — NOT msgpack-c which needs Boost)
FetchContent_Declare(
  msgpackcxx
  GIT_REPOSITORY https://github.com/msgpack/msgpack-c.git
  GIT_TAG        cpp-6.1.1
  GIT_SHALLOW    TRUE
)
# Don't use FetchContent_MakeAvailable — it pulls in Boost via CMakeLists.txt.
# Just fetch the source and create a header-only interface target.
FetchContent_GetProperties(msgpackcxx)
if(NOT msgpackcxx_POPULATED)
  FetchContent_Populate(msgpackcxx)
endif()
add_library(msgpack_header_only INTERFACE)
target_include_directories(msgpack_header_only INTERFACE
  ${msgpackcxx_SOURCE_DIR}/include
)
target_compile_definitions(msgpack_header_only INTERFACE MSGPACK_NO_BOOST)

# 5. BLAKE3 — C library
FetchContent_Declare(
  blake3
  GIT_REPOSITORY https://github.com/BLAKE3-team/BLAKE3.git
  GIT_TAG        1.8.2
  GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(blake3)
if(NOT blake3_POPULATED)
  FetchContent_Populate(blake3)
endif()
# Build blake3 C library as a static lib
add_library(blake3_lib STATIC
  ${blake3_SOURCE_DIR}/c/blake3.c
  ${blake3_SOURCE_DIR}/c/blake3_dispatch.c
  ${blake3_SOURCE_DIR}/c/blake3_portable.c
)
# Add SIMD implementations if available
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_sources(blake3_lib PRIVATE
    ${blake3_SOURCE_DIR}/c/blake3_sse2_x86-64_unix.S
    ${blake3_SOURCE_DIR}/c/blake3_sse41_x86-64_unix.S
    ${blake3_SOURCE_DIR}/c/blake3_avx2_x86-64_unix.S
    ${blake3_SOURCE_DIR}/c/blake3_avx512_x86-64_unix.S
  )
endif()
target_include_directories(blake3_lib PUBLIC ${blake3_SOURCE_DIR}/c)

# ============================================================================
# Existing sources (compiled directly, no pybind11)
# ============================================================================

set(LMCACHE_CSRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

# ============================================================================
# Main target
# ============================================================================

add_executable(lmcache-server
  main.cpp
  types.cpp
  wire_protocol.cpp
  tensor_bridge.cpp
  gpu_context.cpp
  cache_engine.cpp
  mq_server.cpp
  l1_store.cpp
  token_hasher.cpp
  session_manager.cpp

  # Existing C++ sources reused directly
  ${LMCACHE_CSRC_DIR}/mem_kernels.cu
  ${LMCACHE_CSRC_DIR}/mp_mem_kernels.cu
  ${LMCACHE_CSRC_DIR}/storage_manager/bitmap.cpp
  ${LMCACHE_CSRC_DIR}/storage_manager/ttl_lock.cpp
)

target_include_directories(lmcache-server PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LMCACHE_CSRC_DIR}
  ${cppzmq_SOURCE_DIR}
  ${libzmq_headers_SOURCE_DIR}/include
)

target_link_libraries(lmcache-server PRIVATE
  ${TORCH_LIBRARIES}
  CUDA::cudart
  CUDA::cuda_driver
  libzmq_system
  blake3_lib
  msgpack_header_only
  pthread
)

# Torch requires position-independent code
set_target_properties(lmcache-server PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
  # Add rpath for nvidia pip packages (libnvJitLink etc.)
  BUILD_RPATH "/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib"
  INSTALL_RPATH "/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib"
)

# Help the linker find libnvJitLink from the nvidia pip package
target_link_directories(lmcache-server PRIVATE
  /usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib
)

# Suppress torch-generated warnings
target_compile_options(lmcache-server PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
)

# Enable readable symbol names in backtrace_symbols
target_link_options(lmcache-server PRIVATE -rdynamic)
