cmake_minimum_required(VERSION 3.26)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

set(PYTHON_SUPPORTED_VERSIONS "3.10" "3.11" "3.12")
find_python_from_executable(${LMC_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")

find_program(NVCC_EXECUTABLE nvcc)


# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

# Import torch cmake configuration.
# Torch also imports CUDA (and partially HIP) languages with some customizations,
# so there is no need to do this explicitly with check_language/enable_language,
# etc.
find_package(Torch REQUIRED)

# Find pybind
#get_pybind_path(pybind11_DIR)
#find_package(pybind11 REQUIRED)

# Configure `LMC_GPU_LANG`
# Currently, only cuda is supoorted.
set(LMC_GPU_LANG "CUDA")

# Configure `LMC_EXT_SRC`
set(LMC_EXT_SRC
    "csrc/pybind.cpp"
    "csrc/mem_kernels.cu"
    "csrc/cal_cdf.cu"
    "csrc/ac_enc.cu"
    "csrc/ac_dec.cu")

# Configure `LMC_GPU_FLAGS`
get_torch_gpu_compiler_flags(LMC_GPU_FLAGS ${LMC_GPU_LANG})
if(NVCC_THREADS AND LMC_GPU_LANG STREQUAL "CUDA")
  list(APPEND LMC_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif()

# Configure `LMC_GPU_FLAGS`
# Nothing needs to be configured for cuda.


message(STATUS "Enabling C extension.")
define_gpu_extension_target(
  c_ops
  DESTINATION lmcache
  LANGUAGE ${LMC_GPU_LANG}
  SOURCES ${LMC_EXT_SRC}
  COMPILE_FLAGS ${LMC_GPU_FLAGS}
  ARCHITECTURES ${LMC_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR};${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)