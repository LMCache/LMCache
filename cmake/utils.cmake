macro (find_python_from_executable EXECUTABLE SUPPORTED_VERSIONS)
  file(REAL_PATH ${EXECUTABLE} EXECUTABLE)
  set(Python_EXECUTABLE ${EXECUTABLE})
  find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule)
  if (NOT Python_FOUND)
    message(FATAL_ERROR "Unable to find python matching: ${EXECUTABLE}.")
  endif()
  set(_VER "${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")
  set(_SUPPORTED_VERSIONS_LIST ${SUPPORTED_VERSIONS} ${ARGN})
  if (NOT _VER IN_LIST _SUPPORTED_VERSIONS_LIST)
    message(FATAL_ERROR
      "Python version (${_VER}) is not one of the supported versions: "
      "${_SUPPORTED_VERSIONS_LIST}.")
  endif()
  message(STATUS "Found python matching: ${EXECUTABLE}.")
endmacro()


function (run_python OUT EXPR ERR_MSG)
  execute_process(
    COMMAND
    "${Python_EXECUTABLE}" "-c" "${EXPR}"
    OUTPUT_VARIABLE PYTHON_OUT
    RESULT_VARIABLE PYTHON_ERROR_CODE
    ERROR_VARIABLE PYTHON_STDERR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT PYTHON_ERROR_CODE EQUAL 0)
    message(FATAL_ERROR "${ERR_MSG}: ${PYTHON_STDERR}")
  endif()
  set(${OUT} ${PYTHON_OUT} PARENT_SCOPE)
endfunction()

function (run_get_pybind OUT ERR_MSG)
  execute_process(
    COMMAND
    "${Python_EXECUTABLE}" "-m" "pybind11" "--cmakedir"
    OUTPUT_VARIABLE PYTHON_OUT
    RESULT_VARIABLE PYTHON_ERROR_CODE
    ERROR_VARIABLE PYTHON_STDERR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "Python Executable: ${Python_EXECUTABLE}")
  message(STATUS "Pybind11 Output: ${PYTHON_OUT}")
  message(STATUS "Pybind11 Error: ${PYTHON_STDERR}")

  if(NOT PYTHON_ERROR_CODE EQUAL 0)
    message(FATAL_ERROR "${ERR_MSG}: ${PYTHON_STDERR}")
  endif()
  set(${OUT} ${PYTHON_OUT} PARENT_SCOPE)
endfunction()

macro (append_cmake_prefix_path PKG EXPR)
  run_python(_PREFIX_PATH
    "import ${PKG}; print(${EXPR})" "Failed to locate ${PKG} path")
  list(APPEND CMAKE_PREFIX_PATH ${_PREFIX_PATH})
endmacro()

macro (get_pybind_path OUT)
  run_get_pybind(OUT
    "Failed to locate pybind11 path")
endmacro()

function (get_torch_gpu_compiler_flags OUT_GPU_FLAGS GPU_LANG)
  if (${GPU_LANG} STREQUAL "CUDA")
    #
    # Get common NVCC flags from torch.
    #
    run_python(GPU_FLAGS
      "from torch.utils.cpp_extension import COMMON_NVCC_FLAGS; print(';'.join(COMMON_NVCC_FLAGS))"
      "Failed to determine torch nvcc compiler flags")

    if (CUDA_VERSION VERSION_GREATER_EQUAL 11.8)
      list(APPEND GPU_FLAGS "-DENABLE_FP8")
    endif()
    if (CUDA_VERSION VERSION_GREATER_EQUAL 12.0)
      list(REMOVE_ITEM GPU_FLAGS
        "-D__CUDA_NO_HALF_OPERATORS__"
        "-D__CUDA_NO_HALF_CONVERSIONS__"
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__"
        "-D__CUDA_NO_HALF2_OPERATORS__")
    endif()
  endif()
  set(${OUT_GPU_FLAGS} ${GPU_FLAGS} PARENT_SCOPE)
endfunction()



function (define_gpu_extension_target GPU_MOD_NAME)
  cmake_parse_arguments(PARSE_ARGV 1
    GPU
    "WITH_SOABI"
    "DESTINATION;LANGUAGE;USE_SABI"
    "SOURCES;ARCHITECTURES;COMPILE_FLAGS;INCLUDE_DIRECTORIES;LIBRARIES")

  if (GPU_WITH_SOABI)
    set(GPU_WITH_SOABI WITH_SOABI)
  else()
    set(GPU_WITH_SOABI)
  endif()

  if (GPU_USE_SABI)
    Python_add_library(${GPU_MOD_NAME} MODULE USE_SABI ${GPU_USE_SABI} ${GPU_WITH_SOABI} "${GPU_SOURCES}")
  else()
    Python_add_library(${GPU_MOD_NAME} MODULE ${GPU_WITH_SOABI} "${GPU_SOURCES}")
  endif()


  if (GPU_ARCHITECTURES)
    set_target_properties(${GPU_MOD_NAME} PROPERTIES
      ${GPU_LANGUAGE}_ARCHITECTURES "${GPU_ARCHITECTURES}")
  endif()

  set_property(TARGET ${GPU_MOD_NAME} PROPERTY CXX_STANDARD 17)

  target_compile_options(${GPU_MOD_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:${GPU_LANGUAGE}>:${GPU_COMPILE_FLAGS}>)

  target_compile_definitions(${GPU_MOD_NAME} PRIVATE
    "-DTORCH_EXTENSION_NAME=${GPU_MOD_NAME}")

  target_include_directories(${GPU_MOD_NAME} PRIVATE csrc
    ${GPU_INCLUDE_DIRECTORIES})

  target_link_libraries(${GPU_MOD_NAME} PRIVATE torch ${GPU_LIBRARIES})

  # Don't use `TORCH_LIBRARIES` for CUDA since it pulls in a bunch of
  # dependencies that are not necessary and may not be installed.
  if (GPU_LANGUAGE STREQUAL "CUDA")
    target_link_libraries(${GPU_MOD_NAME} PRIVATE CUDA::cudart CUDA::cuda_driver)
  else()
    target_link_libraries(${GPU_MOD_NAME} PRIVATE ${TORCH_LIBRARIES})
  endif()

  install(TARGETS ${GPU_MOD_NAME} LIBRARY DESTINATION ${GPU_DESTINATION} COMPONENT ${GPU_MOD_NAME})
endfunction()