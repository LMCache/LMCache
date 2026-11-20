include(ascendc_with_def.cmake)

# ascendc_library used to add kernel files to generate ascendc library
if (DEFINED ASCEND_AICORE_ARCH)
    message(STATUS "ASCEND_AICORE_ARCH Set: ${ASCEND_AICORE_ARCH} - use custom ascendc_library")

    ascendc_library_with_def(cache_kernels SHARED ${KERNEL_FILES})
    
    ascendc_compile_definitions(cache_kernels PRIVATE
        -DASCEND_AICORE_ARCH=${ASCEND_AICORE_ARCH}
    )
else()
    message(STATUS "Use default ascendc_library")
    ascendc_library(cache_kernels SHARED ${KERNEL_FILES})
endif()
