# SPDX-License-Identifier: Apache-2.0
#
# CMake build script for liblmcache_spdk.so
#
# This script compiles the SPDK C++ implementation into a shared library
# that can be loaded by Python via ctypes (FFI).
#
# Usage:
#   mkdir -p build && cd build
#   cmake .. -DSPDK_ROOT=/path/to/spdk -DDPDK_ROOT=/path/to/spdk/dpdk
#   make -j$(nproc)
#
# SPDK/DPDK roots are passed via -DSPDK_ROOT/-DDPDK_ROOT (see build_spdk.sh);
# conventional defaults are used when CMake is invoked directly.

cmake_minimum_required(VERSION 3.14)
project(lmcache_spdk LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# SPDK/DPDK roots are supplied by build_spdk.sh via -DSPDK_ROOT/-DDPDK_ROOT.
# Fall back to conventional defaults when CMake is invoked directly.
if(NOT SPDK_ROOT OR SPDK_ROOT STREQUAL "")
    set(SPDK_ROOT "/opt/spdk" CACHE PATH "SPDK root directory" FORCE)
endif()
if(NOT DPDK_ROOT OR DPDK_ROOT STREQUAL "")
    set(DPDK_ROOT "${SPDK_ROOT}/dpdk" CACHE PATH "DPDK build directory" FORCE)
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build shared library by default
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries")

# Output directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# ---------------------------------------------------------------------------
# Find SPDK and DPDK
# ---------------------------------------------------------------------------

# SPDK include and library directories
set(SPDK_INCLUDE_DIR "${SPDK_ROOT}/build/include")
set(SPDK_LIB_DIR "${SPDK_ROOT}/build/lib")

# DPDK include and library directories
set(DPDK_INCLUDE_DIR "${DPDK_ROOT}/build/include")
set(DPDK_LIB_DIR "${DPDK_ROOT}/build/lib")

# Verify directories exist
if(NOT EXISTS "${SPDK_INCLUDE_DIR}")
    message(FATAL_ERROR "SPDK include directory not found: ${SPDK_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${SPDK_LIB_DIR}")
    message(FATAL_ERROR "SPDK library directory not found: ${SPDK_LIB_DIR}")
endif()
if(NOT EXISTS "${DPDK_INCLUDE_DIR}")
    message(FATAL_ERROR "DPDK include directory not found: ${DPDK_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${DPDK_LIB_DIR}")
    message(FATAL_ERROR "DPDK library directory not found: ${DPDK_LIB_DIR}")
endif()

message(STATUS "SPDK root: ${SPDK_ROOT}")
message(STATUS "SPDK include: ${SPDK_INCLUDE_DIR}")
message(STATUS "SPDK lib: ${SPDK_LIB_DIR}")
message(STATUS "DPDK root: ${DPDK_ROOT}")
message(STATUS "DPDK include: ${DPDK_INCLUDE_DIR}")
message(STATUS "DPDK lib: ${DPDK_LIB_DIR}")

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------

set(SPDK_SOURCES
    spdk.cpp
)

# ---------------------------------------------------------------------------
# Create shared library
# ---------------------------------------------------------------------------

add_library(lmcache_spdk SHARED ${SPDK_SOURCES})

# Include directories
target_include_directories(lmcache_spdk PRIVATE
    ${SPDK_INCLUDE_DIR}
    ${DPDK_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compiler flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|i[3-6]86)")
    target_compile_options(lmcache_spdk PRIVATE -msse4.1)
endif()

target_compile_options(lmcache_spdk PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
    -fPIC
)

# Link SPDK and DPDK libraries
# Collect all SPDK static libraries
file(GLOB SPDK_LIBS "${SPDK_LIB_DIR}/libspdk_*.a")
file(GLOB DPDK_LIBS "${DPDK_LIB_DIR}/librte_*.a")

# Exclude SPDK libraries that pull in unnecessary or unavailable dependencies:
# - libspdk_ut.a: Unit test framework (CU_* symbols not needed at runtime)
# - libspdk_ut_mock.a: Unit test mock library (__real_* interposition symbols)
# - libspdk_fuse_dispatcher.a: Requires libfuse (FUSE 2 API), system has FUSE 3
list(FILTER SPDK_LIBS EXCLUDE REGEX "libspdk_ut\\.a$")
list(FILTER SPDK_LIBS EXCLUDE REGEX "libspdk_ut_mock\\.a$")
list(FILTER SPDK_LIBS EXCLUDE REGEX "libspdk_fuse_dispatcher\\.a$")

# Find isa-l libraries (used by SPDK crypto/accel modules). FORCE so a
# reconfigure against a different SPDK_ROOT recomputes these paths instead of
# keeping stale values from a previous configure.
set(ISAL_ROOT "${SPDK_ROOT}" CACHE PATH "isa-l root directory (inside SPDK tree)" FORCE)
set(ISAL_LIB "${ISAL_ROOT}/isa-l/.libs/libisal.a" CACHE FILEPATH "isa-l static library" FORCE)
set(ISAL_CRYPTO_LIB "${ISAL_ROOT}/isa-l-crypto/.libs/libisal_crypto.a" CACHE FILEPATH "isa-l-crypto static library" FORCE)

# isa-l is optional: SPDK can build without its crypto/accel modules. Warn and
# skip when the static libraries are absent rather than failing the build.
set(ISAL_LIBS "")
if(EXISTS "${ISAL_LIB}")
    message(STATUS "isa-l library: ${ISAL_LIB}")
    list(APPEND ISAL_LIBS ${ISAL_LIB})
else()
    message(WARNING "isa-l library not found: ${ISAL_LIB}; building without isa-l")
endif()
if(EXISTS "${ISAL_CRYPTO_LIB}")
    message(STATUS "isa-l-crypto library: ${ISAL_CRYPTO_LIB}")
    list(APPEND ISAL_LIBS ${ISAL_CRYPTO_LIB})
else()
    message(WARNING "isa-l-crypto library not found: ${ISAL_CRYPTO_LIB}; building without isa-l-crypto")
endif()

# System libraries required by SPDK/DPDK
# Note: Order matters - libraries should be listed after the objects/libraries
# that depend on them due to how the linker resolves symbols.
set(SYSTEM_LIBS
    pthread
    dl
    rt
    numa
    keyutils
    aio           # Linux AIO (io_setup, io_submit, etc.)
    uuid          # UUID operations (uuid_generate, etc.)
    fuse3         # FUSE 3 (cuse_lowlevel_*, fuse_* functions)
    gcc_s
    crypto
    ssl
)

# RDMA libraries are only needed for the NVMe-oF RDMA transport. Link them only
# when the dev packages (librdmacm/libibverbs) are present, so the build still
# succeeds on machines that only use the TCP transport or lack RDMA libraries.
find_library(RDMACM_LIB NAMES rdmacm)
find_library(IBVERBS_LIB NAMES ibverbs)
if(RDMACM_LIB)
    list(APPEND SYSTEM_LIBS ${RDMACM_LIB})
    message(STATUS "RDMA communication manager: ${RDMACM_LIB}")
else()
    message(STATUS "rdmacm not found; building without RDMA transport support")
endif()
if(IBVERBS_LIB)
    list(APPEND SYSTEM_LIBS ${IBVERBS_LIB})
    message(STATUS "RDMA verbs: ${IBVERBS_LIB}")
else()
    message(STATUS "ibverbs not found; building without RDMA transport support")
endif()

target_link_libraries(lmcache_spdk PRIVATE
    -Wl,--whole-archive
    ${SPDK_LIBS}
    ${DPDK_LIBS}
    -Wl,--no-whole-archive
    ${ISAL_LIBS}
    ${SYSTEM_LIBS}
)

# Set library properties
set_target_properties(lmcache_spdk PROPERTIES
    OUTPUT_NAME "lmcache_spdk"
    SOVERSION "1"
    VERSION "1.0.0"
    POSITION_INDEPENDENT_CODE ON
)

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

install(TARGETS lmcache_spdk
    LIBRARY DESTINATION lib
)

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

message(STATUS "")
message(STATUS "=== lmcache_spdk Build Configuration ===")
message(STATUS "  C++ Standard    : ${CMAKE_CXX_STANDARD}")
message(STATUS "  SPDK Root       : ${SPDK_ROOT}")
message(STATUS "  DPDK Root       : ${DPDK_ROOT}")
message(STATUS "  Install Prefix  : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "========================================")
message(STATUS "")
