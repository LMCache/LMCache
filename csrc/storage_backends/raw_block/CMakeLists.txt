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
# Environment variables (override CMake defaults):
#   SPDK_ROOT   - SPDK installation directory (default: /opt/spdk)
#   DPDK_ROOT   - DPDK build directory (default: $SPDK_ROOT/dpdk)
#   LMCACHE_SPDK_INSTALL_PREFIX - Install destination (default: /usr/local)

cmake_minimum_required(VERSION 3.14)
project(lmcache_spdk LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Set default SPDK and DPDK roots from environment or use defaults
if(NOT DEFINED ENV{SPDK_ROOT})
    set(SPDK_ROOT "$ENV{SPDK_ROOT}" CACHE PATH "SPDK root directory")
else()
    set(SPDK_ROOT "$ENV{SPDK_ROOT}" CACHE PATH "SPDK root directory")
endif()

if(NOT DEFINED ENV{DPDK_ROOT})
    set(DPDK_ROOT "$ENV{DPDK_ROOT}" CACHE PATH "DPDK build directory")
else()
    set(DPDK_ROOT "$ENV{DPDK_ROOT}" CACHE PATH "DPDK build directory")
endif()

# Fallback defaults if not set
if(NOT SPDK_ROOT OR SPDK_ROOT STREQUAL "")
    set(SPDK_ROOT "/opt/spdk" CACHE PATH "SPDK root directory" FORCE)
endif()
if(NOT DPDK_ROOT OR DPDK_ROOT STREQUAL "")
    set(DPDK_ROOT "${SPDK_ROOT}/dpdk" CACHE PATH "DPDK build directory" FORCE)
endif()

# Install prefix
set(LMCACHE_SPDK_INSTALL_PREFIX "$ENV{LMCACHE_SPDK_INSTALL_PREFIX}" CACHE PATH "Install destination")
if(NOT LMCACHE_SPDK_INSTALL_PREFIX OR LMCACHE_SPDK_INSTALL_PREFIX STREQUAL "")
    set(LMCACHE_SPDK_INSTALL_PREFIX "/usr/local" CACHE PATH "Install destination" FORCE)
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

# Find isa-l libraries (required by SPDK crypto/accel modules)
# These provide low-level AES, CRC, and compression intrinsics
set(ISAL_ROOT "${SPDK_ROOT}" CACHE PATH "isa-l root directory (inside SPDK tree)")
set(ISAL_LIB "${ISAL_ROOT}/isa-l/.libs/libisal.a" CACHE FILEPATH "isa-l static library")
set(ISAL_CRYPTO_LIB "${ISAL_ROOT}/isa-l-crypto/.libs/libisal_crypto.a" CACHE FILEPATH "isa-l-crypto static library")

if(NOT EXISTS "${ISAL_LIB}")
    message(FATAL_ERROR "isa-l library not found: ${ISAL_LIB}")
endif()
if(NOT EXISTS "${ISAL_CRYPTO_LIB}")
    message(FATAL_ERROR "isa-l-crypto library not found: ${ISAL_CRYPTO_LIB}")
endif()

message(STATUS "isa-l library: ${ISAL_LIB}")
message(STATUS "isa-l-crypto library: ${ISAL_CRYPTO_LIB}")

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

target_link_libraries(lmcache_spdk PRIVATE
    -Wl,--whole-archive
    ${SPDK_LIBS}
    ${DPDK_LIBS}
    -Wl,--no-whole-archive
    ${ISAL_LIB}
    ${ISAL_CRYPTO_LIB}
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
message(STATUS "  Install Prefix  : ${LMCACHE_SPDK_INSTALL_PREFIX}")
message(STATUS "========================================")
message(STATUS "")
