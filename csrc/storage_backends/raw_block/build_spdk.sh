#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build script for liblmcache_spdk.so
#
# This script compiles the SPDK C++ implementation into a shared library
# and copies it to the lmcache Python package directory for immediate use.
#
# Usage:
#   ./build_spdk.sh [SPDK_ROOT] [DPDK_ROOT]
#
# Environment variables:
#   SPDK_ROOT   - SPDK installation directory (default: /opt/spdk)
#   DPDK_ROOT   - DPDK build directory (default: $SPDK_ROOT/dpdk)
#   NUM_JOBS    - Number of parallel build jobs (default: nproc)
#
# Examples:
#   ./build_spdk.sh                              # Use defaults or env vars
#   ./build_spdk.sh /home/user/spdk              # Specify SPDK root only
#   ./build_spdk.sh /home/user/spdk /home/user/dpdk  # Specify both
#   SPDK_ROOT=/opt/spdk ./build_spdk.sh          # Use env var

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_spdk"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Determine SPDK and DPDK roots
SPDK_ROOT="${1:-${SPDK_ROOT:-/opt/spdk}}"
DPDK_ROOT="${2:-${DPDK_ROOT:-${SPDK_ROOT}/dpdk}}"

# Validate inputs
if [ ! -d "$SPDK_ROOT" ]; then
    log_error "SPDK directory not found: $SPDK_ROOT"
    echo "Set SPDK_ROOT environment variable or pass as first argument."
    exit 1
fi

if [ ! -d "$DPDK_ROOT" ]; then
    log_error "DPDK directory not found: $DPDK_ROOT"
    echo "Set DPDK_ROOT environment variable or pass as second argument."
    exit 1
fi

# Check for required SPDK libraries
if [ ! -d "$SPDK_ROOT/build/lib" ]; then
    log_error "SPDK build/lib directory not found: $SPDK_ROOT/build/lib"
    echo "Please build SPDK first by running: cd $SPDK_ROOT && make"
    exit 1
fi

# Check for required DPDK libraries
if [ ! -d "$DPDK_ROOT/build/lib" ]; then
    log_error "DPDK build/lib directory not found: $DPDK_ROOT/build/lib"
    echo "Please build DPDK first by running: cd $DPDK_ROOT && make install T=x86_64-native-linuxapp"
    exit 1
fi

# Determine number of jobs
NUM_JOBS="${NUM_JOBS:-$(nproc 2>/dev/null || echo 4)}"

# Resolve lmcache package directory (parent of csrc/)
LMCACHE_PKG_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LIB_DEST="$LMCACHE_PKG_DIR/lmcache/v1/storage_backend/raw_block"

log_info "=== lmcache_spdk Build ==="
log_info "SPDK_ROOT   : $SPDK_ROOT"
log_info "DPDK_ROOT   : $DPDK_ROOT"
log_info "Build Dir   : $BUILD_DIR"
log_info "Dest Dir    : $LIB_DEST"
log_info "Jobs        : $NUM_JOBS"
echo ""

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Run CMake
log_info "Running CMake..."
cmake .. \
    -DSPDK_ROOT="$SPDK_ROOT" \
    -DDPDK_ROOT="$DPDK_ROOT" \
    -DCMAKE_BUILD_TYPE=Release

# Build
log_info "Building (jobs=$NUM_JOBS)..."
make -j"$NUM_JOBS" VERBOSE=0

# Check if build succeeded
if [ -f "$BUILD_DIR/liblmcache_spdk.so" ]; then
    log_info "Build successful!"
    echo ""
    log_info "Library location: $BUILD_DIR/liblmcache_spdk.so"
    echo ""

    # Show library info
    if command -v ldd &> /dev/null; then
        log_info "Library dependencies:"
        ldd "$BUILD_DIR/liblmcache_spdk.so" | head -20
        echo ""
    fi

    # Copy to lmcache package directory
    mkdir -p "$LIB_DEST"
    cp -f "$BUILD_DIR/liblmcache_spdk.so" "$LIB_DEST/"
    log_info "Copied liblmcache_spdk.so to $LIB_DEST/"
    echo ""
else
    log_error "Build failed! Check errors above."
    exit 1
fi
