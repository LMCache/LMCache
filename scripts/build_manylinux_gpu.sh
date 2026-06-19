#!/bin/bash
# Build manylinux wheel for LMCache using GPU-enabled container
#
# Prerequisites:
# - Docker with NVIDIA Container Toolkit
# - GPU with CUDA support
#
# Usage:
#   bash scripts/build_manylinux_gpu.sh
#
# The wheel will be created in ./wheelhouse/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WHEELHOUSE="${PROJECT_DIR}/wheelhouse"

# Configuration
IMAGE_NAME="ghcr.io/efschu/lmcache-manylinux-builder-gpu"
PYTHON_VERSION="cp312"

# GPU architectures (comma-separated for torch)
# RTX 3080/3090: 8.6
# RTX 4090/A100: 8.9
# H100: 9.0
# Add multiple for compatibility
CUDA_ARCH_LIST="8.6;8.9;9.0"

echo "=============================================="
echo "LMCache Manylinux Wheel Builder"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Image:        ${IMAGE_NAME}"
echo "  Python:       ${PYTHON_VERSION}"
echo "  CUDA Archs:   ${CUDA_ARCH_LIST}"
echo "  Output:       ${WHEELHOUSE}"
echo ""

# Create wheelhouse directory
mkdir -p "${WHEELHOUSE}"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf "${PROJECT_DIR}/LMCache/build"
rm -rf "${PROJECT_DIR}/LMCache/dist"

echo ""
echo "Building wheel..."
echo "=============================================="

# Run build in GPU-enabled container
# Note: --security-opt apparmor=unconfined is required on systems with AppArmor
docker run --rm \
    --gpus all \
    --security-opt apparmor=unconfined \
    -v "${PROJECT_DIR}/LMCache:/lm" \
    -v "${WHEELHOUSE}:/whl" \
    "${IMAGE_NAME}" \
    bash -c "
        set -e
        cd /lm
        
        # Set CUDA architecture for RTX 3080 (8.6), RTX 4090/A100 (8.9), H100 (9.0)
        export TORCH_CUDA_ARCH_LIST='${CUDA_ARCH_LIST}'
        export ENABLE_CXX11_ABI=1
        export MAX_JOBS=4
        
        # Build the wheel
        /opt/python/${PYTHON_VERSION}-${PYTHON_VERSION}/bin/pip wheel . \
            --no-deps \
            -w /whl
    "

echo ""
echo "=============================================="
echo "Build complete!"
echo ""
echo "Output:"
ls -la "${WHEELHOUSE}/"

echo ""
echo "Wheel file: $(ls ${WHEELHOUSE}/*.whl)"
echo ""
echo "To install:"
echo "  pip install ${WHEELHOUSE}/*.whl --force-reinstall --no-deps"
echo ""
echo "=============================================="
