#!/bin/bash
# Build manylinux wheel for LMCache engine-driven multi-group fork
# Requires Docker with NVIDIA GPU support

set -e

IMAGE_NAME="ghcr.io/efschu/lmcache-manylinux-builder"
CONTAINER_NAME="lmcache-manylinux-build"

echo "Building manylinux wheel for LMCache..."
echo "This will:"
echo "  1. Build the Docker image"
echo "  2. Run cibuildwheel inside container"
echo "  3. Output wheel to ./wheelhouse/"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Build Docker image
echo ""
echo "=== Step 1: Building Docker image ==="
docker build \
    -f "${PROJECT_DIR}/docker/Dockerfile.manylinux" \
    -t "$IMAGE_NAME" \
    "$PROJECT_DIR"

# Create wheelhouse directory
mkdir -p "${PROJECT_DIR}/wheelhouse"

# Run build in container
echo ""
echo "=== Step 2: Running cibuildwheel ==="
docker run --rm \
    --name "$CONTAINER_NAME" \
    -v "${PROJECT_DIR}:/io" \
    "$IMAGE_NAME" \
    bash -c "
        set -e
        cd /io/LMCache
        mkdir -p /io/wheelhouse
        
        # Install torch first
        /opt/python/cp312-cp312/bin/pip install \
            torch==2.11.0 \
            --index-url https://download.pytorch.org/whl/cu130
        
        # Install dependencies
        /opt/python/cp312-cp312/bin/pip install msgspec numpy
        
        # Build wheel
        /opt/python/cp312-cp312/bin/pip wheel . \
            --no-deps \
            -w /io/wheelhouse \
            --python-version cp312 \
            --plat manylinux_2_28_x86_64 \
            --find-links https://download.pytorch.org/whl/cu130
    "

echo ""
echo "=== Step 3: Wheel output ==="
ls -la "${PROJECT_DIR}/wheelhouse/"

echo ""
echo "Build complete!"
echo "Wheel location: ${PROJECT_DIR}/wheelhouse/"
