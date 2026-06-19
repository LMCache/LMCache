#!/bin/bash
# Build pure Python wheel for LMCache (no CUDA extensions)
# This works without GPU access

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELHOUSE="${PROJECT_DIR}/wheelhouse"

echo "Building pure Python wheel for LMCache..."
mkdir -p "$WHEELHOUSE"

# Build using pip wheel with no C extensions
cd "${PROJECT_DIR}/LMCache"

# Install build dependencies
pip install build wheel setuptools_scm

# Build wheel
pip wheel . \
    --no-deps \
    -w "$WHEELHOUSE" \
    --python-version cp312 \
    --only-binary :all: \
    2>&1 || {
        echo "Pure binary wheel failed, trying source build without extensions..."
        # Fallback: build with setup.py but skip extensions
        python setup.py bdist_wheel \
            --plat-name manylinux_2_28_x86_64 \
            --skip-build-ext \
            -d "$WHEELHOUSE" \
            2>&1 || {
                echo "Building pure source wheel..."
                # Last resort: just package the Python files
                pip wheel . \
                    --no-deps \
                    -w "$WHEELHOUSE" \
                    --no-build-isolation
            }
    }

echo ""
echo "=== Wheel output ==="
ls -la "$WHEELHOUSE/"
