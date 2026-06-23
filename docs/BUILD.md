# Building LMCache from Source

This document describes how to build LMCache from source, including building wheels for distribution.

## Prerequisites

### System Requirements

- **GPU**: NVIDIA GPU with CUDA support (RTX 3080, A100, H100, etc.)
- **CUDA**: CUDA 12.x or later
- **Python**: 3.10, 3.11, 3.12, or 3.13
- **Docker**: With NVIDIA Container Toolkit for GPU support
- **Git**: For cloning the repository

### Software Requirements

```bash
# Docker with NVIDIA support
docker --version                    # Docker version 19.03+
nvidia-container-toolkit --version  # For GPU passthrough in containers

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## Build Methods

### Method 1: Build with Docker (Recommended for Wheels)

This method builds a portable manylinux wheel that can be distributed.

#### Step 1: Clone the Repository

```bash
git clone https://github.com/efschu/LMCache.git
cd LMCache
git checkout dev  # or specific branch
```

#### Step 2: Prepare the Build Environment

The build requires a GPU-enabled Docker container with CUDA support. Use the PyTorch manylinux builder image:

```bash
# Build the builder image (optional, already available at ghcr.io/efschu/lmcache-manylinux-builder-gpu)
docker build -f docker/Dockerfile.manylinux -t lmcache-builder .
```

#### Step 3: Build the Wheel

```bash
# Create wheelhouse directory
mkdir -p wheelhouse

# Run the build in a GPU-enabled container
docker run --rm \
    --gpus all \
    --security-opt apparmor=unconfined \
    -v $(pwd):/lm \
    -v $(pwd)/wheelhouse:/whl \
    ghcr.io/efschu/lmcache-manylinux-builder-gpu \
    bash -c '
        set -e
        cd /lm
        
        # Set CUDA architecture for your GPU
        # RTX 3080/3090: 8.6
        # RTX 4090/A100: 8.9
        # H100: 9.0
        export TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"
        export ENABLE_CXX11_ABI=1
        export MAX_JOBS=4
        
        # Build the wheel
        /opt/python/cp312-cp312/bin/pip wheel . \
            --no-deps \
            -w /whl
    '
```

#### Step 4: Verify the Build

```bash
# Check the wheelhouse
ls -la wheelhouse/

# The wheel filename format is:
# lmcache-{version}-cp{python_version}-cp{python_version}-linux_x86_64.whl
# Example: lmcache-0.4.8rc2.dev15-cp312-cp312-linux_x86_64.whl
```

### Method 2: Direct Installation

For development or local testing:

```bash
# Clone and install in editable mode
git clone https://github.com/efschu/LMCache.git
cd LMCache
pip install -e . --no-build-isolation
```

### Method 3: Build Specific Python Versions

To build wheels for multiple Python versions:

```bash
# For each Python version, run:
for pyver in cp310 cp311 cp312 cp313; do
    docker run --rm \
        --gpus all \
        --security-opt apparmor=unconfined \
        -v $(pwd):/lm \
        -v $(pwd)/wheelhouse:/whl \
        ghcr.io/efschu/lmcache-manylinux-builder-gpu \
        bash -c "
            set -e
            cd /lm
            export TORCH_CUDA_ARCH_LIST='8.6;8.9;9.0'
            export ENABLE_CXX11_ABI=1
            
            # Use specific Python version
            PYTHON=/opt/python/${pyver}-${pyver}/bin/python
            \$PYTHON -m pip wheel . --no-deps -w /whl
        "
done
```

## GPU Architecture Guide

Set `TORCH_CUDA_ARCH_LIST` based on your target GPU:

| GPU Model | Compute Capability | Architecture List |
|-----------|-------------------|-------------------|
| RTX 3080/3090 | 8.6 | `8.6` |
| RTX 4070/4080/4090 | 8.9 | `8.9` |
| A100 | 8.0 | `8.0;8.6` |
| A30/A40 | 8.0 | `8.0` |
| H100/H200 | 9.0 | `9.0` |
| L40/L40S | 8.9 | `8.9` |
| Multiple GPUs | - | `8.0;8.6;8.9;9.0` |

## Troubleshooting

### Error: "No such file or directory: 'docker'"

This occurs when cibuildwheel tries to use Docker inside the container but Docker isn't available. Use the direct pip wheel method instead.

### Error: "CUDA version mismatch"

Ensure the CUDA version in the container matches your PyTorch version:

```bash
# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Ensure container has matching CUDA
docker run --rm --gpus all ghcr.io/efschu/lmcache-manylinux-builder-gpu \
    nvcc --version
```

### Error: "Permission denied" with Apparmor

If you see AppArmor denials, ensure `--security-opt apparmor=unconfined` is set:

```bash
docker run --rm \
    --gpus all \
    --security-opt apparmor=unconfined \
    ...
```

### Build Timeout

For slow builds, increase the timeout:

```bash
# Set longer build timeout
export MAX_JOBS=2  # Reduce parallel jobs if memory constrained
```

## Installing the Built Wheel

```bash
# Install from local wheel
pip install wheelhouse/lmcache-*.whl --force-reinstall --no-deps

# Or upload to PyPI/GitHub Releases
twine upload wheelhouse/*.whl
```

## Building with Custom PyTorch Version

To use a specific PyTorch version in the build:

```bash
docker run --rm \
    --gpus all \
    --security-opt apparmor=unconfined \
    -v $(pwd):/lm \
    ghcr.io/efschu/lmcache-manylinux-builder-gpu \
    bash -c '
        set -e
        cd /lm
        
        # Install specific PyTorch version first
        /opt/python/cp312-cp312/bin/pip install \
            torch==2.5.0 \
            --index-url https://download.pytorch.org/whl/cu124
        
        export TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"
        export ENABLE_CXX11_ABI=1
        
        # Build wheel
        /opt/python/cp312-cp312/bin/pip wheel . --no-deps -w wheelhouse/
    '
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build Wheels

on:
  push:
    branches: [main, dev]
  release:
    types: [published]

jobs:
  build:
    runs-on: ubuntu-22.04
    container:
      image: ghcr.io/efschu/lmcache-manylinux-builder-gpu
      options: --gpus all --security-opt apparmor=unconfined
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Build wheels
        run: |
          mkdir -p wheelhouse
          export TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"
          export ENABLE_CXX11_ABI=1
          /opt/python/cp312-cp312/bin/pip wheel . --no-deps -w wheelhouse/
      
      - name: Upload wheels
        uses: actions/upload-artifact@v4
        with:
          name: wheels
          path: wheelhouse/*.whl
```

## Verification

After building, verify the wheel:

```bash
# Check wheel contents
unzip -l wheelhouse/lmcache-*.whl | head -30

# Verify native extensions are included
unzip -l wheelhouse/lmcache-*.whl | grep "\.so$"

# Test installation
pip install wheelhouse/lmcache-*.whl --force-reinstall --no-deps
python -c "import lmcache; print(lmcache.__version__)"
```
