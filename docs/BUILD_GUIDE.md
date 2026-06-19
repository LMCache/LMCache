# Build Guide: LMCache Engine-Driven Multi-Group Fork

This guide explains step-by-step how to build the LMCache wheel from source.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Method 1: Pre-built Wheel (Recommended)](#method-1-pre-built-wheel-recommended)
3. [Method 2: Build with Docker (GPU Required)](#method-2-build-with-docker-gpu-required)
4. [Method 3: Native Build (Development)](#method-3-native-build-development)
5. [Troubleshooting](#troubleshooting)
6. [Verification](#verification)

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GPU with 8GB VRAM | RTX 3080/3090, A100, H100 |
| RAM | 16GB | 32GB+ |
| Disk | 10GB free | 50GB+ SSD |

### Software Requirements

```bash
# Check your setup
docker --version
nvidia-smi
git --version
```

### Docker with NVIDIA Support

```bash
# Install NVIDIA Container Toolkit (Ubuntu)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### GitHub Account (for uploading)

```bash
# Generate a Personal Access Token at:
# https://github.com/settings/tokens

# Set it locally (optional, for uploading releases)
export GITHUB_TOKEN="your_token_here"
```

---

## Method 1: Pre-built Wheel (Recommended)

If you just want to use the pre-built wheel:

### Download

**Latest Release:** [v0.4.8rc2-dev15](https://github.com/efschu/LMCache/releases/tag/v0.4.8rc2-dev15)

```bash
# Download directly
curl -L -O https://github.com/efschu/LMCache/releases/download/v0.4.8rc2-dev15/lmcache-0.4.8rc2.dev15-cp312-cp312-linux_x86_64.whl
```

### Install

```bash
# Install the wheel
pip install lmcache-0.4.8rc2.dev15-cp312-cp312-linux_x86_64.whl --force-reinstall --no-deps

# Verify installation
python -c "import lmcache; print(lmcache.__version__)"
```

---

## Method 2: Build with Docker (GPU Required)

This is the recommended method for building a portable wheel.

### Step 1: Clone the Repository

```bash
git clone https://github.com/efschu/LMCache.git
cd LMCache
git checkout dev
```

### Step 2: Create Build Directory

```bash
mkdir -p wheelhouse
```

### Step 3: Run the Docker Build

**Important:** On systems with AppArmor (like Ubuntu), you need `--security-opt apparmor=unconfined`.

```bash
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

### Step 4: Verify the Build

```bash
# Check the wheel
ls -la wheelhouse/

# Should output something like:
# -rw-r--r-- 1 root root 12670427 ... lmcache-0.4.8rc2.dev15-cp312-cp312-linux_x86_64.whl
```

### Step 5: Install and Test

```bash
# Install locally
pip install wheelhouse/lmcache-*.whl --force-reinstall --no-deps

# Test import
python -c "import lmcache; print(lmcache.__version__)"
```

---

## Method 3: Native Build (Development)

For development or if you have a native CUDA environment.

### Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv lmcache-build
source lmcache-build/bin/activate

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install build dependencies
pip install setuptools wheel numpy msgspec
```

### Step 2: Clone and Build

```bash
git clone https://github.com/efschu/LMCache.git
cd LMCache
git checkout dev

# Set CUDA architecture
export TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"
export ENABLE_CXX11_ABI=1

# Build wheel
pip wheel . --no-deps -w wheelhouse/
```

---

## GPU Architecture Reference

Set `TORCH_CUDA_ARCH_LIST` based on your target GPU(s):

| GPU Model | Compute Capability | Architecture |
|-----------|-------------------|--------------|
| RTX 3060/3070 | 8.6 | `8.6` |
| RTX 3080/3090 | 8.6 | `8.6` |
| RTX 4070/4080 | 8.9 | `8.9` |
| RTX 4090 | 8.9 | `8.9` |
| A100 | 8.0 | `8.0` |
| A30/A40 | 8.0 | `8.0` |
| H100/H200 | 9.0 | `9.0` |
| L40/L40S | 8.9 | `8.9` |
| **Multiple GPUs** | - | `8.0;8.6;8.9;9.0` |

---

## Troubleshooting

### Error: "docker: permission denied"

```bash
# Add your user to the docker group
sudo usermod -aG docker $USER
newgrp docker

# Or run with sudo (not recommended for production)
sudo docker run ...
```

### Error: "No such file or directory: 'docker'" in container

This happens when the build tries to use nested Docker. Solution: Use the direct pip wheel method shown above.

### Error: "AppArmor denied" on Ubuntu

Always include `--security-opt apparmor=unconfined`:

```bash
docker run --rm \
    --gpus all \
    --security-opt apparmor=unconfined \
    ...
```

### Error: "CUDA version mismatch"

Ensure PyTorch CUDA version matches your container:

```bash
# In the container
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

### Error: "Permission denied" when writing wheelhouse

```bash
# Create directory with proper permissions
mkdir -p wheelhouse
chmod 777 wheelhouse
```

### Build is very slow

```bash
# Reduce parallel jobs if memory constrained
export MAX_JOBS=2

# Or use a faster GPU
export TORCH_CUDA_ARCH_LIST="8.9"  # Only RTX 4090/A100
```

---

## Verification

### Basic Import Test

```bash
python -c "
import lmcache
print(f'LMCache version: {lmcache.__version__}')

# Check native extensions
try:
    from lmcache import native_storage_ops
    print('native_storage_ops: OK')
except Exception as e:
    print(f'native_storage_ops: FAILED - {e}')

try:
    from lmcache import lmcache_fs
    print('lmcache_fs: OK')
except Exception as e:
    print(f'lmcache_fs: FAILED - {e}')

try:
    from lmcache import lmcache_redis
    print('lmcache_redis: OK')
except Exception as e:
    print(f'lmcache_redis: FAILED - {e}')
"
```

### Functional Test with vLLM

```bash
# Start LMCache server
CUDA_VISIBLE_DEVICES="" lmcache server \
  --max-workers 1 \
  --max-gpu-workers 0 \
  --chunk-size 1600 \
  --l1-size-gb 1 \
  --eviction-policy LRU \
  --port 6555 &

# Verify server is running
sleep 5
curl http://localhost:6555/health || echo "Server check failed"

# Kill server
pkill -f "lmcache server"
```

### Check GPU Memory Usage

```bash
# Before starting LMCache server
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# After starting server (should be ~same)
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# If server is using GPU, it will show increased memory
```

---

## Uploading to GitHub Releases

### Create a Release

```bash
# Set your GitHub token
export GITHUB_TOKEN="your_pat_token"

# Create release notes
cat > RELEASE_NOTES.md << 'EOF'
# Release Notes

## Changes
- Feature 1
- Feature 2

## Requirements
- Python 3.12
- CUDA 12.x
- PyTorch 2.5+
EOF

# Create the release
gh release create v0.4.8rc2-dev15 \
    --title "v0.4.8rc2-dev15" \
    --notes-file RELEASE_NOTES.md \
    --repo efschu/LMCache
```

### Upload the Wheel

```bash
# Upload wheel to release
gh release upload v0.4.8rc2-dev15 \
    wheelhouse/*.whl \
    --repo efschu/LMCache
```

---

## Complete Example: Full Build and Release

```bash
#!/bin/bash
set -e

REPO_DIR="/path/to/LMCache"
GITHUB_TOKEN="your_token"

cd "$REPO_DIR"

# 1. Pull latest changes
git checkout dev
git pull origin dev

# 2. Clean previous builds
rm -rf wheelhouse/ build/ dist/
mkdir -p wheelhouse

# 3. Build wheel
docker run --rm \
    --gpus all \
    --security-opt apparmor=unconfined \
    -v $(pwd):/lm \
    -v $(pwd)/wheelhouse:/whl \
    ghcr.io/efschu/lmcache-manylinux-builder-gpu \
    bash -c '
        export TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"
        export ENABLE_CXX11_ABI=1
        cd /lm
        /opt/python/cp312-cp312/bin/pip wheel . --no-deps -w /whl
    '

# 4. Get version
WHEEL=$(ls wheelhouse/*.whl | head -1)
VERSION=$(echo $WHEEL | grep -oP '\d+\.\d+\.\d+[^\-]*')

# 5. Create release and upload
gh release create "v${VERSION}" \
    --title "v${VERSION}" \
    --notes "See https://github.com/efschu/LMCache/blob/dev/docs/BUILD.md" \
    --repo efschu/LMCache

gh release upload "v${VERSION}" wheelhouse/*.whl --repo efschu/LMCache

echo "Build complete!"
echo "Wheel: $WHEEL"
echo "Release: https://github.com/efschu/LMCache/releases/tag/v${VERSION}"
```

---

## Support

- **Issues:** https://github.com/efschu/LMCache/issues
- **Discussions:** https://github.com/efschu/LMCache/discussions

---

## See Also

- [README.md](../README.md) - Project overview
- [BUILD.md](./BUILD.md) - Alternative build instructions
- [Lmcache_engine_driven_multigroup.md](../Lmcache_engine_driven_multigroup.md) - Design document
