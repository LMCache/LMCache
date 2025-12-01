# LMCache Docker Images

This directory contains Dockerfiles for building different LMCache images. Each Dockerfile serves a specific use case depending on your needs.

## Available Dockerfiles

### 1. `Dockerfile` - Full Integration with vLLM

**Image**: `lmcache/vllm-openai:latest`

**Description**: The main Dockerfile that builds LMCache from source and integrates it with vLLM OpenAI server. This is the recommended image for production deployments with full feature support including Prefill-Decode Disaggregation (PD).

**Features**:
- ✅ LMCache built from source
- ✅ vLLM integration (nightly or stable)
- ✅ Full NIXL support for Prefill-Decode Disaggregation
- ✅ CUDA support
- ✅ Optimized multi-stage build

**Build Targets**:
- `image-build`: Builds with vLLM nightly and LMCache from source
- `image-release`: Uses stable vLLM release and LMCache from PyPI

**Usage**:

```bash
# Build with nightly vLLM
docker build \
  --build-arg CUDA_VERSION=12.8 \
  --build-arg UBUNTU_VERSION=24.04 \
  --target image-build \
  --tag lmcache/vllm-openai:latest \
  --file docker/Dockerfile .

# Build with stable releases
docker build \
  --build-arg CUDA_VERSION=12.8 \
  --build-arg UBUNTU_VERSION=24.04 \
  --target image-release \
  --tag lmcache/vllm-openai:latest \
  --file docker/Dockerfile .
```

**Run Example**:

```bash
export HF_TOKEN=<your_huggingface_token>

docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  lmcache/vllm-openai:latest \
  serve Qwen/Qwen3-0.6B \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

---

### 2. `Dockerfile.standalone` - LMCache Only

**Image**: `lmcache/standalone:latest`

**Description**: A standalone Docker image that builds and installs LMCache from source without vLLM. This will be useful when running LMCache in the standalone mode.

**Features**:
- ✅ LMCache built from source
- ✅ No vLLM dependency
- ✅ CUDA support

**Build Target**:
- `lmcache-final`: Final optimized image with LMCache installed

**Usage**:

```bash
docker build \
  --build-arg CUDA_VERSION=12.8 \
  --build-arg UBUNTU_VERSION=24.04 \
  --target lmcache-final \
  --tag lmcache/standalone:latest \
  --file docker/Dockerfile.standalone .
```

**Run Example**:

```bash
# Start an interactive shell
docker run --runtime nvidia --gpus all -it \
  lmcache/standalone:latest \
  /opt/venv/bin/python3 \
  -m lmcache.v1.multiprocess.server \
  --cpu-buffer-size 60 \
  --max-workers 4 \
  --port 6555
```

---

### 3. `Dockerfile.lightweight` - Quick Setup

**Image**: `lmcache/vllm-openai:lightweight`

**Description**: A lightweight image that extends the official vLLM image and installs LMCache from PyPI. This is the fastest way to get started but does not include NIXL support.

**Features**:
- ✅ Based on official `vllm/vllm-openai:latest` image
- ✅ LMCache installed from PyPI (latest release)
- ✅ Quick build time
- ✅ Small image size
- ❌ No NIXL support (no Prefill-Decode Disaggregation)

**Limitations**:
- Cannot use Prefill-Decode Disaggregation features

**Usage**:

```bash
docker build \
  --tag lmcache/vllm-openai:lightweight \
  --file docker/Dockerfile.lightweight .
```

**Run Example**:

```bash
export HF_TOKEN=<your_huggingface_token>

docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8000:8000 \
  --ipc=host \
  lmcache/vllm-openai:lightweight \
  serve Qwen/Qwen3-0.6B \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

---

## Which Dockerfile Should I Use?

### Use `Dockerfile` if you:
- Need full LMCache + vLLM integration
- Want Prefill-Decode Disaggregation support
- Are deploying to production
- Need the latest features built from source

### Use `Dockerfile.standalone` if you:
- Want LMCache without vLLM
- Need a clean LMCache installation for development
- Want to integrate LMCache with custom tools

### Use `Dockerfile.lightweight` if you:
- Prefer stable releases from PyPI
- Need fast build times

---

## Build Arguments

All Dockerfiles support the following build arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `CUDA_VERSION` | `12.8` | CUDA version to use |
| `UBUNTU_VERSION` | `24.04` | Ubuntu base version |
| `PYTHON_VERSION` | `3.12` | Python version |
| `max_jobs` | `2` | Max parallel jobs for build |
| `nvcc_threads` | `8` | Number of nvcc threads |
| `torch_cuda_arch_list` | `7.0 7.5 8.0 8.6 8.9 9.0 10.0+PTX` | CUDA architectures |

**Example with custom arguments**:

```bash
docker build \
  --build-arg CUDA_VERSION=12.4 \
  --build-arg max_jobs=4 \
  --build-arg nvcc_threads=16 \
  --target image-build \
  --tag lmcache/vllm-openai:cuda12.4 \
  --file docker/Dockerfile .
```

---

## Published Images

Pre-built images are available on Docker Hub:

- `lmcache/vllm-openai:latest` - Latest stable release with vLLM
- `lmcache/vllm-openai:{version}` - Specific version (e.g., `v0.1.0`)
- `lmcache/vllm-openai:lightweight` - Lightweight version
- `lmcache/standalone:latest` - Latest standalone release
- `lmcache/standalone:{version}` - Specific standalone version

```bash
# Pull pre-built images
docker pull lmcache/vllm-openai:latest
docker pull lmcache/standalone:latest
```

---

## Additional Resources

- [LMCache Documentation](https://docs.lmcache.ai/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Installation Guide](https://docs.lmcache.ai/getting_started/installation.html)
- [Docker Deployment Guide](https://docs.lmcache.ai/production/docker_deployment.html)

