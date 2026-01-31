# LMCache Project Guide for AI Agents

This document provides essential information for AI coding agents working on the LMCache project.

## Project Overview

LMCache is an LLM serving engine extension designed to reduce Time To First Token (TTFT) and increase throughput, especially under long-context scenarios. It stores KV caches across different storage tiers (GPU, CPU, Disk, S3, Redis, etc.) and enables reuse of KV caches across different serving engine instances.

Key integrations:
- **vLLM**: Primary integration via KV connector interface
- **SGLang**: Secondary integration for KV cache offloading

Key features:
- CPU KV cache offloading
- Disaggregated prefill
- P2P KV cache sharing
- Multiple storage backends (CPU, Disk, NIXL, Redis, S3, etc.)

## Technology Stack

- **Language**: Python 3.10-3.13
- **Deep Learning**: PyTorch 2.8.0+
- **CUDA**: CUDA 12.x
- **C++ Extensions**: Custom CUDA kernels for memory operations and compression
- **Build System**: setuptools with pyproject.toml
- **Documentation**: Sphinx (reStructuredText)
- **Platforms**: Linux NVIDIA GPU (CUDA), with ROCm/HIP support

## Project Structure

```
lmcache/
├── __init__.py              # Package root
├── _version.py              # Auto-generated version file
├── config.py                # Legacy configuration
├── logging.py               # Custom logging with color formatting
├── observability.py         # Metrics and stats collection
├── protocol.py              # Communication protocols
├── utils.py                 # Shared utility functions
├──
├── integration/             # Serving engine integrations
│   ├── vllm/               # vLLM integration adapters
│   │   ├── vllm_v1_adapter.py       # Main vLLM v1 adapter
│   │   ├── lmcache_connector_v1.py  # KV connector v1
│   │   └── utils.py
│   └── sglang/             # SGLang integration
│
├── server/                  # V0 server implementation (legacy)
│
├── storage_backend/         # V0 storage backends (legacy)
│   ├── evictor/            # Eviction policies (LRU)
│   └── serde/              # Serialization (CacheGen, etc.)
│
└── v1/                     # Main V1 implementation
    ├── cache_engine.py     # Core cache engine
    ├── config.py           # Configuration system
    ├── memory_management.py # Memory allocation and management
    ├── metadata.py         # Metadata structures
    ├── storage_manager.py  # Storage backend coordinator
    ├── token_database.py   # Token indexing and lookup
    ├──
    ├── cache_controller/   # Distributed cache controller
    ├── compute/            # Computation modules (blending, attention)
    ├── gpu_connector.py    # GPU memory connector interface
    ├── health_monitor/     # Health monitoring system
    ├── internal_api_server/ # Internal REST API
    ├── lookup_client/      # Remote cache lookup clients
    ├── multiprocess/       # Multi-process storage manager
    ├── offload_server/     # Offload server implementations
    ├── server/             # V1 standalone server
    ├── standalone/         # Standalone mode manager
    └── storage_backend/    # Storage backend implementations
        ├── connector/      # Storage connectors (Redis, S3, etc.)
        ├── cache_policy/   # Cache policies (LRU, LFU, FIFO, MRU)
        └── job_executor/   # Async job execution

csrc/                       # C++/CUDA extensions
├── pybind.cpp             # Python bindings
├── mem_kernels.cu         # Memory transfer kernels
├── cal_cdf.cu             # CDF calculation for compression
├── ac_enc.cu / ac_dec.cu  # Arithmetic coding for CacheGen
├── pos_kernels.cu         # Positional encoding kernels
├── mem_alloc.cpp          # Pinned memory allocation
├── utils.cpp              # Utility functions
└── storage_manager/       # Storage manager C++ extensions

tests/                      # Test suite
├── v1/                    # V1 specific tests
├── disagg/                # Disaggregated prefill tests
└── conftest.py            # Shared test fixtures

docs/                       # Sphinx documentation (reStructuredText)
examples/                   # Example usage code
docker/                     # Docker configurations
benchmarks/                 # Benchmarking tools
.buildkite/                 # BuildKite CI configuration
```

## Build and Installation

### Development Installation

```bash
# Install build dependencies first
pip install -r requirements/build.txt

# Install runtime dependencies
pip install -r requirements/common.txt

# Install with CUDA extensions (no build isolation)
pip install -e . --no-build-isolation
```

### Build Options

- `NO_CUDA_EXT=1`: Skip building CUDA extensions (for sdist)
- `BUILD_WITH_HIP=1`: Build with ROCm/HIP support instead of CUDA
- `ENABLE_CXX11_ABI=0/1`: Control C++11 ABI (default: 1)
- `MAX_JOBS=N`: Number of parallel compilation jobs
- `NVCC_THREADS=N`: Number of NVCC threads

### Docker Build

```bash
# Development image with vLLM nightly
docker build --target image-build -f docker/Dockerfile .

# Release image with stable releases
docker build --target image-release -f docker/Dockerfile .
```

## Testing

### Running Tests

```bash
# Install test dependencies
pip install -r requirements/test.txt

# Run non-CUDA unit tests (GitHub Actions style)
pytest --ignore=tests/disagg \
       --ignore=tests/v1/test_nixl_storage.py \
       --ignore=tests/v1/multiprocess/test_cache_server.py \
       --ignore=tests/v1/storage_backend/test_eic.py

# Run with coverage
pytest --cov=lmcache --cov-report=html --cov-report=xml

# Run benchmarks
pytest tests/benchmarks/
```

### Test Configuration

Key pytest fixtures in `tests/conftest.py`:
- `mock_redis`: Mocks Redis for testing
- `lmserver_v1_process`: Starts V1 LMCache server for integration tests
- `autorelease_v1`: Automatic cleanup of test resources
- `memory_allocator`: Shared 5GB memory allocator for tests

### CI/CD

- **GitHub Actions**: Code quality checks, unit tests (Python 3.10-3.13)
- **BuildKite**: Comprehensive testing with GPU, integration tests with vLLM

## Code Style Guidelines

### Required Headers

All Python and C++ files must include SPDX license identifier:

```python
# SPDX-License-Identifier: Apache-2.0
```

```cpp
// SPDX-License-Identifier: Apache-2.0
```

### Import Style

Imports must be organized with specific headings (enforced by isort):

```python
# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
import os
from typing import Optional

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

# Local
from .utils import helper_function
```

Configure `.isort.cfg`:
```ini
[settings]
profile=black
from_first=true
import_heading_future=Future
import_heading_stdlib=Standard
import_heading_thirdparty=Third Party
import_heading_firstparty=First Party
import_heading_localfolder=Local
```

### Code Quality Tools

```bash
# Install pre-commit hooks
pip install -r requirements/lint.txt
pre-commit install

# Run all checks manually
pre-commit run --all-files
```

Pre-commit hooks include:
- **SPDX header check**: Custom script `tools/check_spdx_header.py`
- **isort**: Import sorting
- **ruff**: Linting and formatting (line length: 88)
- **codespell**: Spell checking
- **clang-format**: C++/CUDA formatting
- **mypy**: Type checking (configured in pyproject.toml)

### Logging

Always use the project's logger:

```python
from lmcache.logging import init_logger

logger = init_logger(__name__)
logger.info("Message")
```

Log levels controlled via `LMCACHE_LOG_LEVEL` environment variable (DEBUG, INFO, WARNING, ERROR, CRITICAL).

## Configuration System

Configuration is defined in `lmcache/v1/config.py` using a declarative approach:

```python
_CONFIG_DEFINITIONS: dict[str, dict[str, Any]] = {
    "chunk_size": {"type": int, "default": 256, "env_converter": int},
    "local_cpu": {"type": bool, "default": True, "env_converter": _to_bool},
    # ...
}
```

Configuration can be set via:
1. YAML configuration file
2. Environment variables (prefixed with `LMCACHE_`)
3. Runtime configuration API

Key configuration categories:
- Storage: `local_cpu`, `local_disk`, `remote_url`, `max_local_cpu_size`
- Features: `use_layerwise`, `enable_blending`, `enable_p2p`, `enable_controller`
- P2P: `p2p_host`, `p2p_init_ports`, `p2p_lookup_ports`
- Controller: `enable_controller`, `controller_pull_url`

## Key Architecture Components

### Cache Engine (`LMCacheEngine`)

Main interface for storing and retrieving KV caches:

```python
from lmcache.v1.cache_engine import LMCacheEngine

# Store KV cache
engine.store(tokens, kv_cache)

# Retrieve KV cache
retrieved = engine.retrieve(tokens)
```

### Controller API

`lmcache_controller` exposes a REST API (default port 9000) for orchestration:
- `POST /move`
- `POST /compress`
- `POST /decompress`
- `POST /health`
- `POST /query_worker_info`
- `POST /check_finish`

### Storage Backends

Storage backends implement `StorageBackendInterface`:
- `LocalCPUBackend`: Local CPU RAM storage
- `LocalDiskBackend`: Local disk storage
- `RemoteBackend`: Remote LMCache server
- `P2PBackend`: Peer-to-peer sharing
- Various connectors: Redis, S3, NIXL, Infinistore, etc.

### GPU Connectors

Adapters for different serving engines:
- `VLLMPagedMemLayerwiseGPUConnector`: vLLM paged memory
- `VLLMBufferLayerwiseGPUConnector`: vLLM buffer-based
- `SGLangLayerwiseGPUConnector`: SGLang integration

### Memory Management

Key classes:
- `MemoryObj`: Represents a chunk of memory
- `MixedMemoryAllocator`: Manages pinned CPU memory
- `TensorMemoryObj`: Tensor-backed memory object

## Security Considerations

- **No secrets in code**: Use environment variables for credentials
- **Redis/S3 connectors**: Support authentication via URL/connection strings
- **P2P communication**: Configurable ports and hosts
- **Internal API server**: Restrict access in production

## Common Development Tasks

### Adding a New Storage Backend

1. Create connector in `lmcache/v1/storage_backend/connector/`
2. Implement `BaseStorageConnector` interface
3. Add adapter if needed (see `redis_adapter.py` as example)
4. Register in `CreateStorageBackends` factory
5. Add tests in `tests/v1/storage_backend/`

### Adding Configuration Options

1. Add entry to `_CONFIG_DEFINITIONS` in `lmcache/v1/config.py`
2. Specify type, default value, and env_converter
3. Add alias if replacing deprecated option
4. Update documentation

### Adding CUDA Kernels

1. Add `.cu` file in `csrc/`
2. Declare in `.cuh` header
3. Add binding in `csrc/pybind.cpp`
4. Add Python wrapper in `lmcache/`
5. Update `setup.py` if new files added

## Debugging Tips

- Enable debug logging: `export LMCACHE_LOG_LEVEL=DEBUG`
- Use NVTX annotations: Already instrumented with `_lmcache_nvtx_annotate`
- Check stats: `LMCStatsMonitor` provides runtime metrics
- Memory issues: Look for `PinMonitor` and allocator stats
- Test isolation: Use `autorelease_v1` fixture in tests

## Resources

- **Documentation**: https://docs.lmcache.ai/
- **Issues**: https://github.com/LMCache/LMCache/issues
- **Community**: [Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite)
- **Blogs**: https://blog.lmcache.ai/
