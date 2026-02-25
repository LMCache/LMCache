# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LMCache is an LLM serving engine extension that reduces Time-To-First-Token (TTFT) and increases throughput for long-context scenarios by storing KV caches across distributed storage (GPU, CPU, Disk, S3). It integrates with vLLM and SGLang serving engines.

## Build & Development Commands

### Environment Setup
```bash
pip install -e .                    # Install in development mode
pip install -r requirements/test.txt    # Install test dependencies
pip install -r requirements/lint.txt    # Install linting tools
pre-commit install                  # Set up pre-commit hooks
```

### Building with CUDA Extensions
```bash
pip install -e .                    # Standard build with CUDA
NO_CUDA_EXT=1 pip install -e .      # CPU-only build (no CUDA extensions)
BUILD_WITH_HIP=1 pip install -e .   # ROCm/HIP build for AMD GPUs
```

### Running Tests
```bash
pytest                              # Run all tests
pytest tests/v1/                    # Run V1 API tests only
pytest tests/v1/storage_backend/    # Run storage backend tests
pytest -k "test_name"               # Run specific test by name
pytest --ignore=tests/disagg --ignore=tests/v1/multiprocess  # Skip GPU-intensive tests
```

### Code Quality
```bash
pre-commit run --all-files          # Run all linting/formatting checks
pre-commit run ruff --all-files     # Run ruff linter only
pre-commit run ruff-format --all-files  # Run ruff formatter only
```

Pre-commit hooks include: SPDX header check, isort, ruff (lint + format), codespell, clang-format, mypy.

## Architecture

### Core Components (lmcache/v1/)

- **cache_engine.py**: Main cache engine - converts GPU KV caches to MemoryObjs and coordinates storage
- **manager.py**: High-level cache manager interface for serving engines
- **gpu_connector.py**: Adapters for vLLM and SGLang GPU tensor formats
- **memory_management.py**: Memory allocation strategies (Tensor, Paged, Mixed, CuFile allocators)
- **storage_backend/**: Pluggable storage backends (CPU, Disk, Remote, P2P, GDS, NIXL)

### Data Flow
```
Serving Engine (vLLM/SGLang)
    ↓
LMCacheManager / Cache Engine
    ↓
GPU Connector (format conversion)
    ↓
Memory Management (allocate/deallocate)
    ↓
Storage Backends (CPU/Disk/Remote/P2P/GDS/NIXL)
```

### Server Components
- **api_server/**: External REST API (FastAPI)
- **internal_api_server/**: Internal RPC for cluster coordination
- **offload_server/**: ZMQ-based async offload handling
- **cache_controller/**: Cluster mode coordination

### C/CUDA Extensions (csrc/)
- **mem_kernels.cu**: Memory operation kernels
- **ac_enc.cu / ac_dec.cu**: Arithmetic coding for compression
- **storage_manager/**: Native C++ storage manager with TTL locks

### Entry Points
- `lmcache_server`: V1 server (`lmcache.v1.server.__main__:main`)
- `lmcache_v0_server`: Legacy V0 server
- `lmcache_controller`: API controller (`lmcache.v1.api_server.__main__:main`)

## Code Conventions

### Commit Message Format
Use prefixed tags: `[MP]`, `[CI/Build]`, `[UT]`, `[Bugfix]`, `[refactor]`, etc.

### File Headers
All new files require Apache-2.0 SPDX license headers (enforced by pre-commit).

### V1 vs Legacy
The `lmcache/v1/` directory contains the production API. The root-level modules (`lmcache/server/`, `lmcache/storage_backend/`) are deprecated V0 code.

## CI/CD

- **GitHub Actions**: Runs pre-commit checks and non-GPU pytest on Python 3.10-3.13
- **Buildkite**: Runs GPU tests (both NVIDIA CUDA and AMD ROCm)
- Tests that require GPUs are skipped in GitHub Actions: `disagg`, `multiprocess`, `nixl`, `eic_backend`
