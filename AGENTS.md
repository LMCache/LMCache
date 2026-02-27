# AGENTS.md

Guidelines for AI coding agents (Copilot, Cursor, Claude Code, etc.) working in this repository.

## Project Overview

LMCache is a KV cache management engine for LLM serving that reduces Time To First Token (TTFT) and increases throughput. It stores KV caches across multiple tiers (GPU, CPU, disk, S3) and integrates with vLLM and SGLang.

## Python Environment

We recommend using [uv](https://docs.astral.sh/uv/) to manage Python environments and dependencies:

```bash
# Create and activate a virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install torch               # pre-requisite for CUDA extensions
uv pip install -e . --no-build-isolation
```

## Build & Install

```bash
# Standard install with CUDA extensions (requires torch pre-installed)
pip install -e . --no-build-isolation

# Source-only (no CUDA extensions)
NO_CUDA_EXT=1 pip install -e .

# HIP/ROCm build
BUILD_WITH_HIP=1 pip install -e .
```

## Testing

### Running Tests

```bash
# Run standard test suite (mirrors CI)
pytest -xvs --ignore=tests/disagg \
  --ignore=tests/v1/test_nixl_storage.py \
  --ignore=tests/v1/multiprocess/ \
  --ignore=tests/v1/distributed/ \
  --ignore=tests/skipped \
  --ignore=tests/v1/storage_backend/test_eic.py

# Run a single test file
pytest -xvs tests/v1/test_cache_engine.py

# Run a single test
pytest -xvs tests/v1/test_cache_engine.py::test_function_name
```

Test dependencies: `uv pip install -r requirements/test.txt`

Pytest marker: `@pytest.mark.no_shared_allocator` disables the shared-allocator monkeypatch for a test.

### Testing Practices

- Write tests against the **public interface and docstring contract**, not the implementation. Test as if you don't know the internals — verify that behavior matches what the docstring describes.
- Avoid accessing private members in tests unless strongly needed.
- All new features and bug fixes should include corresponding tests.
- Ensure existing tests still pass before submitting changes.

## Linting & Code Quality

```bash
# Run all checks (mirrors CI exactly)
pre-commit run --all-files

# Individual tools
ruff check .              # Lint (E, F, B, SLF rules)
ruff format .             # Format (line-length 88)
isort .                   # Import sorting (black profile, from_first=true)
mypy --config-file=pyproject.toml   # Type checking
codespell --toml pyproject.toml     # Spell checking
```

C++/CUDA files use clang-format (Google style, 80-col). Rust code in `rust/` uses `cargo fmt` and `cargo clippy`.

All Python files require an `# SPDX-License-Identifier: Apache-2.0` header as the first line.

### Import Ordering

Imports must follow this section-heading convention:

```python
# Standard
import os

# Third Party
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig

# Local
from .utils import helper
```

### SLF (Private Member Access)

SLF lint rules are currently enforced by CI only in `lmcache/v1/multiprocess/` and `lmcache/v1/distributed/`. However, **all new code should follow SLF discipline regardless of location** — never access private members (prefixed with `_`) of other classes. Treat this as a project-wide coding standard for any new or modified code.

## Coding Conventions

### Type Hints

All functions and methods must have type hints for arguments and return values.

### Docstrings

Every public function and method must have a clear docstring covering:
- What the function does
- Arguments (with types and descriptions)
- Return values
- Raised exceptions (if any)
- Additional notes when behavior is non-obvious

### Encapsulation

Never access private members (prefixed with `_`) of other classes. Interact only through their public API.

### Code Organization

- **Module-level helper functions** go at the top of the file (after imports, before classes).
- **Private/helper methods** within a class go at the end of the class, after all public methods.

## Code Review Checklist

When reviewing code (or self-checking before submitting), verify all of the following:

### Correctness
- [ ] The code does what it claims to do and matches the PR description.
- [ ] Edge cases are handled (empty inputs, None values, boundary conditions).
- [ ] No regressions to existing functionality — existing tests still pass.

### Style & Standards
- [ ] `pre-commit run --all-files` passes with no errors.
- [ ] All new/modified functions have type hints for arguments and return values.
- [ ] All new/modified public functions have complete docstrings.
- [ ] License header (`# SPDX-License-Identifier: Apache-2.0`) is present on all Python files.
- [ ] Import ordering follows the section-heading convention (Standard / Third Party / First Party / Local).

### Encapsulation & Design
- [ ] No direct access to private members (`_`-prefixed) of other classes.
- [ ] New public APIs are minimal and well-defined — avoid exposing internals.
- [ ] Module-level helpers are placed at the top; private methods at the end of the class.

### Testing
- [ ] New features and bug fixes include corresponding tests.
- [ ] Tests target the public interface and docstring contract, not implementation details.
- [ ] Tests pass locally: `pytest -xvs` with the standard ignore flags.

### Safety & Performance
- [ ] No security vulnerabilities (injection, unsafe deserialization, etc.).
- [ ] No unnecessary memory copies or allocations in hot paths.
- [ ] Thread safety is maintained for shared data structures.
- [ ] CUDA/GPU resources are properly managed (allocated, freed, synchronized).

## Architecture

### Core Engine (lmcache/v1/)

The v1 API is the active codebase. `lmcache/v0/` and `lmcache/server/` are legacy.

- **LMCacheManager** (`manager.py`) — Top-level lifecycle manager. Initializes and coordinates all components (engine, lookup client, offload server, API server, plugins). Entry point for vLLM integration.
- **LMCacheEngine** (`cache_engine.py`) — Core cache logic. Processes tokens, manages cache store/retrieve operations, coordinates with storage and GPU connectors.
- **LMCacheEngineConfig** (`config.py`, `config_base.py`) — YAML-based configuration with environment variable overrides and alias support for deprecated keys.

### Storage Layer

- **StorageManager** (`storage_backend/storage_manager.py`) — Coordinates across storage backends and manages data flow.
- **Backends** (`storage_backend/`) — `local_cpu_backend.py` (CPU memory), `local_disk_backend.py` (disk), `gds_backend.py` (GPUDirect Storage), `remote_backend.py`, `pd_backend.py` (disaggregated prefill), `p2p_backend.py` (peer-to-peer), `nixl_storage_backend.py`.
- **Cache Policies** (`storage_backend/cache_policy/`) — LRU, LFU, FIFO, MRU eviction strategies.
- **Serialization** (`storage_backend/naive_serde/`) — Encoding/decoding for cache data.

### Memory Management

- **MemoryAllocatorInterface** (`memory_management.py`) — Abstract allocator with implementations: `MixedMemoryAllocator`, `PagedTensorMemoryAllocator`, `CuFileMemoryAllocator`. Thread-safe, NUMA-aware.
- **MemoryObj / MemoryFormat** — Core data containers for cache entries.

### Token Processing

- **TokenDatabase** (`token_database.py`) — Converts tokens to cache keys. `ChunkedTokenDatabase` for chunk-based, `SegmentTokenDatabase` for segment-based processing.

### GPU Connector

- **GPUConnectorInterface** (`gpu_connector/`) — Abstracts GPU memory operations. CUDA/HIP implementations plus a mock for testing.

### Multi-process mode:

- **Distributed storage manager** (`distributed/`) — Hierarchical cache with L2 adapters and storage controllers.
- **Multiprocess frontend** (`multiprocess/`) — Running LMCache as a separate process with an API server for KV cache management.

### Integration

- **vLLM** (`integration/vllm/`) — `vllm_v1_adapter.py` is the main adapter; `lmcache_connector_v1.py` connects to vLLM's KV cache.
- **SGLang** (`integration/sglang/`) — SGLang-specific adapter.

### Supporting Components

- **LookupClient** (`lookup_client/`) — Cache lookup interface with async/sync variants.
- **OffloadServer** (`offload_server/`) — ZMQ-based server for KV cache offloading.
- **Observability** (`observability.py`) — Prometheus metrics, stats logging, performance monitoring.
- **Plugin System** (`plugin/`) — Dynamic runtime plugin loading.

### C++/CUDA Extensions (`csrc/`)

- `mem_kernels.cu` — Memory operation CUDA kernels
- `redis/` — Native Redis connector (multi-threaded RESP)
- `storage_manager/` — Native storage manager operations
- Python bindings via `pybind.cpp`

