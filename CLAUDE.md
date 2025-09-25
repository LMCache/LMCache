# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

LMCache is an LLM serving engine extension that reduces TTFT (Time to First Token) and increases throughput by caching KV caches across multiple storage backends (GPU, CPU, disk). The project integrates with vLLM and supports both v0 and v1 architectures.

## Development Commands

### Testing
- Run all tests: `pytest`
- Run specific test suite: `pytest tests/v1/storage_backend/` (non-CUDA tests)
- Run single test file: `pytest tests/test_blend.py`
- Run with coverage: `pytest --cov=lmcache`

### Code Quality
- Install pre-commit hooks: `pip install -r requirements/lint.txt && pre-commit install`
- Run all pre-commit checks manually: `pre-commit run --all-files`
- Format code: Pre-commit handles ruff formatting, isort, and clang-format automatically
- Type checking: `mypy` (configured in pyproject.toml)
- Lint: `ruff check` (with auto-fix: `ruff check --fix`)

### Build and Installation
- Install from source: `pip install -e . --no-build-isolation` (recommended for development)
- Install dependencies: `pip install -r requirements/common.txt`
- Install test dependencies: `pip install -r requirements/test.txt`

## Architecture

### Core Components
- **cache_engine.py**: Main caching logic and KV cache management
- **v1/**: New architecture supporting vLLM v1 integration
  - **cache_engine.py**: V1 cache engine implementation
  - **memory_management.py**: Memory allocation and management
  - **gpu_connector.py**: GPU-specific operations and CUDA integration
  - **storage_backend/**: Multiple storage backends (local_cpu, local_disk, gds, weka, etc.)
  - **compute/**: Compute kernels and operations
- **storage_backend/**: V0 storage backends
- **server/**: Server implementations for both v0 and v1
- **integration/**: Integration modules for different serving engines

### Key Concepts
- **KV Cache Reuse**: Stores and reuses key-value caches across requests
- **Multi-tier Storage**: GPU -> CPU -> Disk hierarchy for cache storage
- **Disaggregated Architecture**: Separates prefill and decode phases
- **Non-prefix Caching**: Supports caching of any reused text, not just prefixes

## Configuration Files

### Primary Config
- **pyproject.toml**: Main project configuration, dependencies, build settings, and tool configs (ruff, mypy)
- **requirements/**: Split requirements files for different use cases
  - `common.txt`: Core dependencies
  - `test.txt`: Testing framework dependencies
  - `lint.txt`: Code quality tools
  - `build.txt`: Build-time dependencies

### Code Quality
- **.pre-commit-config.yaml**: Pre-commit hooks for automated code quality checks
- **.isort.cfg**: Import sorting configuration
- **pytest.ini**: Test configuration with logging and markers

## Testing Structure

Tests are organized by version and functionality:
- **tests/v1/**: V1 architecture tests (storage backends, connectors, etc.)
- **tests/**: V0 tests and shared test utilities
- **tests/conftest.py**: Pytest fixtures and test configuration
- Uses pytest with asyncio support for testing async components

## Entry Points

The project provides several CLI entry points:
- `lmcache_server`: V1 server (main)
- `lmcache_v0_server`: V0 server (legacy)
- `lmcache_controller`: V1 API server controller