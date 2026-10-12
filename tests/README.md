# LMCache Tests: Structure and Contribution Rules

This document describes the current test layout under `tests/` and the baseline rules for adding new tests.

## Current Structure

Top-level test areas:

- `tests/benchmarks/`: benchmark-oriented tests and benchmark validation helpers.
- `tests/cli/`: tests for CLI commands and user-facing CLI behavior.
- `tests/data/`: test fixtures and data-driven inputs.
- `tests/disagg/`: disaggregated-mode tests and related docs.
- `tests/lmcache_frontend/`: frontend integration tests.
- `tests/sdk/`: SDK-level tests.
- `tests/tools/`: utility/tooling tests.
- `tests/v1/`: core v1 unit and integration-style tests (platform, storage, multiprocess, compute, etc.); representative v1 sub-areas include:
- Top-level files like `tests/test_*.py`: cross-cutting tests (utils, serde, telemetry, observability, and similar).

- `tests/v1/platform/`: platform abstraction and device-specific behavior tests.
- `tests/v1/storage_backend/`: storage backends and storage path behavior tests.
- `tests/v1/multiprocess/`: multiprocess runtime behavior, IPC primitives, engine-driven transfer, and process-interaction scenarios.
- `tests/v1/mp_coordinator/`: coordinator lifecycle, worker orchestration, and MP control-plane coordination checks.
- `tests/v1/mp_observability/`: MP observability coverage, including event recording, error surfacing, and timeout/reporting behavior.
- `tests/v1/shm_allocator/`: shared-memory allocation, reuse, boundary handling, and allocator safety checks.
- Other v1 areas (for example `compute`, `gpu_connector`, `distributed`, `cache_controller`, etc.): specialized kernel, connector, distributed, and service-behavior tests.

## Requirements for New Tests

### 1) Local validation is mandatory

Before opening or updating a PR, new/changed tests must pass locally.

Recommended baseline checks:

```bash
pre-commit run --all-files
pytest -q tests/<target_path>
```

If your change impacts multiple modules, run a broader test selection, not only a single file.

### 2) Do not hardcode torch device APIs

Do not introduce direct hardcoded device usage patterns like:

- `torch.cuda.*`, `torch.xpu.*`, `torch.musa.*` in generic/shared logic
- `torch.device("cuda")`, `torch.device("xpu")`, `torch.device("musa")`

Prefer LMCache abstractions:

- `from lmcache import torch_dev, torch_device_type`
- platform/spec abstractions in `lmcache.v1.platform.*`

This aligns with repository policy enforced by pre-commit (`Ban direct torch device usage`).

### 3) Device-related tests must include proper marks

If a test depends on a specific device/runtime, add matching pytest marks defined in `pytest.ini`:

- `@pytest.mark.cuda`
- `@pytest.mark.xpu`
- `@pytest.mark.musa`
- `@pytest.mark.sglang` (for SGLang-specific tests)

When the test is runtime-availability dependent, also gate with `skipif`, for example:

```python
@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason=f"Requires available {torch_device_type} runtime",
)
```

Use marks and skip conditions together where appropriate so CI and local runs remain explicit and predictable.
