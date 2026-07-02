## 1. Scope

This guide is a **step-by-step checklist for enabling a new, non-CUDA
accelerator in LMCache Multiprocess (MP) mode** over the **engine-driven
transfer path**. It is written so that a hardware vendor can integrate a new
device by touching only a small, self-contained set of files, without needing
to understand the rest of LMCache.

The engine-driven path is LMCache's device-agnostic default for every non-CUDA
backend (it already backs the CPU, XPU, and other non-CUDA devices). KV cache
is gathered/scattered by the vLLM worker process and exchanged with the LMCache
server through CPU-side chunks (SHM or pickle). It requires **no cross-process
device-memory sharing** on the new hardware, so the integration surface is
minimal.

**In scope**

- Device detection so LMCache recognizes the new accelerator.
- Registering an ops backend so the worker-side KV transfer primitive resolves.
- A `platform/<device>/` sub-package that carries the above.
- (Optional, for production throughput) a native KV-transfer kernel that keeps
  gather/scatter on-device instead of routing through the Python fallback.

**Explicitly out of scope** (a new backend does **not** need these to run):

- The **in-process GPU Connector** layer (used only by the in-process, non-MP
  path). See [`ARCHITECTURE_MULTI_HARDWARE.md`](../../ARCHITECTURE_MULTI_HARDWARE.md).
- The **lmcache-driven** transfer path (CUDA IPC handles / interprocess events).
  Non-CUDA devices stay on the engine-driven path; the `DeviceIPCWrapper`
  auto-discovery registry is an advanced, opt-in capability, not a requirement.
- Any C++/kernel work in **Phase 1** — a first bring-up runs entirely on the
  generic Python fallback.

For background on how the engine-driven transport works end to end, see
[`../multiprocess/engine_driven_transfer_design.md`](../multiprocess/engine_driven_transfer_design.md).

> Throughout this guide, `<device>` is the placeholder for your device type
> string (the value `_detect_device()` returns, e.g. the same token you would
> see as `tensor.device.type`). Substitute it consistently everywhere:
> the detection branch, the sub-package name `lmcache/v1/platform/<device>/`,
> and the ops backend predicate.

## 2. Architecture at a Glance

```text
+----------------------------------------------------------------------+
|                        lmcache/__init__.py                           |
|   _detect_device()  -> (torch_dev, "<device>")                       |
|   _get_backend()    -> merges platform/<device>/ops.py over          |
|                        python_ops_fallback into lmcache.c_ops        |
+---------------------------------+------------------------------------+
                                  |
                                  v
+----------------------------------------------------------------------+
|              vLLM Worker Process (runs on the new device)            |
|                                                                      |
|   LMCacheMPConnector                                                 |
|      -> EngineDrivenTransferContext  (worker-side gather/scatter)    |
|            +- SHM   (shared memory, lowest copy count)               |
|            +- Pickle (serialized, universal fallback)                |
|         gather/scatter calls                                         |
|         lmcache.c_ops.multi_layer_block_kv_transfer                  |
+---------------------------------+------------------------------------+
                                  |  ZMQ + (SHM / pickle)
                                  v
+----------------------------------------------------------------------+
|                     LMCache MP Server Process                       |
|   Storage Manager (L1 memory + L2 remote storage)                   |
+----------------------------------------------------------------------+
```

Enabling a new backend touches exactly three things: **device detection**, the
**ops backend**, and a small **`platform/<device>/` sub-package** that hosts
them. Everything above the transport line is hardware-agnostic and unchanged.

## 3. Prerequisites

Before starting, the new device's PyTorch backend must support standard tensor
operations used by the engine-driven path:

| Capability | Why it is needed |
|---|---|
| `torch.<device>.is_available()`, `device_count()` | Device detection / enumeration |
| `set_device()` / `current_device()` / `synchronize()` / `empty_cache()` | Device binding and lifecycle used by the middle layer |
| `tensor.cpu()` / `tensor.to("cpu")` | Copy KV tensors from device to host |
| `tensor.to(<device>)` | Copy KV tensors from host back to device |

If PyTorch on the new device supports these, the engine-driven path can run.
No cross-process memory sharing is required.

## 4. The Three Steps

Each step is self-contained and solves exactly one problem. Do them in order;
after Step 3 you have a working (fallback-speed) integration.

### Step 1 — Make LMCache detect the device

**Problem this solves:** LMCache must know which accelerator is present so the
middle layer binds to the right `torch_dev`.

**File:** `lmcache/__init__.py`, function `_detect_device()`.

Add a branch for the new device, keeping the existing priority order. Return
the device's torch module and its device-type string:

```python
def _detect_device() -> tuple[Any, str]:
    try:
        import torch
    except ImportError:
        return None, "cpu"  # CLI-only fallback

    if hasattr(torch, "musa") and torch.musa.is_available():
        return torch.musa, "musa"
    # ===== new device branch =====
    elif hasattr(torch, "<device>") and torch.<device>.is_available():
        logger.info("<device> is available. Using <device> for LMCache engine.")
        return torch.<device>, "<device>"
    # =============================
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu, "xpu"
    elif hasattr(torch, "hpu") and torch.hpu.is_available():
        return torch.hpu, "hpu"
    elif torch.cuda.is_available():
        return torch.cuda, "cuda"
    else:
        from lmcache.v1.platform.cpu.stub_cpu_device import StubCPUDevice
        return StubCPUDevice("cpu"), "cpu"
```

> If your PyTorch extension is not exposed as `torch.<device>` (for example it
> needs an explicit `import <ext>` first, or it reuses the `torch.cuda`
> namespace on an adapter layer), adjust the `hasattr(...)`/`is_available()`
> check accordingly, or distinguish it via an environment variable.

### Step 2 — Register the ops backend

**Problem this solves:** even on the engine-driven path, the worker's
`gather_paged_kv_to_cpu()` / `scatter_cpu_to_paged_kv()` call
`lmcache.c_ops.multi_layer_block_kv_transfer` (see
`lmcache/v1/multiprocess/transfer_context/base.py`). If no backend is
registered for the new device, `lmcache.c_ops` will not resolve and engine
initialization fails.

**File:** `lmcache/__init__.py`, function `_get_backend()`.

Add the new device to `backend_candidates`, ahead of the generic entries and
consistent with the detection order in Step 1. The comment
`# should extend to more HWs..` in the current code marks exactly where this
goes:

```python
    backend_candidates = [
        # ===== new device backend (Python adapter under platform/<device>) =====
        (
            "lmcache.v1.platform.<device>.ops",
            "<device>_ops",
            lambda: hasattr(torch, "<device>") and torch.<device>.is_available(),
        ),
        # =======================================================================
        (
            "lmcache.v1.platform.musa.ops",
            "musa_ops",
            lambda: hasattr(torch, "musa") and torch.musa.is_available(),
        ),
        (
            "lmcache.xpu_ops",
            "xpu_ops",
            lambda: torch.xpu.is_available(),
        ),
        (
            "lmcache.c_ops",
            "cuda_ops",
            lambda: torch.cuda.is_available(),
        ),
    ]
```

The selected backend module is merged **on top of** `python_ops_fallback` at
function granularity: any function you define in `platform/<device>/ops.py`
overrides the fallback, and everything you leave undefined keeps using the
generic Python implementation. This is why a minimal backend only has to
provide `multi_layer_block_kv_transfer`.

### Step 3 — Add the `platform/<device>/` sub-package

**Problem this solves:** Steps 1 and 2 point at
`lmcache/v1/platform/<device>/`; this step creates it. LMCache auto-loads
sub-packages under `lmcache.v1.platform`, so the new directory is picked up
once it exists.

Create two files.

**3a. `lmcache/v1/platform/<device>/__init__.py`**

Must exist (even empty). Optionally register an availability predicate so the
platform registry knows the backend is live:

```python
# lmcache/v1/platform/<device>/__init__.py
# SPDX-License-Identifier: Apache-2.0
"""<device>-specific platform primitives."""

from lmcache.v1.platform._registry import register_availability


def _<device>_is_available() -> bool:
    """Lazy availability check to avoid a circular import at module load."""
    from lmcache import torch_dev

    return torch_dev.is_available()


register_availability("<device>", _<device>_is_available)
```

**3b. `lmcache/v1/platform/<device>/ops.py`**

The dispatch entry that Step 2 registered. Mirror the structure of the existing
`lmcache/v1/platform/musa/ops.py`: try an optional native kernel first, then
fall back to the generic Python implementation so the primitive is **always**
callable — even before any native kernel exists:

```python
# lmcache/v1/platform/<device>/ops.py
# SPDX-License-Identifier: Apache-2.0
"""<device> ops backend assembled into ``lmcache.c_ops`` at import time.

The package initializer merges this module over ``python_ops_fallback`` when
<device> is the active device. Functions not defined here continue to use the
generic Python fallback implementation.
"""

from __future__ import annotations

import torch

import lmcache.python_ops_fallback as py_ops


def _tensor_list(value: object) -> list[torch.Tensor] | None:
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, torch.Tensor) for item in value):
        return None
    return value


def multi_layer_block_kv_transfer(
    paged_buffer_ptrs_tensor: torch.Tensor | list,
    lmcache_objects_ptrs: list[int] | list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    device: torch.device | str,
    direction: py_ops.TransferDirection,
    shape_desc: py_ops.PageBufferShapeDesc,
    lmcache_chunk_size: int,
    engine_kv_format: py_ops.EngineKVFormat,
    skip_prefix_n_blocks: int,
) -> None:
    """Block-based multi-layer KV transfer for <device>.

    Fast path: dispatch to the optional native kernel adapter. Slow path:
    fall back to the generic Python implementation so the primitive is always
    callable while a native adapter rolls out.
    """
    from lmcache.v1.platform.<device>.native_kv_transfer import (
        try_native_multi_layer_block_kv_transfer,
    )

    object_tensors = _tensor_list(lmcache_objects_ptrs)
    if object_tensors is not None and try_native_multi_layer_block_kv_transfer(
        paged_layers=paged_buffer_ptrs_tensor,
        object_tensors=object_tensors,
        block_ids=block_ids,
        direction=direction,
        shape_desc=shape_desc,
        lmcache_chunk_size=lmcache_chunk_size,
        engine_kv_format=engine_kv_format,
        skip_prefix_n_blocks=skip_prefix_n_blocks,
    ):
        return

    py_ops.multi_layer_block_kv_transfer(
        paged_buffer_ptrs_tensor,
        lmcache_objects_ptrs,
        block_ids,
        device,
        direction,
        shape_desc,
        lmcache_chunk_size,
        engine_kv_format,
        skip_prefix_n_blocks,
    )
```

For a Phase-1 bring-up you may omit the native import entirely and forward
directly to `py_ops.multi_layer_block_kv_transfer(...)`. Keeping the native
dispatch scaffold in place from the start makes the Phase-2 upgrade a drop-in.

After these three steps, the engine-driven path (pickle and SHM) runs
end-to-end on the new device using the Python fallback for KV transfer.

## 5. Optional: Native KV-Transfer Acceleration

The Python fallback copies the full KV between host and device on the worker
side, which becomes a throughput bottleneck for long prompts. It is a safe
development-time floor, **not** a production target. To reach native
throughput, ship an on-device kernel and let `ops.py` dispatch to it.

This is intentionally **decoupled from the LMCache repository**: the kernel is
delivered as a standalone, optional Python extension (a wheel), so the LMCache
main repo never gains a device-specific build dependency. Users enable it with
`pip install <your-kernel-wheel>` plus one environment variable.

### 5.1 The adapter: `platform/<device>/native_kv_transfer.py`

Model it on `lmcache/v1/platform/<device_reference>/native_kv_transfer.py`
(the MUSA adapter is a complete reference). It must be **fail-closed**: never a
required dependency, returning `False` whenever native dispatch is unavailable
so callers transparently fall back.

```python
# lmcache/v1/platform/<device>/native_kv_transfer.py
# SPDX-License-Identifier: Apache-2.0
"""Optional native <device> KV-transfer adapter.

Fail-closed: never makes the native kernel a required dependency; returns
``False`` whenever native dispatch is unavailable so callers fall back to the
Python implementation.
"""

from importlib import import_module
from typing import Any
import os

import torch

ENV_NATIVE_KV_TRANSFER = "LMCACHE_<DEVICE>_NATIVE_KV_TRANSFER"
NATIVE_KV_TRANSFER_ABI_VERSION = 1

_REQUIRED_NATIVE_SYMBOLS = (
    "native_lmcache_kv_transfer_abi_version",
    "lmcache_kv_paged_to_buffer",
    "lmcache_kv_buffer_to_paged",
    "lmcache_mla_paged_to_buffer",
    "lmcache_mla_buffer_to_paged",
)


def is_native_kv_transfer_enabled() -> bool:
    return os.environ.get(ENV_NATIVE_KV_TRANSFER, "").lower() in {"1", "true", "yes"}


def load_native_module() -> Any | None:
    try:
        return import_module("<your_kernel_module>")
    except Exception:
        return None


def check_native_abi(module: Any) -> bool:
    for name in _REQUIRED_NATIVE_SYMBOLS:
        if not callable(getattr(module, name, None)):
            return False
    try:
        return int(module.native_lmcache_kv_transfer_abi_version()) == \
            NATIVE_KV_TRANSFER_ABI_VERSION
    except Exception:
        return False


def _is_device_contiguous_tensor(t: torch.Tensor) -> bool:
    return t.device.type == "<device>" and t.is_contiguous()

# The remaining dispatch logic — try_native_multi_layer_block_kv_transfer(...),
# block_id -> slot_mapping expansion, MLA dimension derivation, per-object
# iteration — is device-agnostic. Copy it verbatim from the reference adapter
# and only swap the device literal and the env-var / module names.
```

### 5.2 The kernel contract (ABI)

The native module must expose five symbols. The first reports an ABI version
for compatibility checks; the other four move data between paged KV and a
contiguous buffer, in both directions, for the standard and MLA layouts:

| Symbol | Role |
|---|---|
| `native_lmcache_kv_transfer_abi_version()` | Return the ABI version (currently `1`) |
| `lmcache_kv_paged_to_buffer(...)` | **D2H gather**: copy the `slot_mapping` tokens from multi-layer paged KV into a contiguous buffer |
| `lmcache_kv_buffer_to_paged(...)` | **H2D scatter**: write buffer tokens back into paged KV, skipping the first `skip_prefix_n_tokens` |
| `lmcache_mla_paged_to_buffer(...)` | D2H gather for the MLA layout (single latent, `num_heads = 1`) |
| `lmcache_mla_buffer_to_paged(...)` | H2D scatter for the MLA layout |

Implementation notes that apply to any accelerator:

1. **Inputs** are on-device, contiguous tensors; reject anything else in the
   adapter before calling the kernel.
2. **Streams:** launch on the worker's current device stream; do **not** force
   a synchronize inside the kernel — the LMCache middle layer synchronizes via
   `torch_dev.synchronize()` when needed, and forcing it breaks overlap with
   the compute stream.
3. **Errors:** on unsupported layout or launch failure, return a C++ `bool`
   `false` so the adapter falls back. Do **not** throw across the ABI boundary.

### 5.3 Enable / fallback policy

- **Default off:** with `LMCACHE_<DEVICE>_NATIVE_KV_TRANSFER` unset, the adapter
  returns `False` and the Python fallback runs — a safe net for dev/CI before
  the kernel is ready.
- **Enable in production:** `export LMCACHE_<DEVICE>_NATIVE_KV_TRANSFER=1` with
  the kernel wheel installed. If the wheel is missing, the loader returns
  `None` and it falls back again — inference never crashes.
- **Layout mismatch:** if the KV layout is unsupported or a tensor is not a
  contiguous on-device tensor, `try_native_*` returns `False` for that single
  call only; the next call retries native.

## 6. Verification

Start the MP server, point vLLM at it on the new device, and confirm cache
reuse. The connector configuration is device-independent; the only backend
specific choice is forcing the engine-driven transfer mode.

```bash
# Start the LMCache MP server
lmcache server --l1-size-gb 10 --eviction-policy LRU --port 5555
```

```bash
# Start vLLM on the new device with the LMCache MP connector
vllm serve <your-model> \
    --kv-transfer-config '{
        "kv_connector": "LMCacheMPConnector",
        "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "lmcache.mp.host": "tcp://localhost",
            "lmcache.mp.port": "5555",
            "lmcache.mp.mp_transfer_mode": "engine_driven"
        }
    }' \
    --no-enable-prefix-caching \
    --port 8000
```

Send the same long prompt twice; the second request should hit the cache and
show a markedly lower TTFT.

**Checklist**

- [ ] `_detect_device()` returns `(torch.<device>, "<device>")`.
- [ ] `_get_backend()` loads `lmcache.v1.platform.<device>.ops` on the device.
- [ ] `lmcache.c_ops.multi_layer_block_kv_transfer` resolves (native or fallback).
- [ ] Engine-driven + pickle passes end-to-end.
- [ ] Engine-driven + SHM passes end-to-end.
- [ ] Store/retrieve data correctness verified.
- [ ] Multi-worker (TP > 1) verified.
- [ ] *(Optional)* native `try_native_multi_layer_block_kv_transfer` hits and is correct.

### `kv_connector_extra_config` reference

| Key | Default | Notes |
|---|---|---|
| `lmcache.mp.host` | `tcp://localhost` | MP server address (with ZMQ transport prefix) |
| `lmcache.mp.port` | `5555` | MP server port |
| `lmcache.mp.mq_timeout` | `300.0` | Message-queue request timeout (seconds) |
| `lmcache.mp.heartbeat_interval` | `10.0` | Heartbeat interval (seconds) |
| `lmcache.mp.mp_transfer_mode` | `auto` | Transfer mode; a non-CUDA device should set this to `engine_driven` |

## 7. Reference Implementations

| Topic | Reference | Notes |
|---|---|---|
| Full backend sub-package | `lmcache/v1/platform/musa/` | Detection, ops backend, native adapter — all three pieces present |
| Ops dispatch template | `lmcache/v1/platform/musa/ops.py` | `multi_layer_block_kv_transfer` try-native-then-fallback pattern |
| Native KV transfer template | `lmcache/v1/platform/musa/native_kv_transfer.py` | Device-agnostic dispatch logic to copy |
| Engine-driven call site | `lmcache/v1/multiprocess/transfer_context/base.py` | Where gather/scatter invoke `multi_layer_block_kv_transfer` |
| Availability / wrapper registry | `lmcache/v1/platform/_registry.py` | `register_availability`; advanced auto-discovery (out of scope) |
| CPU SHM wrapper | `lmcache/v1/platform/cpu/shm.py` | Engine-driven SHM backend reference |

## 8. Related Design Docs

- Multi-hardware architecture (in-process / GPU Connector): [`../../ARCHITECTURE_MULTI_HARDWARE.md`](../../ARCHITECTURE_MULTI_HARDWARE.md)
- Engine-driven transfer design: [`../multiprocess/engine_driven_transfer_design.md`](../multiprocess/engine_driven_transfer_design.md)
- Cross-platform event notification: [`event_notifier.md`](event_notifier.md)
- MP protocol system: [`../../../../lmcache/v1/multiprocess/protocols/README.md`](../../../../lmcache/v1/multiprocess/protocols/README.md)
