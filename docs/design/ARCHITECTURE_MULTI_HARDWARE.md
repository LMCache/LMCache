
# LMCache Multi-Hardware Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│               lmcache/v1/platform/__init__.py                   │
│                                                                 │
│  torch_dev, torch_device_type = _detect_device()                │
│  (re-exported from lmcache/__init__.py as `lmcache.torch_dev`)  │
│                                                                 │
│  ┌───────────┐     ┌───────────┐     ┌───────────┐              │
│  │ torch.cuda│     │ torch.xpu │     │ torch.hpu │  ...         │
│  └─────┬─────┘     └─────┬─────┘     └─────┬─────┘              │
│        └──────────────────┴──────────────────┘                  │
│                           │                                     │
│                     torch_dev (unified entry)                   │
│              torch_device_type ("cuda"/"musa"/"xpu"/            │
│                                 "hpu"/"cpu")                    │
│                                                                 │
│  [Registry-driven]                                              │
│  Backends are discovered by scanning `lmcache.v1.platform`      │
│  for `DeviceInfo` subclasses (see `base_device_info.py`).       │
│  Adding a new hardware requires:                                │
│    - a `DeviceInfo` subclass in a `platform/<backend>/`         │
│      sub-package `__init__.py`,                                 │
│    - a gpu_connector implementation.                            │
│  The `DEVICE_TYPE` env var forces the detector to prefer        │
│  one registered device_type when multiple are available.        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
              ┌────────────────┼──────────────────┐
              ▼                ▼                  ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────────────┐
│ Cache Engine     │ │ Storage      │ │ Multiprocess             │
│                  │ │ Backends     │ │ Server / Client          │
│ • store          │ │              │ │                          │
│ • retrieve       │ │ • LocalCPU   │ │ • IPC futures            │
│ • lookup         │ │ • Disk       │ │ • message queue          │
│                  │ │ • Remote     │ │ • blend server           │
│ torch_dev:       │ │ • PD Backend │ │                          │
│ .synchronize()   │ │              │ │ torch_dev:               │
│ .empty_cache()   │ │ torch_dev:   │ │ .device()                │
│ .set_device()    │ │ .current_    │ │ .stream()                │
│                  │ │  device()    │ │ .Event()                 │
│                  │ │ .device_     │ │ .Stream()                │
│                  │ │  count()     │ │                          │
│                  │ │              │ │ CUDA-only (hasattr):     │
│                  │ │              │ │ .Event(interprocess)     │
│                  │ │              │ │ .from_ipc_handle()       │
│                  │ │              │ │ .cudart()                │
└────────┬─────────┘ └──────┬───────┘ └─────────────┬────────────┘
         │                  │                       │
         └──────────────────┼───────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Memory Management Layer                      │
│                                                                 │
│ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│ │ MixedMemory  │  │ PinMemory    │  │ LazyMemory   │            │
│ │ Allocator    │  │ Allocator    │  │ Allocator    │            │
│ └──────────────┘  └──────────────┘  └──────────────┘            │
│ ┌──────────────┐  ┌──────────────┐                              │
│ │ XPUMemory    │  │ PagedTensor  │   uses torch_dev:            │
│ │ Allocator    │  │ MemAllocator │   .synchronize()             │
│ └──────────────┘  └──────────────┘   .cudart() (hasattr)        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│        GPU Connector Layer (per-hardware, no unification)       │
│                                                                 │
│ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│ │ CUDA            │  │ XPU             │  │ HPU             │   │
│ │                 │  │                 │  │                 │   │
│ │ • PagedMemV2/V3 │  │ • PagedMemXPUV2 │  │ • PagedMemHPU   │   │
│ │ • Layerwise     │  │ • LayerwiseXPU  │  │                 │   │
│ │ • Buffer        │  │                 │  │ torch.hpu.*     │   │
│ │ • SGLang        │  │ torch.xpu.*     │  │                 │   │
│ │                 │  │ python_ops_fb   │  │                 │   │
│ │ torch.cuda.*    │  │                 │  │                 │   │
│ │ c_ops + cupy    │  │                 │  │                 │   │
│ └─────────────────┘  └─────────────────┘  └─────────────────┘   │
│                                                                 │
│ Route: torch_device_type -> cuda/xpu/hpu -> Connector           │
└─────────────────────────────────────────────────────────────────┘
```

## Design Principles

| Layer | Device Reference | Notes |
|-------|-----------------|-------|
| **Entry** `lmcache/v1/platform/__init__.py` | `_detect_device()` -> `torch_dev` (re-exported from `lmcache/__init__.py`) | Registry-driven detection over `DeviceInfo` subclasses. Detect once, reuse globally. |
| **Middle** engine / storage / multiprocess | `from lmcache import torch_dev` | Hardware-agnostic unified code |
| **Middle** CUDA-only APIs | `hasattr(torch_dev, 'xxx')` guard | Graceful runtime degradation |
| **Bottom** GPU Connector | Direct `torch.cuda` / `torch.xpu` / `torch.hpu` | Per-hardware impl, no abstraction |

## Connector Routing (`gpu_connector/__init__.py`)

```
torch_device_type == "cuda"  -->  VLLMPagedMemGPUConnectorV2/V3
torch_device_type == "xpu"   -->  VLLMPagedMemXPUConnectorV2
torch_device_type == "hpu"   -->  VLLMPagedMemHPUConnector
torch_device_type == "cpu"   -->  (no GPU connector; raises RuntimeError)
```

## CPU-Only Stub Fallback

`_detect_device()` (defined in `lmcache/v1/platform/__init__.py`) also
accepts a CPU-only environment where none of the supported accelerators
(CUDA, MUSA, XPU, HPU) is available. In that case
`torch_device_type` is `"cpu"` and `torch_dev` is either:

- `lmcache.v1.platform.cpu.stub_cpu_device.StubCPUDevice` — when `torch`
  is importable but no GPU is detected. The stub implements the subset of
  the `torch.cuda` / `torch.xpu` / `torch.hpu` surface used by the middle
  layer (`Event`, `Stream`, `device`, `synchronize`, `set_device`,
  `current_device`, `device_count`, `get_device_properties`,
  `empty_cache`), as no-op or constant returns. `is_available()` is
  `False`, so any `hasattr(torch_dev, 'xxx')` consumer that gates on the
  real device's availability stays on the degraded path.
- `None` — when `torch` itself is not importable (the `lmcache-cli`
  slim install). The CLI surface (`lmcache ping`, `lmcache describe`,
  `lmcache query`, `lmcache bench engine`) tolerates this; engine and
  storage paths do not.

The stub is intended for L1-adapter-only flows (e.g., end-to-end MP
server smoke tests on a CPU-only host) and CLI loading without torch. It
is **not** a CPU connector: there is no entry for `"cpu"` in
`gpu_connector/__init__.py`, so calling `CreateGPUConnector` with
`torch_device_type == "cpu"` raises `RuntimeError("No supported cpu
connector found.")`.

`normalize_kv_and_discover_format` also hardcodes `kv_layout = "HND"`
when `torch_device_type == "cpu"`, because vLLM's
`get_kv_cache_layout()` reports `NHD` for its CPU attention backend
which is wrong for that backend's actual KV cache layout.

## Adding New Hardware

1. Create a `lmcache/v1/platform/<backend>/` sub-package and define a
   concrete `DeviceInfo` subclass in its `__init__.py`. Fill in the
   abstract properties (`device_type`, `torch_module_name`,
   `ops_module`) and `is_available()`; optionally override
   `pin_memory_backend` and `is_handle_transfer_available`. The
   sub-package is picked up automatically by `discover_subclasses` in
   `lmcache.v1.platform` -- no manual registration is required. See the
   existing `platform/cuda/`, `platform/musa/`, `platform/xpu/`,
   `platform/hpu/` entries for reference. Optionally: users can set
   `DEVICE_TYPE=<device_type>` at runtime to force selection when
   multiple registered devices are available.
2. Create `gpu_connector/xxx_connectors.py`, implement `GPUConnectorInterface`
3. Add routing branch in `gpu_connector/__init__.py`
4. Add kernels in `c_ops/` or fallback in `python_ops_fallback.py`; if
   the new backend ships its own compiled ops, point the `DeviceInfo`
   subclass' `ops_module` at the correct fully-qualified module path so
   `get_backend()` merges it on top of `python_ops_fallback`.
5. No changes needed in middle layer code
