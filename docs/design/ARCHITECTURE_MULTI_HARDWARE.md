
# LMCache Multi-Hardware Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 lmcache/v1/platform/__init__.py                │
│                                                                 │
│  torch_dev, torch_device_type = _detect_device()                │
│  _ops = get_backend(torch_device_type)                          │
│                                                                 │
│  ┌───────────┐     ┌───────────┐     ┌───────────┐              │
│  │ torch.cuda│     │ torch.xpu │     │ torch.hpu │  ...         │
│  └─────┬─────┘     └─────┬─────┘     └─────┬─────┘              │
│        └──────────────────┴──────────────────┘                  │
│                           │                                     │
│                     torch_dev (unified entry)                   │
│                  torch_device_type ("cuda"/"xpu"/"hpu"/"cpu")   │
│                                                                 │
│  [Registry Discovery Point]                                     │
│  DeviceSpec subclasses are auto-discovered under                │
│  lmcache.v1.platform and selected by availability.              │
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
| **Entry** `v1/platform/__init__.py` | `_detect_device()` + `get_backend()` | Registry-driven detection and backend composition. |
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

`_detect_device()` also accepts a CPU-only environment where none of the
supported accelerators (CUDA, XPU, HPU) is available. In that case
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

1. Add a `DeviceSpec` subclass under `lmcache/v1/platform/<device>/__init__.py`
2. Point `ops_module` to `lmcache.v1.platform.<device>.ops`
3. Implement `multi_layer_block_kv_transfer` in `ops.py` with Python fallback
4. Add `native_kv_transfer.py` with ABI checks and fail-closed behavior
5. Use MP `engine_driven` mode for validation and rollout

No edits to `lmcache/__init__.py` or global backend candidate lists are required.
