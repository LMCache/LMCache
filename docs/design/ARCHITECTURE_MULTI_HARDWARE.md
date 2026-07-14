# LMCache Multi-Hardware Architecture

This document describes the multi-hardware architecture for LMCache's
multiprocess (MP) mode. 

```
┌─────────────────────────────────────────────────────────────────┐
│                 lmcache/v1/platform/__init__.py                 │
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
│                  │ │              │ │ IPC-capable (hasattr):   │
│                  │ │              │ │ .Event(interprocess)     │
│                  │ │              │ │ .from_ipc_handle()       │
│                  │ │              │ │ CUDA-only (hasattr):     │
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
│ ┌──────────────────────┐                                        │
│ │ PagedTensorMemory    │   uses torch_dev:                      │
│ │ Allocator            │   .synchronize()                       │
│ └──────────────────────┘   .cudart() (hasattr)                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              Transfer Context Layer (per-hardware routing)      │
│                                                                 │
│ ┌────────────────────────┐  ┌──────────────────────────────┐    │
│ │ EngineDriven           │  │ LMCacheDriven                │    │
│ │ TransferContext        │  │ TransferContext              │    │
│ │                        │  │                              │    │
│ │ • CPU workers          │  │ • CUDA workers (IPC)         │    │
│ │ • Pickle / SHM backend │  │ • SHM wrappers (CPU+CUDA)    │    │
│ │ • gather/scatter copy  │  │ • zero-copy transfer         │    │
│ └────────────────────────┘  └──────────────────────────────┘    │
│                                                                 │
│ Route: create_transfer_context(kv_caches, mode)                 │
│   mode = auto | engine_driven | lmcache_driven                  │
└─────────────────────────────────────────────────────────────────┘
```

## Design Principles

| Layer | Device Reference | Notes |
|-------|-----------------|-------|
| **Entry** `v1/platform/__init__.py` | `_detect_device()` + `get_backend()` | Registry-driven detection and backend composition. |
| **Middle** engine / storage / multiprocess | `from lmcache import torch_dev` | Hardware-agnostic unified code |
| **Middle** IPC-capable / device-specific APIs | `hasattr(torch_dev, 'xxx')` guard | Graceful runtime degradation |
| **Bottom** Transfer Context | `create_transfer_context(kv_caches, mode)` | Per-device routing: CUDA→LMCacheDriven, others→EngineDriven |

## Transfer Mode Routing (`transfer_context/worker_transfer.py`)

```
MPTransferMode.AUTO (default):
  device_type == "cuda"  -->  LMCacheDrivenTransferContext  (IPC zero-copy)
  device_type != "cuda"  -->  EngineDrivenTransferContext    (gather/scatter copy)

MPTransferMode.ENGINE_DRIVEN:
  any device             -->  EngineDrivenTransferContext

MPTransferMode.LMCACHE_DRIVEN:
  any device             -->  LMCacheDrivenTransferContext   (requires IPC support)

Override: LMCACHE_MP_TRANSFER_MODE env var or the mode argument to create_transfer_context()
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
server smoke tests on a CPU-only host) and CLI loading without torch. In
MP mode, CPU workers use `EngineDrivenTransferContext` (Pickle or SHM
backend) for KV transfer; there is no GPU-side connector involved.

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
