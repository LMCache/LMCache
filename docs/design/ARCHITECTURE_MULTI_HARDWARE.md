
# LMCache Multi-Hardware Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      lmcache/__init__.py                        │
│                                                                 │
│  torch_dev, torch_device_type = _detect_device()                │
│                                                                 │
│  ┌───────────┐     ┌───────────┐     ┌───────────┐              │
│  │ torch.cuda│     │ torch.xpu │     │ torch.hpu │  ...         │
│  └─────┬─────┘     └─────┬─────┘     └─────┬─────┘              │
│        └──────────────────┴──────────────────┘                  │
│                           │                                     │
│                     torch_dev (unified entry)                   │
│                  torch_device_type ("cuda"/"xpu"/"hpu"/"cpu")   │
│                                                                 │
│  [Monkey Patch Point]                                           │
│  New hardware can be added by extending _detect_device()        │
│  and providing a gpu_connector implementation.                  │
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
│ │                 │  │ non_cuda_equiv  │  │                 │   │
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
| **Entry** `__init__.py` | `_detect_device()` -> `torch_dev` | Monkey patch point. Detect once, reuse globally. |
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

1. Add detection branch in `__init__.py` `_detect_device()`
2. Create `gpu_connector/xxx_connectors.py`, implement `GPUConnectorInterface`
3. Add routing branch in `gpu_connector/__init__.py`
4. Add kernels in `c_ops/` or fallback in `non_cuda_equivalents.py`
5. No changes needed in middle layer code
