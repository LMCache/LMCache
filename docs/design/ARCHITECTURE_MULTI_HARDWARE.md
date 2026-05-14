
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
│                     torch_device_type ("cuda"/"xpu"/"hpu")      │
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
```

## Adding New Hardware

1. Add detection branch in `__init__.py` `_detect_device()`
2. Create `gpu_connector/xxx_connectors.py`, implement `GPUConnectorInterface`
3. Add routing branch in `gpu_connector/__init__.py`
4. Add kernels in `c_ops/` or fallback in `non_cuda_equivalents.py`
5. No changes needed in middle layer code
