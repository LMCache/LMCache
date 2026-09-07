# LMCache Multi-Hardware Architecture

This document describes the multi-hardware architecture for LMCache's
multiprocess (MP) mode.

```
┌─────────────────────────────────────────────────────────────────┐
│                 lmcache/v1/platform/__init__.py                 │
│                                                                 │
│  torch_dev, torch_device_type = _detect_device()                │
│  _ops = resolve_device_ops(torch_device_type)                   │
│                                                                 │
│  ┌───────────┐     ┌───────────┐     ┌───────────┐              │
│  │ torch.cuda│     │ torch.xpu │     │ torch.hpu │  ...         │
│  └─────┬─────┘     └─────┬─────┘     └─────┬─────┘              │
│        └──────────────────┴──────────────────┘                  │
│                           │                                     │
│                     torch_dev (unified entry)                   │
│              torch_device_type (e.g. "cuda"/"musa"/"xpu"/       │
│                                 "hpu"/"cpu"; auto-discoverable) │
│                                                                 │
│  [Registry Discovery Point]                                     │
│  DeviceSpec subclasses are auto-discovered under                │
│  lmcache.v1.platform and selected by availability.              │
│  The DEVICE_TYPE env var forces the detector to prefer one      │
│  registered device_type when multiple are available.            │
│                                                                 │
│  [DeviceOps Resolution]                                         │
│  DeviceSpec.ops_cls → DeviceOps subclass (e.g. CudaDeviceOps)  │
│  DeviceSpec.get_ops() → cached singleton instance               │
│  lmcache.device_ops = resolve_device_ops(device_type)           │
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
│ │ • host-side workers    │  │ • IPC-capable device workers │    │
│ │ • Pickle / SHM backend │  │ • SHM wrappers (host+device) │    │
│ │ • gather/scatter copy  │  │ • zero-copy handle transfer  │    │
│ └────────────────────────┘  └──────────────────────────────┘    │
│                                                                 │
│ Route: create_transfer_context(kv_caches, mode)                 │
│   mode = auto | engine_driven | lmcache_driven                  │
└─────────────────────────────────────────────────────────────────┘
```

## Design Principles

| Layer | Device Reference | Notes |
|-------|-----------------|-------|
| **Entry** `v1/platform/__init__.py` | `_detect_device()` + `resolve_device_ops()` | Registry-driven detection and DeviceOps resolution. |
| **Entry** DeviceOps | `DeviceSpec.ops_cls` → `DeviceSpec.get_ops()` | OOP polymorphism: each accelerator subclasses `DeviceOps` with native ops. |
| **Middle** engine / storage / multiprocess | `from lmcache import torch_dev` | Hardware-agnostic unified code |
| **Middle** ops call sites | `from lmcache import device_ops` | Direct reference to the resolved `DeviceOps` singleton. |
| **Middle** IPC-capable / device-specific APIs | `hasattr(torch_dev, 'xxx')` guard | Graceful runtime degradation |
| **Bottom** Transfer Context | `create_transfer_context(kv_caches, mode)` | Per-device routing. In `AUTO` mode: CUDA→LMCacheDriven, other devices→EngineDriven. Other IPC-capable devices (e.g. MUSA) can opt-in to LMCacheDriven via explicit `mode=lmcache_driven` when their `DeviceSpec` reports `is_handle_transfer_available() == True`. |
| **Bottom** Cache Context | `DeviceSpec.create_cache_context()` | Per-device cache context factory dispatched via `DeviceSpec` registry. |

## DeviceOps Architecture

```
┌────────────────────────────────────────────────────────────────┐
│              lmcache.device_ops (DeviceOps instance)          │
│                                                                │
│  resolve_device_ops(torch_device_type) returns this singleton  │
└────────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────┐
│  DeviceOps (base)                                              │
│  ─────────────────────                                         │
│  • torch_ops baseline (delegates to torch_ops.py)              │
│  • bind_native(module)                                         │
│  • ensure_native()                                             │
│                                                                │
│  Class attributes (from ops_types.py):                         │
│  • EngineKVFormat                                              │
│  • TransferDirection                                           │
│  • PageBufferShapeDesc                                         │
│  • NativePlanType stubs (StagingCopy, LaunchVar, etc.)         │
└────────────────────────────────┬───────────────────────────────┘
                                 │ (inheritance)
             ┌─────────┬─────────┼─────────┬─────────┐
             ▼         ▼         ▼         ▼         ▼
         ┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐
         │CudaDev ││XpuDev  ││MusaDev ││HpuDev  ││CpuDev  │
         │iceOps  ││iceOps  ││iceOps  ││iceOps  ││iceOps  │
         │        ││        ││        ││        ││        │
         │ensure_ ││ensure_ ││ensure_ ││(no     ││(no     │
         │native: ││native: ││native: ││native) ││native) │
         │lmcache ││lmcache ││method  ││        ││        │
         │.cuda_  ││.xpu_ops││override││        ││        │
         │ops     ││        ││        ││        ││        │
         │(pybind)││(SYCL)  ││        ││        ││        │
         └────────┘└────────┘└────────┘└────────┘└────────┘
```

**Key design points:**

- `DeviceOps` instances are singletons cached by `DeviceSpec.get_ops()`.
- The torch baseline in `torch_ops.py` provides every op as a pure-Python
  function using standard PyTorch APIs — works on any device.
- `bind_native(module)` walks the native module's public symbols and
  rebinds them on the instance, replacing torch baseline ops with compiled implementations
  or adding native-only symbols that have no pure-Python equivalent
  (e.g. `execute_object_group_transfer`). Consumers feature-detect with
  `hasattr(device_ops, "op_name")`.
- `ops_types.py` defines shared enums, descriptors, and native plan type stubs.

## Transfer Mode Routing (`transfer_context/worker_transfer.py`)

```
MPTransferMode.AUTO (default):
  device_type == "cuda"  -->  LMCacheDrivenTransferContext  (IPC zero-copy)
  device_type != "cuda"  -->  EngineDrivenTransferContext    (gather/scatter copy)

MPTransferMode.ENGINE_DRIVEN:
  any device             -->  EngineDrivenTransferContext

MPTransferMode.LMCACHE_DRIVEN:
  any device that reports  --> LMCacheDrivenTransferContext
  `DeviceSpec.is_handle_transfer_available() == True`
  (otherwise the factory raises and the caller must fall back)

Override: LMCACHE_MP_TRANSFER_MODE env var or the mode argument to create_transfer_context()
```

## Cache Context Dispatch

The `create_cache_context` factory in `lmcache.v1.platform.cache_context`
delegates to the `DeviceSpec` registry:

```
create_cache_context(kv_caches, ...)
  → device_type = kv_caches[0].to_tensor().device.type
  → spec = get_device_spec(device_type)
  → spec.create_cache_context(*args, **kwargs)
```

Each `DeviceSpec` subclass overrides `create_cache_context()` with a lazy
import of its concrete `BaseCacheContext` implementation. The base-class
default raises `NotImplementedError`.

## CPU-Only Stub Fallback

`_detect_device()` also accepts a CPU-only environment where none of the
supported accelerators (CUDA, MUSA, XPU, HPU) is available. In that case
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

1. Add a ``DeviceSpec`` subclass under
   ``lmcache/v1/platform/<device>/__init__.py``.  Override ``ops_cls``
   to return your ``DeviceOps`` subclass (or inherit the base which
   returns ``DeviceOps`` itself for pure torch baseline).
2. Add a ``DeviceOps`` subclass under
   ``lmcache/v1/platform/<device>/device_ops.py``.  Override
   ``ensure_native()`` to bind your compiled extension (or leave it
   empty for pure torch baseline).
3. Verify with MP ``engine_driven`` mode (see the :doc:`developer guide
   <../source/developer_guide/extending_lmcache/adding_a_new_device_backend>`).
4. (Optional) Add ``ipc_wrapper_cls`` and ``create_cache_context()``
   overrides on your ``DeviceSpec`` for LMCache-driven transfer.

No edits to ``lmcache/__init__.py`` or global backend candidate lists
are required. Users can set ``DEVICE_TYPE=<device_type>`` at runtime to
force selection of a registered device when multiple are available.
