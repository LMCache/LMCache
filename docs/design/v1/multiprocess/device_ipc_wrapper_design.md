# `DeviceIPCWrapper` — a device-agnostic base for KV-cache IPC

## Motivation

The multiprocess (MP) transport shares a paged KV-cache buffer between the producer process (vLLM / SGLang / TRT-LLM) and the LMCache server by sending an IPC handle for the underlying storage once, at `REGISTER_KV_CACHE` time. Every later `STORE`/`RETRIEVE` carries only paged block IDs, never tensors.

Historically all of these handle wrappers subclassed `CudaIPCWrapper`:

```text
CudaIPCWrapper
├── RawCudaIPCWrapper      (TRT-LLM raw cudaMalloc pool)
└── CpuShmTensorWrapper    (CPU POSIX-SHM, in platform/cpu/shm.py)
```

In fact a CPU shared-memory wrapper and a TRT-LLM raw-pointer wrapper are not CUDA caching-allocator tensors, yet they inherited `_share_cuda_`- based machinery they never used. It also made non-CUDA backends impossible to add cleanly.

## The new hierarchy

A device-agnostic base, `DeviceIPCWrapper`, now owns everything that is not transport-specific. Each concrete wrapper is a direct sibling:

```text
DeviceIPCWrapper                        base: contract + (de)serialize
├── CudaIPCWrapper                      cuda  — torch caching allocator
├── RawCudaIPCWrapper                   cuda — raw cudaMalloc, TRT-LLM
└── CpuShmTensorWrapper                 cpu   — POSIX shared memory
```

All of them live behind a single msgspec ext code 1 and a single `KVCache = list[DeviceIPCWrapper]` wire type, so new device backends can be added as further siblings without touching the wire format.

## What the base class owns

`DeviceIPCWrapper` (in `platform/base_ipc_wrapper.py`) provides the parts that every
transport shares:

- **Interface fields** — `dtype`, `shape`, `stride`, `storage_offset`, `device_uuid`. Subclasses populate these in `__init__`; the base uses them for equality and the receiving side uses them to rebuild the logical view.
- **Device discovery** — `_get_device_uuid`, `_discover_devices`,`_get_device_index_from_uuid`. These are `@classmethod`s (not static) and route through the `torch_dev` abstraction, so they work across device backends, and a subclass can override `_get_device_uuid` if its backend needs a different identity source.
- **Equality** — `__eq__` uses a `type(self) is type(other)` guard, so two different wrapper subclasses never compare equal even if their fields coincide.
- **Serialization** — `Serialize`/`Deserialize` are `pickle.dumps` / `pickle.loads`. Pickle preserves the concrete subclass identity across the wire, which is what lets a single ext code carry every wrapper type (see below).
- **`to_tensor()`** — abstract; raises `NotImplementedError`. Every subclass overrides it with its transport-specific reconstruction.

## How to dispatch

- `_CUSTOMERIZED_SERIALIZERS` is keyed on `DeviceIPCWrapper` with ext code 1, dispatched by `isinstance` in the encoder hook. Every subclass instance therefore encodes through the same path.
- `Serialize` is `pickle.dumps(obj)` -> the concrete subclass survives on the wire. `Deserialize` reconstructs it and `to_tensor()` dispatches to the correct override.
- `KVCache = list[DeviceIPCWrapper]` is the registered msgspec type, so a single `list[...]` payload can mix any of the wrappers and the server needs zero per-type branching.

## The concrete wrappers

| Wrapper | Device type | Transport | Reconstruction |
|---|---|---|---|
| `CudaIPCWrapper` | `cuda` | `UntypedStorage._share_cuda_()` | `_new_shared_cuda` + `set_()` |
| `RawCudaIPCWrapper` | `cuda` | `cudaIpcGetMemHandle` (raw ptr) | `cudaIpcOpenMemHandle` → CuPy → DLPack |
| `CpuShmTensorWrapper` | `cpu` | POSIX `shm_open` | `mmap` same segment |

## Platform registration

The factory lookup (`platform/_registry.py`) keys on `tensor.device.type`, so
the integration adapter never has an if/elif chain.  Concrete wrappers are
discovered automatically at runtime — no static `register_kv_wrapper` calls
needed:

- Each concrete subclass sets a ``device_type`` ClassVar (e.g. ``"cuda"``)
  and exposes a ``wrap`` factory classmethod.
- :func:`~lmcache.v1.platform._registry._discover_wrappers_once` scans
  ``lmcache.v1.platform`` two levels deep for ``DeviceIPCWrapper`` subclasses
  on first use, indexes them by ``device_type``, and skips any subclass where
  ``_is_default_wrapper`` is ``False`` (so ``RawCudaIPCWrapper`` coexists with
  ``CudaIPCWrapper`` without colliding).
- Adding a new accelerator backend only requires shipping a sub-package under
  ``platform/<device>/`` with a ``DeviceIPCWrapper`` subclass — zero changes
  to the dispatcher or the registry.

## Backward compatibility

- Wire format is unchanged. Still ext code 1, still `pickle`-over-`Ext`. Previously serialized payloads round-trip.
- `CudaIPCWrapper` / `RawCudaIPCWrapper` keep their names, fields, and behavior; only their base class changed. Existing callers and the TRT-LLM adapter are unaffected.
- The single `isinstance`-based equality check now uses `type(self) is type(other)`, which is stricter and correct.
