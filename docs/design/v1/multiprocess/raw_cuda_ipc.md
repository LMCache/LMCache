# `RawCudaIPCWrapper` — sharing tensors allocated outside PyTorch

## Why a second wrapper

The default `CudaIPCWrapper` in `platform/cuda/ipc_wrapper.py` calls
`tensor.untyped_storage()._share_cuda_()` to publish the storage over
CUDA IPC. That path only works when the storage is owned by PyTorch's
caching allocator, *and* it requires both processes to share a
`/dev/shm` tmpfs because torch keeps its IPC reference-counter file
there. Two call sites cannot use it:

- TRT-LLM's KV pool is published via `at::for_blob(...)` over a raw
  `cudaMalloc`, so `_share_cuda_()` raises and the vLLM-style wrapper
  cannot be used at all.
- Under isolated-IPC deployments (`--isolated-ipc` on the LMCache
  server + matching `lmcache.mp.isolated_ipc` on the worker), producer
  and consumer containers share no host IPC namespace and no common
  `/dev/shm` — even a torch-owned KV pool is unshareable through
  `_share_cuda_()` there.

`RawCudaIPCWrapper` bypasses PyTorch's IPC layer:

- **Sender** calls `cudaIpcGetMemHandle(data_ptr)` directly via
  `cuda.bindings.runtime` to obtain a portable handle.
- **Receiver** calls `cudaIpcOpenMemHandle(handle, ...)` to map the
  remote pointer, wraps it as a flat `uint8` `cupy.ndarray` via
  `UnownedMemory`, converts to `torch.Tensor` via DLPack, then
  `view(dtype).reshape(shape)` to restore the logical layout.

The `uint8` round-trip is deliberate — `bfloat16` and FP8 dtypes have
no direct CuPy/NumPy equivalent, but the size in bytes is enough.
DLPack carries no dtype semantics for the bytes view; `view(dtype)` on
the torch side restores them.

## Shared base rather than separate type

`RawCudaIPCWrapper` is a **sibling** of `CudaIPCWrapper`: both subclass
the device-agnostic `DeviceIPCWrapper` base (see
[`device_ipc_wrapper_design.md`](device_ipc_wrapper_design.md) for the
full hierarchy). Sharing a single base is load-bearing for the wire
format:

- `KVCache = list[DeviceIPCWrapper]` is the registered msgspec type for
  `REGISTER_KV_CACHE`. msgspec does **not** support unions of custom
  ext-encoded types — adding a parallel class with its own ext code
  would force a wider decoder type and break either round-trip or the
  existing `DeviceIPCWrapper` consumers.
- The customized serializer (`_CUSTOMERIZED_SERIALIZERS`) is keyed on
  `DeviceIPCWrapper` and dispatched by `isinstance`, so every subclass
  instance encodes through the same path with **ext code 1**.
- `Serialize` is `pickle.dumps(obj)`, which preserves the concrete
  subclass identity. On the receiving side `Deserialize` reconstructs
  the concrete subclass and `to_tensor` dispatches to the correct
  override.

The receiving server therefore needs no per-type branching: a
`list[DeviceIPCWrapper]` arriving at
`LMCacheDrivenTransferModule.register_kv_cache` contains any mix of
concrete wrappers, and `to_tensor()` does the right thing.

## Sender-side validation

`RawCudaIPCWrapper.__init__` first runs
`attempt_permute_to_contiguous_view` (matching `CudaIPCWrapper`) and
then rejects anything still non-contiguous. TRT-LLM allocates its
pool contiguously and the vLLM MP path likewise ships contiguous
per-layer views, so the only valid recovery from a non-contiguous
input is "the sender did something wrong" — surface it loudly rather
than silently `.contiguous()`-ing and copying GBs of KV cache.

## Reconstruction lifetime

`UnownedMemory` takes `owner=self`, so the wrapper instance pins the
mapping for the tensor's lifetime. The underlying producer-side
allocation (TRT-LLM's `cudaMalloc` pool, or the vLLM caching-allocator
pool under isolated IPC) outlives the wrapper.
`cudaIpcCloseMemHandle` is called on last-use through a process-wide
open registry: the driver returns one mapping per (process,
allocation) no matter how many wrappers open it, and one close unmaps
it for every user, so opens are refcounted and only the final wrapper
closes the mapping. Closing matters — an open mapping pins the
*exporting* process's device memory even after the exporter dies,
which would otherwise leak a full KV pool per worker restart. See
[`../../platform/cuda/ipc_wrapper.md`](../../platform/cuda/ipc_wrapper.md)
for the driver-level constraints (VMM incompatibility, PID-collision
requirement, exporter-liveness rule).

## Why no `_share_cuda_` fallback

The wrapper does not try `_share_cuda_()` first. That would couple the
two codepaths, and the failure mode is silent corruption (PyTorch
returns a handle for a different region of memory than what the
caller intended). `RawCudaIPCWrapper` stays a concrete sibling of
`CudaIPCWrapper`, and the choice between them is made outside the
wrapper: `CudaDeviceSpec.ipc_wrapper_cls` returns `RawCudaIPCWrapper`
when `is_isolated_ipc()` is true and `CudaIPCWrapper` otherwise, while
the TRT-LLM adapter continues to instantiate `RawCudaIPCWrapper`
directly regardless of the switch.
