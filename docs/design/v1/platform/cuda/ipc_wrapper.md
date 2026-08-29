# CUDA KV-Cache IPC Wrappers

## Motivation

MP-mode registration ships every worker KV-cache tensor to the LMCache
server as an IPC wrapper (`wrap_kv_caches` → `resolve_kv_wrapper_factory`
→ `DeviceSpec.ipc_wrapper_cls`). The default CUDA wrapper,
`CudaIPCWrapper`, rides PyTorch's `UntypedStorage._share_cuda_()` /
`_new_shared_cuda()`, which works only when both processes share a
`/dev/shm` tmpfs: torch keeps its IPC reference-counter file there.
That is the *memory leg* of the `hostIPC: true` requirement (the *event
leg* is covered by `timeline_semaphore_event_ipc.md`).

`RawCudaIPCWrapper` removes it: `cudaIpcGetMemHandle` /
`cudaIpcOpenMemHandle` rendezvous entirely in the kernel driver and work
across fully isolated containers — no shared IPC namespace, no common
`/dev/shm` (verified empirically on driver 580 / CUDA 13).

## Mechanism

An IPC mem handle always maps the **whole allocation** it was taken
from, and opening it returns the allocation's **base** pointer. Torch
caching-allocator tensors usually sit at an interior pointer, so:

- **Producer**: `base = cuMemGetAddressRange(data_ptr)`, handle =
  `cudaIpcGetMemHandle(base)`, ship `(handle bytes, offset = data_ptr -
  base, nbytes, dtype/shape/stride, device uuid)`.
- **Consumer**: `cudaIpcOpenMemHandle` → mapped base, view the tensor's
  bytes at `mapped_base + offset` (CuPy `UnownedMemory` → DLPack →
  `torch`, flat `uint8` then `view(dtype).reshape(shape)` — `uint8`
  avoids CuPy dtype gaps for bf16/fp8).

Layout normalization matches `CudaIPCWrapper`:
`attempt_permute_to_contiguous_view` first, then a hard reject of
anything still non-contiguous (the flat-bytes reconstruction would
silently reorder elements).

## Selection

`CudaDeviceSpec.ipc_wrapper_cls` consults the process-global
isolated-IPC switch (`lmcache/v1/platform/isolated_ipc.py`) via the
module-level `_select_ipc_wrapper_cls()`: on → `RawCudaIPCWrapper`,
off → `CudaIPCWrapper`. The same switch selects the event backend, so
one goal-named knob (`lmcache.mp.isolated_ipc` / `--isolated-ipc`)
covers both IPC legs. Registration happens after process init sets the
switch, and the class is resolved per call (no caching), so there is no
ordering hazard. The TRT-LLM adapter keeps instantiating
`RawCudaIPCWrapper` directly (its `cudaMalloc`'d pool cannot go through
`_share_cuda_()` at all); its base pointers simply get offset 0.

## Constraints (empirical, driver 580 / CUDA 13)

- **Only `cudaMalloc`-style allocations.** Memory from the CUDA VMM API
  has no legacy IPC handle: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  and vLLM's sleep mode (`CuMemAllocator`) both break registration under
  isolated IPC. The wrapper fails loudly with that hint
  (`_NON_IPC_MEMORY_HINT`).
- Exporter and importer must have different PID *values* (namespace
  isolation is fine); a collision fails at `cudaIpcOpenMemHandle` with
  error 201.
- Imported mappings are refcounted and explicitly closed. The driver
  returns one mapping per (process, allocation) no matter how often it
  is opened, and a single `cudaIpcCloseMemHandle` unmaps it for every
  user — so opens are counted in a process-wide registry and the
  mapping is unmapped when the last wrapper closes. Closing matters:
  an open mapping pins the **exporting** process's device memory even
  after the exporter dies, so an unclosed registration leaks a full KV
  pool per worker restart (observed as a crash-looping vLLM pod whose
  replacement could not allocate). `GPUCacheContext.close()` — called
  by the server's unregister and worker-reaper teardown — closes the
  wrappers; a registration that fails mid-construction rolls back the
  mappings it already opened. Until a dead worker is reaped (grace
  timeout), its pool stays transiently pinned — bounded and
  self-healing.
- The exporting process must stay alive while consumers import — same
  exporter-liveness rule as the event backend.
- Consumer-side reconstruction needs `cupy` (already a hard dependency)
  and `cuda-python` (declared in the CUDA requirement files).
- NVIDIA-only, same as the timeline-semaphore backend: ROCm has no
  `cuda.bindings`.

## Status

Selected by `CudaDeviceSpec.ipc_wrapper_cls` behind the isolated-IPC
switch, off by default. With both legs behind the switch, an
isolated-IPC deployment has **zero `/dev/shm` dependencies** in the MP
path — `--ipc host` / `hostIPC: true` can be dropped (see
`docs/source/mp/deployment.rst`). Remaining series work: migrate the
SGLang/CacheBlend/qstore call sites, flip the default, drop `hostIPC`
from the operator.
