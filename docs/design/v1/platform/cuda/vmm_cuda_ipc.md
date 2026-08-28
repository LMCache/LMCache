# VMM CUDA IPC wrapper (`VmmCudaIPCWrapper`)

Shares KV-cache tensors backed by CUDA VMM memory (`cuMemCreate` +
`cuMemMap`) with the LMCache MP server. Code:
`lmcache/v1/platform/cuda/ipc_wrapper.py`. Companion of
[raw_cuda_ipc.md](../../multiprocess/raw_cuda_ipc.md), which covers the
legacy-IPC wrappers this one takes over from when the memory is
VMM-backed.

## Why it exists

Legacy CUDA IPC (`cudaIpcGetMemHandle`) supports only
`cudaMalloc`-style allocations. On any VMM-mapped pointer it fails with
`cudaErrorInvalidValue` — verified empirically on H200 / driver 580:
base and interior pointers, with and without exportable handle types
requested at creation. There is no opt-back-in flag; the two IPC
mechanisms are disjoint by design (CUDA Programming Guide, "Shareable
Memory Allocations").

vLLM allocates its KV cache through VMM whenever
`enable_cumem_allocator` is set (sleep mode implies it, and upstream
work proposes making it the default when a KV connector is configured).
Torch's `expandable_segments` is VMM-backed too. Without this wrapper,
`REGISTER_KV_CACHE` on such a pool dies in the legacy wrappers'
`cudaIpcGetMemHandle` call.

## Mechanism

VMM has its own IPC: the exporter turns the allocation handle into an
OS shareable handle, the importer maps it into its own address space.

```
exporter (vLLM worker)                     importer (LMCache server)
----------------------                     -------------------------
cuMemRetainAllocationHandle(data_ptr)
cuMemGetAllocationPropertiesFromHandle     .
  -> requestedHandleTypes                  .
cuMemExportToShareableHandle               .
  -> fd  (POSIX_FILE_DESCRIPTOR)   --SCM_RIGHTS-->  cuMemImportFromShareableHandle
  -> 64B blob (FABRIC)             --pickle----->   cuMemAddressReserve
cuMemRelease(retained handle)              .        cuMemMap
                                           .        cuMemSetAccess
                                           .        tensor = base + offset (CuPy/DLPack)
```

Key driver facts the design leans on (all verified on H200 / driver
580 except where noted):

- `cuMemRetainAllocationHandle` recovers the allocation handle from a
  bare pointer, interior pointers included — so the wrapper needs no
  cooperation from the allocator (no vLLM patch to reach the handle).
- Which shareable forms exist is fixed by `requestedHandleTypes` at
  `cuMemCreate` time and cannot be added afterwards (export on
  `NONE`-typed memory fails with `CUDA_ERROR_INVALID_VALUE`). The
  wrapper reads the property from the allocation and fails loudly with
  an actionable hint when nothing is exportable (vLLM only requests
  exportable memory on fabric-capable devices; without IMEX it falls
  back to POSIX fd, on non-fabric devices it requests nothing).
- A POSIX fd is only meaningful inside its own process: the raw fd
  *number* fails to import elsewhere (`CUDA_ERROR_UNKNOWN`). It must be
  duplicated by the kernel — `SCM_RIGHTS` over an `AF_UNIX` socket —
  which is why the fd never rides the pickled wrapper. The wrapper
  carries a 16-byte `export_id`; the fd travels out of band and the
  importer surfaces it through `set_vmm_fd_resolver`. A fabric blob is
  plain bytes and travels inline (requires IMEX channel access in both
  processes; single-node needs only `/dev/nvidia-caps-imex-channels/`
  plus a channel node, no daemon).
- `cuMemGetAddressRange` on VMM memory reports the *mapped chunk*
  containing the pointer, not any larger stitched range, and one
  exported handle maps exactly one chunk. The exporter therefore ships
  explicit `(alloc_size, alloc_offset, nbytes)` and refuses tensors
  spanning multiple chunks (vLLM's CUDA path is one handle per
  segment; the chunked ROCm path is out of scope and rejected loudly).
- VMM handles are refcounted by the driver — the opposite of legacy
  IPC's exporter-owned lifetime. An imported mapping keeps the physical
  memory alive after the exporter releases it, so `close()` is
  mandatory on the importing side (`GPUCacheContext` retains wrappers
  and closes them on teardown, same as the raw wrapper), and "the
  exporter freed it" must never be assumed while imports are open.

## Dispatch

Three orthogonal modes, one wrapper each, selected by two mutually
exclusive process-global switches consulted in
`CudaDeviceSpec.ipc_wrapper_cls` (`_select_ipc_wrapper_cls`):

| switches | wrapper |
|---|---|
| default | `CudaIPCWrapper` (torch storage IPC) |
| `isolated_ipc` | `RawCudaIPCWrapper` (raw legacy IPC handles) |
| `use_vmm_api` (`lmcache/v1/platform/vmm_ipc.py`) | `VmmCudaIPCWrapper` |

`use_vmm_api` composes with `isolated_ipc`, enforced per allocation at
wrap time: the fabric kind is isolation-clean (inline blob; the IMEX
channel is device injection, not a shared namespace or volume — the
event leg rides the timeline-semaphore backend, whose buffer LMCache
`cudaMalloc`s itself), while a POSIX-fd-only allocation is rejected —
fd passing needs a shared filesystem path for `SCM_RIGHTS`, which the
zero-share model rules out. Validated cross-container (two isolated
docker containers, toolkit-injected channel via
`NVIDIA_IMEX_CHANNELS=0`, blob over TCP, both directions).

The knob and the memory must agree, and mismatches fail loudly at
registration: a legacy wrapper on VMM memory dies in
`cudaIpcGetMemHandle` with the existing VMM hint, and
`VmmCudaIPCWrapper` on `cudaMalloc` memory dies in
`cuMemRetainAllocationHandle` ("not VMM-backed"). The TRT-LLM adapter's
direct `RawCudaIPCWrapper(...)` instantiation bypasses the switches
(its pool is `cudaMalloc`'d).

## Deliberate limitations (fail loudly, revisit when materialized)

- **No exporter-side dedup.** Each wrapper exports and imports its own
  mapping. N tensor views of one segment cost N fds + N mappings —
  correct but wasteful; vLLM typically allocates one segment per layer
  tensor, so the shared-segment case is rare. Known trigger that would
  materialize it: vLLM's packed KV layout (DeepSeek-V4 default /
  `enable_cross_layers_blocks`), where every layer aliases one giant
  allocation — N full-pool mappings per registration. Add dedup then.
- **No multi-chunk allocations** (vLLM ROCm cumem path).
- **Fd delivery transport is out of scope here.** This module defines
  only the seam (`fd_payload` send side, `set_vmm_fd_resolver` receive
  side); the `AF_UNIX` channel between worker and server is a separate
  concern (follow-up PR; decided default: socket at
  `/dev/shm/lmcache_fd_<port>.sock` — `/dev/shm` is already shared in
  every `use_vmm_api` deployment because the default event backend
  requires it). Until it lands, fd-kind wrappers fail at `to_tensor`
  with a pointed error; fabric-kind wrappers work end-to-end today.

## Testing

`tests/v1/platform/test_vmm_ipc_wrapper.py`. VMM pools are built
directly with cuda-bindings (`_VmmPool`), emulating vLLM's allocator
exactly (same `cuMemCreate` properties). The cross-process test stands
in for the fd transport with `socket.send_fds` over a socketpair. The
fabric round trip is `skipif`-gated on an IMEX channel being present
(`/dev/nvidia-caps-imex-channels`), so it activates on fabric-enabled
nodes without configuration.
