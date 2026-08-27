# Timeline-Semaphore Event IPC Backend

## Motivation

The `lmcache_driven` handle path orders cross-process KV transfers with
device events (`event_ipc_abstraction.md`). The CUDA implementation behind
`DefaultEventIPCBackend` uses interprocess event handles — and empirically
(driver 580, CUDA 13) those only resolve when both processes share a
`/dev/shm` tmpfs. CUDA IPC *memory* handles have no such dependency: they
rendezvous in the kernel driver and work across fully isolated containers.
On Kubernetes, the event handles are what force `hostIPC: true` (or a
shared hostPath tmpfs) onto the vLLM pod and the LMCache DaemonSet for
every STORE/RETRIEVE. `TimelineSemaphoreEventIPCBackend` removes that requirement
by implementing event semantics on memory handles alone.

## Mechanism

A timeline semaphore is a monotonically increasing 64-bit value: signaling
writes a target value, waiting blocks until the value reaches one (the
Vulkan/Direct3D 12 concept). Each process lazily allocates one buffer of
semaphore slots per GPU (4096 × 8 bytes) and exports it once with
`cudaIpcGetMemHandle`. An event is a `(slot, sequence)` event object:

| protocol method | implementation |
| --- | --- |
| `create_event` | new event object; no slot until first record |
| `record_event(e, stream)` | assign the stream's slot, bump its sequence, `cuStreamWriteValue64(slot, seq)` on `stream` |
| `export_event` | pack `(version, mem handle, slot offset, seq)` — 81 bytes, self-contained, rides the existing opaque `bytes` wire field |
| `import_event` | open the exporter's semaphore buffer once (cached), return a wait/query-only event |
| `wait_event(e, stream)` | `cuStreamWaitValue64(slot, seq, GEQ)` on `stream` |
| `query_event` | 8-byte device read |
| `synchronize_event` | semaphore wait on a dedicated stream + `cudaStreamSynchronize` |

Slots are assigned per recording stream: each slot's values are
stream-ordered, hence monotonic, which keeps the `>= seq` wait race-free.
Slot 0 is reserved for the `check_event_support` probe.

Slots and sequences are 64-bit (the 32-bit memops exist and cost the
same) so counter wrap-around is unreachable in practice: a 32-bit slot at
~10k records/s wraps in days, and while the GPU-side GEQ is cyclic and
would tolerate it, host-side comparisons (`query_event` and anything
built on it) are plain `>=`. 64-bit keeps the design's core invariant
simple and absolute: slot values only grow.

## Empirically load-bearing details (driver 580 / CUDA 13)

- **Never device-sync inside lazy buffer allocation** — it runs inside the
  first `record_event`, and a `cudaDeviceSynchronize` there drains the
  caller's recording stream, silently breaking record ordering. The buffer
  is zeroed via `cudaMemsetAsync` on its own stream.
- **`cudaMalloc` may implicitly synchronize the device**, so lazy buffer
  allocation mid-traffic could stall until in-flight work drains.
  Production never hits this: `check_event_support` allocates the buffer,
  and all three MP paths call it at KV-cache registration, before any
  transfer.
- **`CUDA_LAUNCH_BLOCKING=1` constrains the tests, not the backend.** The
  k3 unit CI sets it (`.buildkite/k3_tests/unit/pipeline.yml`), making
  every kernel launch synchronous: a stream is never observably "still
  busy behind kernels" (each kernel completes inside its launch call),
  and launching a kernel behind a *pending* `cuStreamWaitValue64` blocks
  the launching host thread until the wait clears. Memops themselves stay
  asynchronous and correctly ordered under that mode. Tests that assert
  incompleteness therefore gate streams behind a host-released
  `cuStreamWaitValue64` and keep the gated region free of kernel
  launches; allocations happen before gating (an implicit `cudaMalloc`
  device-sync would deadlock behind the gate).
- **Host-side slot reads go through a dedicated non-blocking stream** so
  they cannot synchronize with caller streams blocked in a semaphore wait.
- **`CU_STREAM_WAIT_VALUE_GEQ` is cyclic**: `(int64_t)(*addr - value) >= 0`,
  not a plain unsigned compare. Harmless with monotonic sequences, but any
  future special value written into a slot must stay below `INT64_MAX`.
- **Do not gate on `CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS`** — it
  reports 0 on drivers where the v2 memops (default since CUDA 12) work.
  `check_event_support` probes with real calls.

## Semantics vs `DefaultEventIPCBackend`

Identical for the lifecycle LMCache uses (create → record → export →
import → wait/query/synchronize), including "never-recorded = complete".
Deliberate differences:

- **Snapshot exports**: an exported handle captures the sequence at export
  time; a re-record does not move it (a CUDA event handle tracks the live
  object). LMCache exports once, after the single record.
- **Same-process import works** (resolved via a local registry).
- **Dead peer = unbounded wait, same as CUDA events.** No event backend
  has a timeout on the event leg (`cudaEventSynchronize` has none either);
  peer-death detection belongs to the message-queue timeouts, and
  `DeviceMessagingFuture.wait()` already documents that its timeout covers
  only the message, not the device event.

## Possible extension: force-releasing a dead peer's waiters ("poison")

Unlike CUDA events, this design *could* offer an escape hatch, because the
completion state is plain memory the waiting side can write: on an mq
timeout, write a huge value into the slot to satisfy every pending and
future wait, retire the slot from allocation, and fail later imports of
that `(handle, slot)` pair closed (a live peer still recording there would
otherwise look complete before its GPU work ran). A prototype was
implemented and tested, then removed as unused until the mq-timeout wiring
exists. Two empirically verified requirements for reviving it:

- The release write must be a `cuStreamWriteValue64` on a never-blocked
  stream — a plain host/DMA memcpy updates the memory but never wakes a
  pending `cuStreamWaitValue64` (semaphore wait, verified on driver 580).
- The value must be `INT64_MAX`, not `UINT64_MAX`: the GEQ comparison is
  cyclic, and `UINT64_MAX` is -1 signed, satisfying nothing.

## Constraints

- Same as KV-cache tensor IPC: exporter and importer must have different
  PID *values* (namespace isolation is fine); a collision fails at
  `cudaIpcOpenMemHandle` with error 201.
- Buffers and imported mappings live for the process lifetime (32 KiB per
  process/device); slots are not reclaimed. Never freeing the buffer is
  also a safety requirement, not just simplicity: the driver may hand a
  new allocation at a freed address a **byte-identical** IPC handle
  (observed on driver 580), so a stale handle still circulating in
  another process could silently alias fresh memory.
- Requires `cuda-python` (`cuda.bindings`), resolved lazily on first use
  through the package-shared accessor in `platform/cuda/utils.py` (also
  used by `RawCudaIPCWrapper`), so importing lmcache never requires it.
- NVIDIA-only for now: ROCm reports the `cuda` device type but has no
  `cuda.bindings`, so `check_event_support` fails closed there. HIP has
  equivalent memops (`hipStreamWriteValue64`/`hipStreamWaitValue64`), so
  a ROCm port is possible when needed.
- `synchronize_event` blocks in the driver like `cudaEventSynchronize`:
  it enqueues the semaphore wait on a dedicated per-(thread, device)
  stream and `cudaStreamSynchronize`s it. Measured wake latency on H200:
  ~20 µs vs ~10 µs for CUDA events (an earlier 100 µs-poll implementation
  measured ~174 µs). The sync streams live for the process lifetime.

## Integration constraints (must be resolved before spec binding)

`DefaultEventIPCBackend.export_event` accepts any interprocess-capable
device event; this backend only exports its own event objects. Call sites that
bypass the backend break the moment `CudaDeviceSpec.event_ipc_backend`
returns it:

- vLLM MP connectors create producer events with
  `torch_dev.Event(interprocess=True)` (`lmcache_mp_connector.py:510/:586`
  + `_0180`/`_0201` twins); SGLang and TRT-LLM adapters likewise. They
  must use `backend.create_event` / `record_event` — the event objects satisfy the
  `IPCEvent` duck protocol, so `event.wait(stream)` call sites keep
  working.
- CacheBlend/qstore server modules return raw `event.ipc_handle()` bytes
  instead of `export_event(...)`; already outside the event-IPC
  abstraction (see `event_ipc_abstraction.md` non-goals), must migrate
  first.

Only `lmcache_driven_transfer.py` (server) and `futures.py` route every
event operation through the backend today.

## Status

Standalone implementation, not yet bound to
`CudaDeviceSpec.event_ipc_backend`. Selection/config plumbing is a
follow-up, alongside the raw KV-wrapper work for hostIPC-free deployment.
