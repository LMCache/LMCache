# Platform Event IPC Abstraction

## Motivation

The `lmcache_driven` multiprocess handle path uses device events to order
cross-process KV-cache transfers:

1. The worker records an event after producing or reserving KV-cache blocks.
2. The server imports that event and waits before reading or writing shared
   KV-cache memory.
3. The server records a completion event after transfer work is queued.
4. The worker future imports, polls, and waits for that completion event.

CUDA-style backends expose this through `Event(interprocess=True)`,
`Event.ipc_handle()`, and `Event.from_ipc_handle(...)`. MUSA has its own
capability gate and TorchMUSA event module. If generic multiprocess modules
import MUSA code or branch on `device.type == "musa"`, every new accelerator
must add another special case to the same logic layer.

The platform event IPC abstraction keeps that decision in
`lmcache/v1/platform/`. Generic multiprocess code calls one backend-neutral
interface; `DeviceSpec` selects the implementation for the requested device.

## Goals

- Keep the `lmcache_driven` handle path free of backend-specific imports and
  device-type branches.
- Preserve `CUDAMessagingFuture` and `to_cuda_future` compatibility aliases.
- Provide CUDA-style/default, ROCm and MUSA event backends behind one
  interface.
- Fail before cross-process memory access when event IPC is unavailable.
- Require each concrete `DeviceSpec` to declare its event IPC capability; do
  not select an unrelated backend through an implicit fallback.

## Non-Goals

- The `STORE` and `RETRIEVE` wire formats do not change; event handles remain
  serialized `bytes` values.
- Tensor memory IPC remains owned by `DeviceIPCWrapper` implementations.
- Worker adapters continue to construct and initially record their producer
  events. The abstraction covers export from the worker and all server/future
  event operations.

## API

`lmcache/v1/platform/base/event_ipc.py` defines the runtime-checkable protocol:

```python
class EventIPCBackend(Protocol):
    device_type: str

    def check_event_support(self, device: object) -> None: ...
    def create_event(self, device: object) -> object: ...
    def export_event(self, event: object, device: object) -> bytes: ...
    def import_event(self, handle: bytes, device: object) -> object: ...
    def record_event(self, event: object, stream: object) -> None: ...
    def wait_event(self, event: object, stream: object) -> None: ...
    def query_event(self, event: object) -> bool: ...
    def synchronize_event(self, event: object, device: object) -> None: ...
```

### Ordering contract

`import_event` must return an event that represents the *producer's* work, on
the producer's stream, in the producer's process. The imported event is the
only thing ordering the two processes: the producer records it and ships the
handle without synchronizing, and the consumer's `wait_event` before touching
shared KV-cache memory is what keeps it from reading blocks the producer has
not finished writing.

A backend must therefore not fabricate the event. Concretely, if
`import_event` cannot open the handle and instead returns an event it created
locally, or one it has already recorded and synchronized, then that event is
complete the moment the consumer receives it. The subsequent `wait_event`
returns immediately, the consumer's transfer kernel runs alongside the
producer's writes, and it can read partially written KV-cache blocks.

Substituting host-side synchronization does not work either. Draining the
importing process's own streams says nothing about the exporting process --
they are separate contexts, and without MPS-style serialization their kernels
are interleaved rather than ordered.

The failure is silent: no error is raised, and it only shows up as wrong cache
contents under load. See LMCache/LMCache#4422 for a concrete case.

If a device cannot support cross-process event import, it must fail
`check_event_support` so the caller falls back to a transfer mode that does not
depend on cross-process ordering, rather than silently degrading correctness.

### Waiting on an imported producer event

Server code never calls `wait_event` on an event imported from an engine
directly. It calls:

```python
def wait_producer_event(
    backend: EventIPCBackend,
    event: object,
    stream: object,
    *,
    is_alive: Callable[[], bool] | None = None,
    timeout_s: float = _PRODUCER_WAIT_TIMEOUT_S,  # 150 s
) -> None: ...
```

On most backends this is `backend.wait_event(event, stream)`. A backend whose
GPU wait on an *imported* event occupies a compute unit while it waits sets the
class attribute `host_ready_before_imported_wait = True`. There the host polls
`query_event` (0.5 ms, then 2 ms after the first 100 ms) until the fence has
completed, and only then queues `wait_event`, which returns at once.

Why: on ROCm, a stream wait on an imported IPC event is not a command-processor
barrier. HIP launches a one-work-item kernel that spins on a CU until the
event's signal is written. If the producer queued a persistent kernel that
needs every CU resident at once ahead of its fence, the spinner takes a CU that
kernel needs, the kernel never finishes, and the fence never signals.

The poll gives up, without queueing the wait, and raises
`ProducerEventWaitError`:

- when `is_alive()` returns False (checked once the poll has slowed down). The
  server modules pass `is_current_entry(instance_id, entry)`, which turns False
  once the instance is unregistered or reaped;
- after `timeout_s` (150 s), which is half the engine adapters' default
  `mq_timeout` (300 s). `mq_timeout` is the longest an engine waits for a store
  before it reuses that store's KV blocks.

The caller must then fail the transfer without touching the engine's buffers:
a store reserves and commits nothing, and a retrieve reads nothing but still
releases its read locks. The deadline matters because the poll blocks the
server's GPU affinity worker, which clients share in FIFO order when there are
fewer workers than client connections (`--max-gpu-workers` defaults to 1).

The server does not learn a client's `mq_timeout`. The ordering above, where the
server gives up on a stuck fence before an engine gives up on a store queued
behind it, holds only while every client keeps `mq_timeout` above the 150 s
deadline.

The lookup entry point is:

```python
def get_event_ipc_backend(device: object) -> EventIPCBackend: ...
```

`device` may be a `torch.device`-like object, a device-type string, or an
integer device index. The lookup resolves a `DeviceSpec` and returns its
`event_ipc_backend`. A missing `DeviceSpec` is a platform registration error,
and a registered spec without an event backend is an unsupported-capability
error. Both cases raise `RuntimeError` instead of selecting a fallback.

All event, device, and stream values are opaque `object` handles at this layer.
The concrete backend owns their types and ABI details.

## Backend Implementations

```text
lmcache/v1/platform/base/event_ipc.py     # protocol, lookup, default backend
lmcache/v1/platform/base/device_spec.py  # optional DeviceSpec capability
lmcache/v1/platform/devices/cpu/__init__.py      # CPU stub backend registration
lmcache/v1/platform/devices/cuda/__init__.py     # CUDA backend registration
lmcache/v1/platform/devices/rocm/__init__.py     # ROCm backend registration
lmcache/v1/platform/devices/rocm/event_ipc.py    # ROCm host-ready backend
lmcache/v1/platform/devices/musa/event_ipc.py    # MUSA capability gate and adapter
lmcache/v1/platform/devices/musa/ipc_wrapper.py  # MUSA availability and torch.musa loader
```

### Default backend

`DefaultEventIPCBackend` adapts an event module with the CUDA-style API:

```python
event = event_module.Event(interprocess=True)
handle = event.ipc_handle()
remote_event = event_module.Event.from_ipc_handle(device, handle)
event.record(stream)
remote_event.wait(stream)
remote_event.query()
remote_event.synchronize()
```

The base `DeviceSpec.event_ipc_backend` returns `None`. `CpuDeviceSpec`
explicitly binds this adapter to `StubCPUDevice`, while `CudaDeviceSpec` binds
it to `torch.cuda`. Other accelerators must explicitly register a compatible
backend before the multiprocess handle path can use them. This prevents an
unsupported or misspelled device type from silently selecting the active
accelerator's event implementation.

### ROCm backend

`RocmEventIPCBackend` (`lmcache/v1/platform/devices/rocm/event_ipc.py`) is
`DefaultEventIPCBackend` over `torch.cuda` (HIP) with
`host_ready_before_imported_wait = True`. `RocmDeviceSpec.event_ipc_backend`
returns it when isolated IPC is off. With isolated IPC, the CUDA selection (the
timeline-semaphore backend) applies unchanged.

### MUSA backend

`MusaEventIPCBackend` lives under `lmcache/v1/platform/devices/musa/`. It first checks
`is_musa_event_ipc_available()` from `musa/ipc_wrapper.py`, then adapts the
current TorchMUSA event API through `DefaultEventIPCBackend`:

```python
torch_musa = get_torch_musa_module()
event = torch_musa.Event(interprocess=True)
handle = event.ipc_handle()
remote_event = torch_musa.Event.from_ipc_handle(device, handle)
```

The generic layer does not import `platform.devices.musa`; only the MUSA `DeviceSpec`
and backend do. If the opt-in MUSA event API is unavailable,
`check_event_support()` raises a device-named `RuntimeError`. Memory IPC and
server-side block transfer are separate capabilities and are not required to
validate the standalone event backend.

## Registration

Event backends are properties of `DeviceSpec`, alongside device detection and
memory IPC wrapper selection:

```python
class DeviceSpec:
    @property
    def event_ipc_backend(self) -> EventIPCBackend | None:
        return None

class CudaDeviceSpec(DeviceSpec):
    @property
    def event_ipc_backend(self) -> EventIPCBackend:
        # TimelineSemaphoreEventIPCBackend when isolated IPC is enabled
        # (the default), else DefaultEventIPCBackend(event_module=torch.cuda)
        return _select_event_ipc_backend(self.device_type)

class RocmDeviceSpec(CudaDeviceSpec):
    @property
    def event_ipc_backend(self) -> EventIPCBackend:
        # The CUDA selection under isolated IPC, else RocmEventIPCBackend()
        ...

class MusaDeviceSpec(DeviceSpec):
    @property
    def event_ipc_backend(self) -> EventIPCBackend:
        return MusaEventIPCBackend()
```

This keeps platform capabilities together. Adding another backend requires a
new platform package/spec, not an edit to the multiprocess transfer modules.

CUDA selects between two backends via the process-global isolated-IPC switch
(`lmcache.v1.platform.ipc_policy`, default off): the timeline-semaphore
backend works across containers that share no host IPC namespace or
`/dev/shm` (see `../cuda/timeline_semaphore_event_ipc.md`), while
`DefaultEventIPCBackend` uses CUDA interprocess event handles. The switch is
set at process initialization from `lmcache.mp.isolated_ipc` (vLLM connector
extra_config), `--isolated-ipc` (MP server CLI) and `lmcache bench server
--isolated-ipc`, before the first backend
resolution; the selection is module-level, not spec-instance state, so every
`DeviceSpec` instance in the process resolves identically.

## Generic Multiprocess Flow

```text
Worker adapter
  after mode/config selection: resolve and validate event_backend once
  event_backend.create_event / record_event / export_event
  send event handle in STORE/RETRIEVE

Server: lmcache_driven_transfer.py (and blend/retrieve.py, plugin_modules/qstore.py)
  at KV registration: resolve, validate, and cache event_backend in ContextEntry
  event_backend.import_event; wait_producer_event (fail the transfer on give-up)
  enqueue KV transfer
  event_backend.create_event / record_event / export_event

Worker: futures.py
  reuse the transfer context's cached event_backend
  event_backend.import_event / query_event / wait_event / synchronize_event
```

Backend lookup and capability validation belong to initialization or cache
registration, after process-wide IPC configuration is finalized. Request hot
paths call the cached backend directly; they do not repeat platform discovery
or support probing for each event operation.

The following generic modules use only the platform API:

- `lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py`
- `lmcache/v1/multiprocess/modules/blend/retrieve.py`
- `lmcache/v1/multiprocess/plugin_modules/qstore.py`
- `lmcache/v1/transport/futures.py`
- `lmcache/v1/client/worker_transfer.py`

`DeviceMessagingFuture` is the device-neutral implementation. The old
`CUDAMessagingFuture` name remains an alias, and `to_cuda_future()` remains an
alias for `to_device_future()`.

## Error Contract

`event_backend.check_event_support(device)` runs once during initialization or
registration, before any cross-process memory transfer.
The default backend rejects missing `Event`, missing `interprocess` support, or
missing `Event.from_ipc_handle`. The MUSA backend additionally rejects a
disabled/unavailable MUSA event API, a missing TorchMUSA module, or an event
module without the required interprocess API. For opaque C/pybind event
bindings, capability detection may probe `Event(interprocess=True)` when a
Python signature is unavailable.

Errors include the backend name and missing capability. They must not claim
that CUDA is required when a non-CUDA backend has its own implementation.

## Migration Plan

1. Define `EventIPCBackend`, the default adapter, and lookup in
   `platform/base/event_ipc.py`.
2. Add the optional `DeviceSpec.event_ipc_backend` capability and explicit
   CPU, CUDA, and MUSA registrations.
3. Replace direct event creation/import/wait/record/export/query calls in the
   `lmcache_driven` server path with the platform API.
4. Route `DeviceMessagingFuture` import/query/wait/synchronize through the
   platform API while preserving the CUDA aliases.
5. Route worker-side producer-event export through the platform API.
6. Route every server-side wait on an imported engine event (lmcache_driven
   STORE / RETRIEVE, the CacheBlend retrieve, the Q store's `store_q` and SDK
   ring wait) through `wait_producer_event`, and register the ROCm backend.

## Testing

Tests cover:

- default event creation, export, import, record, wait, query, synchronize;
- CPU backend selection while another accelerator is active;
- strict failures for unregistered devices and unsupported event IPC;
- MUSA capability failure and missing-module failure;
- MUSA delegation to the TorchMUSA event API using a fake module;
- future compatibility aliases and device-aware event behavior;
- structural absence of direct MUSA imports in the generic handle-path modules;
- `wait_producer_event`: poll-before-wait on a host-ready backend, no polling
  elsewhere, slow-down and warning, give-up at the deadline and when the
  engine is gone, no liveness check on the fast path;
- ROCm backend selection with and without isolated IPC;
- server transfers (lmcache_driven STORE / RETRIEVE, CacheBlend retrieve,
  `store_q`, SDK transfers) failing without a stream wait when the engine is
  reaped mid-wait.

Real MUSA hardware tests remain capability-gated. The platform tests do not
need MUSA hardware because they validate the public backend contract using
injected event modules.

## Review Checklist

- [ ] Generic handle-path modules do not import `platform.devices.musa`.
- [ ] Generic handle-path modules do not branch on a concrete device type.
- [ ] `CUDAMessagingFuture` and `to_cuda_future` remain compatible aliases.
- [ ] Concrete device specs explicitly register their event IPC backends.
- [ ] Missing specs and unsupported event IPC fail without fallback.
- [ ] MUSA event behavior is isolated under `platform/devices/musa`.
- [ ] Unsupported event IPC fails before unsafe memory access.
- [ ] Platform and future tests cover import/export and stream ordering.
- [ ] Server code waits on imported engine events only through
      `wait_producer_event` and fails the transfer on `ProducerEventWaitError`.
