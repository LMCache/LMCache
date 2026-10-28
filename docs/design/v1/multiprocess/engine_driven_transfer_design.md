# Engine-Driven Transfer Design (Multiprocess Mode)

## 1. Motivation

LMCache multiprocess mode originally depended on CUDA IPC: workers send IPC
handles, and the server reads/writes worker GPU memory directly. That path
(now exposed as the **lmcache-driven** transfer mode) works well on CUDA, but
the required primitives are CUDA-specific (IPC memory handles, interprocess
CUDA events, CUDA stream semantics).

For **CPU, XPU, HPU, and other non-CUDA devices**, those primitives do not
exist. The **engine-driven** transfer mode introduces a device-agnostic path
where the engine side (worker adapter) gathers/scatters KV through CPU chunks
instead of CUDA IPC handles, then commits the bytes to the server.

Goal: keep the existing lmcache-driven (CUDA IPC) path unchanged while adding
a second engine-driven path that works across non-CUDA backends.

## 2. Design

### 2.1 Architecture Overview

```text
Worker adapter (vLLM MP adapter)
  └─ TransferContext (transfer_context/worker_transfer.py)
      ├─ LMCacheDrivenTransferContext  (IPC path via stream/event)
      └─ EngineDrivenTransferContext    (data path via data copying in adapter)
          ├─ AsyncEngineDrivenTransferContext (async_engine_driven.py)
          │    (store-only async: all three phases run in background thread pool)
          └─ EngineDrivenContext (transfer_context/base.py)
             ├─ EngineDrivenContextInline (safe inline tensor codec)
             └─ EngineDrivenContextShm    (transfer_context/shm.py)

MPCacheServer (server)
├─ MPCacheServerContext (engine_context.py)
│    ├─ StorageManager
│    ├─ TokenHasher
│    ├─ SessionManager
│    ├─ EventBus
│    ├─ LayoutDescRegistry
│    └─ shm_pool_info (pre-computed once)
└─ EngineDrivenTransferModule (modules/engine_driven_transfer.py)
     └─ TransferStrategy (modules/server_transfer.py)
          ├─ InlineTransferStrategy (safe inline tensor codec)
          └─ ShmTransferStrategy
```

State machine overview (worker-side):

```text
                       create_transfer_context()
                                 |
                 +---------------+---------------+
                 |                               |
                 v                               v
      LMCacheDrivenTransferContext    EngineDrivenTransferContext
          (device declares              (device declares
           lmcache-driven;               engine-driven;
           CUDA, NPU)                    sync fallback)
                 |                       AsyncEngineDrivenTransferContext
                 |                       (engine-driven, async primitives available)
                 |                               |
                 v                               v
              register()                      register()
                 |                               |
                 +---------------+---------------+
                                 |
                                 v
                                READY
                                 |
                 +---------------+-------------------------------+
                 |                                               |
                 v                                               v
    submit_store (lmcache-driven path)         submit_store (engine-driven path)
    -> STORE request (async)          sync:    -> prepare_store -> gather -> commit_store
                                      async:   -> background thread:
                                                    prepare_store -> gather (copy stream)
                                                    -> commit_store
                                                  returns unresolved future
                 |                                               |
                 +---------------+-------------------------------+
                                 |
                                 v
                                READY
                                 |
                 +---------------+-------------------------------+
                 |                                               |
                 v                                               v
  submit_retrieve (lmcache-driven path)      submit_retrieve (engine-driven path)
  -> RETRIEVE request (async)                 -> prepare_retrieve -> scatter -> commit_retrieve
                 |                                               |
                 +---------------+-------------------------------+
                                 |
                                 v
                                READY
                                 |
                                 v
                               close()
```

Overall data flow:
- **lmcache-driven path** (CUDA IPC): worker sends a handle, server pulls/pushes
  data directly via device memory.
- **engine-driven path** (CPU/SHM/inline): worker gathers/scatters paged KV and
  exchanges CPU-side data via a transport-specific `EngineDrivenContext`
  implementation.

### 2.2 Worker Side: TransferContext

`TransferContext` is the worker-side transport abstraction with four methods:
`register`, `submit_store`, `submit_retrieve`, and `close`.
The contract is intentionally minimal so worker adapters only depend on these
four lifecycle and transfer operations.

- **LMCacheDrivenTransferContext** keeps the original CUDA IPC behavior:
  worker sends a handle and server performs direct GPU-side transfer.
- **EngineDrivenTransferContext** is the engine-driven (non-CUDA) path:
  worker transfers actual data chunks through `EngineDrivenContext`.
- **AsyncEngineDrivenTransferContext** extends `EngineDrivenTransferContext` to make
  the store path fully asynchronous. `submit_retrieve` is unchanged (synchronous);
  only `submit_store` becomes async.

`EngineDrivenTransferContext` flows:
- **submit_store**: `prepare_store` → `gather_paged_kv_to_cpu` → `commit_store`
- **submit_retrieve**: `prepare_retrieve` → `scatter_cpu_to_paged_kv` → `commit_retrieve`

`AsyncEngineDrivenTransferContext` async store flow:
- **submit_store**: performs only O(1) work on the forward thread (registration check
and block-id flattening), then submits all three phases to a background
`ThreadPoolExecutor` (`commit_executor`) and returns an unresolved
`MessagingFuture`
- **submit_retrieve**: `prepare_retrieve` → `scatter_cpu_to_paged_kv` → `commit_retrieve`

During `register`, worker receives `RegisterEngineDrivenContextResponse(shm_name, pool_size)`
from server and then calls `create_engine_driven_context(...)` to construct
`EngineDrivenContextInline` or `EngineDrivenContextShm`.

Why `prepare → data operation → commit`:
- `prepare_*`: set up transport state (for SHM this allocates/returns shared buffers;
  for inline transfer it is a protocol RPC that does not allocate buffers).
- gather/scatter: worker-local data movement between paged KV and contiguous
  CPU chunks, performed between protocol phases.
- `commit_*`: finalize and notify server to consume or release transfer state.

`create_transfer_context()` selects the implementation once based on the
resolved mode and async capability. An explicit mode (extra_config /
`LMCACHE_MP_TRANSFER_MODE`) wins; `auto` resolves to the device spec's
declared default (`DeviceSpec.default_mp_transfer_mode()` — the base
derives it from `is_lmcache_driven_available()`, so CUDA and NPU get
lmcache-driven while opt-in-only stacks and every other device stay
engine-driven):
- Resolved lmcache-driven → `_build_lmcache_driven_context()`
- Resolved engine-driven → `_build_engine_driven_context()`, which probes async primitives:
  - async primitives available (stream, event with record/synchronize/wait, pin_memory) →
    `AsyncEngineDrivenTransferContext`
  - otherwise → `EngineDrivenTransferContext` (synchronous fallback)

It also validates that all KV cache tensors share one device type and rejects
mixed-device configurations by raising an error.

| Context | What is transferred | Who performs copy work | Completion style |
|---|---|---|---|
| LMCacheDrivenTransferContext | Device handle/reference | Server pulls/pushes via IPC | Async MQ future |
| EngineDrivenTransferContext | Actual CPU chunk data | Worker gather/scatter + transport commit | Synchronous worker-side flow |
| AsyncEngineDrivenTransferContext | Actual CPU chunk data | Worker gather/scatter (copy stream) + background commit | Async future (resolved in background thread) |

### 2.3 Server Side: LMCache-Driven Module vs Engine-Driven Module

- **LMCache-driven module (existing path):** server uses `LMCacheDrivenTransferModule`
  with CUDA IPC handles to access worker device memory directly.
- **Engine-driven module:** server uses `EngineDrivenTransferModule`, which
  stores per-instance `EngineDrivenContextEntry` metadata and delegates transfer
  logic to a `TransferStrategy`.

Server transfer strategy implementations:
- **InlineTransferStrategy**: typed inline tensor prepare/commit behavior.
- **ShmTransferStrategy**: SHM slot-based prepare/commit behavior, with the safe
  typed codec as fallback when inline bytes are provided.

This mirrors the worker split (`EngineDrivenContextInline` / `EngineDrivenContextShm`):
both sides keep common request flow while isolating transport-specific logic.

`MPCacheServerContext` is the shared container injected into modules at init.
It also computes `shm_pool_info` once from `StorageManagerConfig`:
- disable SHM when `shm_name` is empty or `use_lazy=True`
- normalize name (`lstrip("/")` and enforce `lmcache_l1_pool_` prefix)
- keep final `{shm_name, pool_size}` for all later registrations

### 2.4 Transport Comparison

**Store (worker → server storage):**

| Transport | Copies | Data flow |
|---|---|---|
| LMCache-driven (CUDA IPC) | 2 | GPU KV → GPU staging buffer → CPU memory object |
| Inline | 4 | GPU KV → CPU chunk → serialize → deserialize → CPU memory object |
| SHM | 1 | GPU KV → CPU memory object (SHM mapped) |

**Retrieve (server storage → worker):**

| Transport | Copies | Data flow |
|---|---|---|
| LMCache-driven (CUDA IPC) | 2 | CPU memory object → GPU staging buffer → GPU KV |
| Inline | 4 | CPU memory object → serialize → deserialize → CPU chunk → GPU KV |
| SHM | 1 | CPU memory object (SHM mapped) → GPU KV |

| Transport | Pros | Cons | Best fit |
|---|---|---|---|
| LMCache-driven (CUDA IPC) | Mature path, good async overlap | CUDA-only | NVIDIA CUDA deployments |
| Inline | Works everywhere, no SHM setup | Extra serialization + copy overhead | Universal fallback |
| SHM | Lowest copy count, no serialization | Requires enough `/dev/shm` and synchronization | High-throughput engine-driven setups |

### 2.5 Current File Layout (Key Components)

- `lmcache/v1/multiprocess/modules/engine_driven_transfer.py`: `EngineDrivenTransferModule`
- `lmcache/v1/multiprocess/modules/server_transfer.py`: `TransferStrategy`, `InlineTransferStrategy`, `ShmTransferStrategy`
- `lmcache/v1/multiprocess/transfer_context/worker_transfer.py`: `EngineDrivenTransferContext`, `LMCacheDrivenTransferContext`, `create_transfer_context`, `MPTransferMode`
- `lmcache/v1/multiprocess/transfer_context/async_engine_driven.py`: `AsyncEngineDrivenTransferContext`
- `lmcache/v1/multiprocess/transfer_context/base.py`: `EngineDrivenContext`, `gather_paged_kv_to_cpu`, `scatter_cpu_to_paged_kv`, `compute_kv_layout`
- `lmcache/v1/multiprocess/transfer_context/inline.py`:
  `EngineDrivenContextInline` using the safe inline tensor codec
- `lmcache/v1/multiprocess/transfer_context/tensor_wire.py`: versioned tensor
  metadata and raw-byte encoding with strict layout validation
- `lmcache/v1/multiprocess/transfer_context/shm.py`: `EngineDrivenContextShm`

## 3. Protocol & Data Flow

### 3.1 MQ Request Types Used by Engine-Driven Path

The engine-driven path uses five request types:

1. `REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT`  
   Worker registers engine-driven KV layout metadata. Server then:
   - stores `EngineDrivenContextEntry` (metadata + model/world info)
   - registers `MemoryLayoutDesc` in `LayoutDescRegistry`
   - creates `TransferStrategy` from engine-level `shm_pool_info`
   - returns `shm_name/pool_size` so worker creates matching `EngineDrivenContext`

2. `PREPARE_STORE`  
   Worker asks server/transport to prepare store-side transfer state.

3. `COMMIT_STORE`  
   Worker commits store data so server can persist it into storage.

4. `PREPARE_RETRIEVE`  
   Worker asks server to prepare retrieval payload/state for a key.

5. `COMMIT_RETRIEVE`  
   Worker acknowledges retrieval completion so transport state can be finalized.

### 3.2 Data Flow: Inline Tensor Path

Store:
1. Worker `prepare_store` RPC.
2. Worker gathers paged KV into CPU chunks.
3. Worker `commit_store` sends serialized bytes.
4. Server deserializes and writes to storage.

Retrieve:
1. Worker `prepare_retrieve` RPC.
2. Server reads from storage and returns serialized bytes.
3. Worker deserializes to CPU chunks.
4. Worker scatters chunks back to paged KV.
5. Worker `commit_retrieve` finalizes protocol state.

The inline payload is a versioned MessagePack envelope. Each tensor carries only
an allowlisted dtype name, shape, and raw contiguous bytes. Both endpoints require
the registered dtype and shape, an exact request-derived tensor count, exact byte
lengths, and bounded individual and cumulative sizes. Unknown versions or fields
and payloads from the previous untyped protocol are rejected without a fallback.
This wire change is intentionally fail closed, so workers and servers must upgrade
together.

```text
Store (inline tensor)
Worker: prepare_store --> Server
Worker: gather paged KV -> CPU chunks
Worker: commit_store(serialized bytes) --> Server
Server: deserialize -> storage write

Retrieve (inline tensor)
Worker: prepare_retrieve --> Server
Server: read storage -> serialize bytes
Server: serialized bytes --> Worker
Worker: deserialize -> scatter to paged KV
Worker: commit_retrieve --> Server
```

### 3.3 Data Flow: SHM Path

Store:
1. Worker `prepare_store` gets `slots` and `chunk_indices`.
2. Server includes only chunks that still need writes (already-cached chunks are skipped).
3. Worker gathers only `chunk_indices` into SHM-backed buffers.
4. Worker `commit_store` notifies server to finalize reserved write locks.

If all chunks are already cached, server returns empty `slots/chunk_indices` and
worker short-circuits store as success (no gather, no commit payload).

Retrieve:
1. Worker `prepare_retrieve` asks server to populate SHM.
2. Server reads from storage and returns SHM slot descriptors.
3. Worker scatters from SHM-backed buffers into paged KV.
4. Worker `commit_retrieve` releases/read-completes SHM state.

Notes:
- SHM pool metadata is computed once in `MPCacheServerContext` init, not per registration.
- `chunk_indices` optimization reduces unnecessary gather/copy work on partial cache hits.

## 4. Observability

The module publishes transfer events so the L0↔L1 throughput histograms, the
extra-stats log and the continuous usage report cover the engine-driven
transport too (before, they were empty exactly where the transfer cost is
highest — the devices that have no other transport). See
[EVENTS.md](../mp_observability/EVENTS.md) for the metadata contract and the
ways an engine-driven END differs from a GPU one.

- **Everything is published at `commit_*`; `prepare_*` publishes nothing.**
  `commit_store` / `commit_retrieve` publish `MP_*_SUBMITTED` immediately
  followed by `MP_*_END`. There is no `MP_*_START`.

  The GPU path brackets a transfer with START and END because its
  subscribers infer the duration from the timestamp difference. This path
  measures the window itself and reports `duration_seconds`, which both
  subscribers prefer, so a START would add no information — only correlation
  state. And that state is unreclaimable here: a prepared transfer is not
  guaranteed to commit, and the common abandonment is a worker whose gather
  raised, which keeps pinging and so is *never reaped*. Nothing on the
  publisher side can reach what such a START left behind — a pending entry
  in `L0L1ThroughputSubscriber` / `ExtraStatsLoggingSubscriber`, an inflated
  submitted-minus-finished difference, an open `mp.store` span. Measured:
  4200 abandoned prepares retained 4200 entries in each of those four
  structures, against a 4096-entry publisher cap. Publishing only once the
  transfer is known to have completed leaves nothing to strand.

  SUBMITTED is kept, paired with the END, so the submitted/finished counters
  stay balanced; feeding only the END side would drive the in-flight
  subtraction negative. It reads zero in flight, which is honest — the
  server learns of an engine-driven transfer only once it is done.

  Cost: no `mp.store` / `mp.retrieve` child span for engine-driven
  transfers. That span is currently wrong for *any* overlapping transfers,
  including on the GPU path — `MPServerTracingSubscriber` keys pending spans
  by `(session_id, event_type)`, so two ranks of one request cross-wire
  their durations and leak a span. Emitting nothing is better than emitting
  that; the span becomes available once the subscriber keys per transfer.
- A store that prepared nothing — every chunk already cached, or all-null in
  every group — records no start, because the worker then returns without
  committing and the entry would never be popped.
- Each END carries `duration_seconds` measured by the server: the
  `prepare_*` → `commit_*` window, which is the worker's device copy plus
  the transport, and is what the matching `Stored/Retrieved N tokens` log
  line reports.
- Timing lives in a dict keyed by
  `(phase, instance_id, request_id, key.start, key.end)`, not in the
  per-request session: `IPCCacheServerKey` excludes `request_id` from
  equality, so two concurrent requests over the same token range would share
  one entry, and one rank can have several chunks of one request in flight,
  so the request alone is too coarse. It is capped
  (`_MAX_TRACKED_TRANSFERS`) because a prepared transfer is not guaranteed to
  commit and a live worker is never reaped; a reaped worker's entries are
  swept with its registration.
- An END whose entry is missing (reaped, unprepared, or evicted) omits
  `stored_count`/`retrieved_count`, `total_bytes`, `num_tokens` and
  `duration_seconds` rather than falling back to the requested range: an SHM
  store reserves only the chunks that were missing, so the range would
  report already-cached chunks as newly stored.
- Bytes come from whichever source the transport offers, and are always
  exact: SHM sums its reserved slots, inline takes the payload length
  (`len(cpu_data)` at `commit_store`, `len(response.data)` at
  `prepare_retrieve`). This matters because a non-CUDA accelerator with no
  SHM pool runs inline by default, and deriving nothing there left the
  throughput histograms empty on the one configuration with no alternative
  transport.
- Chunk counts are exact too, with one exception. A retrieve is
  all-or-nothing over keys resolved before the payload exists, so the count
  is the union of every group's served window — never group 0's length,
  which on a hybrid model is a recurrent group's single trailing chunk and
  would log a 22-chunk restore as one. A multi-group **inline store** is the
  exception: it omits chunks that are all-null in a group without telling the
  server which, so its count and `num_tokens` are omitted rather than
  guessed from the range. Its bytes are still reported.
