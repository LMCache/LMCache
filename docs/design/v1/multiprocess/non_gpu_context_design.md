# Non-GPU Context Design (Multiprocess Mode)

## 1. Motivation

LMCache multiprocess mode originally depended on CUDA IPC: workers send IPC handles,
and the server reads/writes worker GPU memory directly. That path works well on
CUDA, but the required primitives are CUDA-specific (IPC memory handles,
interprocess CUDA events, CUDA stream semantics).

For **CPU, XPU, HPU, and other non-CUDA devices**, those primitives do not exist.
The non-GPU context design introduces a device-agnostic path where workers move KV
data through CPU chunks instead of CUDA IPC handles.

Goal: keep the existing CUDA path unchanged while adding a second path that works
across non-CUDA backends.

## 2. Design

### 2.1 Architecture Overview

```text
Worker adapter (vLLM MP adapter)
  └─ TransferContext (transfer_context/worker_transfer.py)
      ├─ HandleTransferContext  (IPC path via stream/event)
      └─ DataTransferContext    (data path via data copying in adapter)
          └─ NonGpuContext (transfer_context/base.py)
             ├─ NonGpuContextPickle (transfer_context/pickle.py)
             └─ NonGpuContextShm    (transfer_context/shm.py)

MPCacheEngine (server)
  └─ MPCacheEngineContext (engine_context.py)
      ├─ StorageManager
      ├─ TokenHasher
      ├─ SessionManager
      ├─ EventBus
      ├─ LayoutDescRegistry
      └─ shm_pool_info (pre-computed once)
  └─ NonGPUTransferModule (modules/non_gpu_transfer.py)
      └─ TransferStrategy (modules/server_transfer.py)
         ├─ PickleTransferStrategy
         └─ ShmTransferStrategy
```

State machine overview (worker-side):

```text
                       create_transfer_context()
                                 |
                 +---------------+---------------+
                 |                               |
                 v                               v
      HandleTransferContext            DataTransferContext
          (device == CUDA)            (device != CUDA)
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
    submit_store (handle path)                  submit_store (data path)
    -> STORE request (async)                    -> prepare_store -> gather -> commit_store
                 |                                               |
                 +---------------+-------------------------------+
                                 |
                                 v
                                READY
                                 |
                 +---------------+-------------------------------+
                 |                                               |
                 v                                               v
  submit_retrieve (handle path)               submit_retrieve (data path)
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
- **CUDA path**: worker sends a handle, server pulls/pushes data directly.
- **Non-CUDA path**: worker gathers/scatters paged KV and exchanges CPU-side data
  via a transport-specific `NonGpuContext` implementation.

### 2.2 Worker Side: TransferContext

`TransferContext` is the worker-side transport abstraction with four methods:
`register`, `submit_store`, `submit_retrieve`, and `close`.
The contract is intentionally minimal so worker adapters only depend on these
four lifecycle and transfer operations.

- **HandleTransferContext** keeps the original CUDA IPC behavior:
  worker sends a handle and server performs direct GPU-side transfer.
- **DataTransferContext** is the non-CUDA path:
  worker transfers actual data chunks through `NonGpuContext`.

`DataTransferContext` flows:
- **submit_store**: `prepare_store` → `gather_paged_kv_to_cpu` → `commit_store`
- **submit_retrieve**: `prepare_retrieve` → `scatter_cpu_to_paged_kv` → `commit_retrieve`

During `register`, worker receives `RegisterNonGpuContextResponse(shm_name, pool_size)`
from server and then calls `create_non_gpu_context(...)` to construct
`NonGpuContextPickle` or `NonGpuContextShm`.

Why `prepare → data operation → commit`:
- `prepare_*`: set up transport state (for SHM this allocates/returns shared buffers;
  for pickle it is a protocol RPC that does not allocate transfer buffers).
- gather/scatter: worker-local data movement between paged KV and contiguous
  CPU chunks, performed between protocol phases.
- `commit_*`: finalize and notify server to consume or release transfer state.

`create_transfer_context()` selects the implementation once based on device type
(CUDA → `HandleTransferContext`, otherwise → `DataTransferContext`).
It also validates that all KV cache tensors share one device type and rejects
mixed-device configurations by raising an error.

| Context | What is transferred | Who performs copy work | Completion style |
|---|---|---|---|
| HandleTransferContext | Device handle/reference | Server pulls/pushes via IPC | Async MQ future |
| DataTransferContext | Actual CPU chunk data | Worker gather/scatter + transport commit | Synchronous worker-side flow |

### 2.3 Server Side: GPU Context vs Non-GPU Context

- **GPU Context (existing path):** server uses CUDA IPC handles to access worker
  device memory directly.
- **Non-GPU Context:** server uses `NonGPUTransferModule`, which stores
  per-instance `NonGPUContextEntry` metadata and delegates transfer logic to a
  `TransferStrategy`.

Server transfer strategy implementations:
- **PickleTransferStrategy**: pure pickle prepare/commit behavior.
- **ShmTransferStrategy**: SHM slot-based prepare/commit behavior, with pickle
  fallback when inline bytes are provided.

This mirrors the worker split (`NonGpuContextPickle` / `NonGpuContextShm`):
both sides keep common request flow while isolating transport-specific logic.

`MPCacheEngineContext` is the shared container injected into modules at init.
It also computes `shm_pool_info` once from `StorageManagerConfig`:
- disable SHM when `shm_name` is empty or `use_lazy=True`
- normalize name (`lstrip("/")` and enforce `lmcache_l1_pool_` prefix)
- keep final `{shm_name, pool_size}` for all later registrations

### 2.4 Transport Comparison

**Store (worker → server storage):**

| Transport | Copies | Data flow |
|---|---|---|
| Handle (CUDA IPC) | 2 | GPU KV → GPU staging buffer → CPU memory object |
| Pickle | 4 | GPU KV → CPU chunk → serialize → deserialize → CPU memory object |
| SHM | 1 | GPU KV → CPU memory object (SHM mapped) |

**Retrieve (server storage → worker):**

| Transport | Copies | Data flow |
|---|---|---|
| Handle (CUDA IPC) | 2 | CPU memory object → GPU staging buffer → GPU KV |
| Pickle | 4 | CPU memory object → serialize → deserialize → CPU chunk → GPU KV |
| SHM | 1 | CPU memory object (SHM mapped) → GPU KV |

| Transport | Pros | Cons | Best fit |
|---|---|---|---|
| Handle (CUDA IPC) | Mature path, good async overlap | CUDA-only | NVIDIA CUDA deployments |
| Pickle | Works everywhere, no SHM setup | Extra serialization + copy overhead | Universal fallback |
| SHM | Lowest copy count, no serialization | Requires enough `/dev/shm` and synchronization | High-throughput non-CUDA setups |

### 2.5 Current File Layout (Key Components)

- `lmcache/v1/multiprocess/modules/non_gpu_transfer.py`: `NonGPUTransferModule`
- `lmcache/v1/multiprocess/modules/server_transfer.py`: `TransferStrategy`, `PickleTransferStrategy`, `ShmTransferStrategy`
- `lmcache/v1/multiprocess/transfer_context/worker_transfer.py`: `DataTransferContext`, `HandleTransferContext`
- `lmcache/v1/multiprocess/transfer_context/base.py`: `NonGpuContext`, `gather_paged_kv_to_cpu`, `scatter_cpu_to_paged_kv`, `compute_kv_layout`
- `lmcache/v1/multiprocess/transfer_context/pickle.py`: `NonGpuContextPickle`
- `lmcache/v1/multiprocess/transfer_context/shm.py`: `NonGpuContextShm`

## 3. Protocol & Data Flow

### 3.1 MQ Request Types Used by Non-GPU Path

The non-GPU path uses five request types:

1. `REGISTER_KV_CACHE_NON_GPU_CONTEXT`  
   Worker registers non-CUDA KV layout metadata. Server then:
   - stores `NonGPUContextEntry` (metadata + model/world info)
   - registers `MemoryLayoutDesc` in `LayoutDescRegistry`
   - creates `TransferStrategy` from engine-level `shm_pool_info`
   - returns `shm_name/pool_size` so worker creates matching `NonGpuContext`

2. `PREPARE_STORE`  
   Worker asks server/transport to prepare store-side transfer state.

3. `COMMIT_STORE`  
   Worker commits store data so server can persist it into storage.

4. `PREPARE_RETRIEVE`  
   Worker asks server to prepare retrieval payload/state for a key.

5. `COMMIT_RETRIEVE`  
   Worker acknowledges retrieval completion so transport state can be finalized.

### 3.2 Data Flow: Pickle Path

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

```text
Store (pickle)
Worker: prepare_store --> Server
Worker: gather paged KV -> CPU chunks
Worker: commit_store(serialized bytes) --> Server
Server: deserialize -> storage write

Retrieve (pickle)
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
- SHM pool metadata is computed once in `MPCacheEngineContext` init, not per registration.
- `chunk_indices` optimization reduces unnecessary gather/copy work on partial cache hits.
