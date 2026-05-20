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
  └─ TransferContext
      ├─ HandleTransferContext  (CUDA IPC path)
      └─ DataTransferContext    (non-CUDA data path)
          └─ NonGpuContext
             ├─ NonGpuContextPickle
             └─ NonGpuContextShm (TODO)
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
- **Non-GPU Context:** server participates in two separate two-phase protocols
  exposed by `NonGpuContext`: `prepare_store/commit_store` for store, and
  `prepare_retrieve/commit_retrieve` for retrieve, plus lifecycle cleanup via
  `close`.

`NonGpuContext` implementations:
- **NonGpuContextPickle**: serialize/deserialize chunk payloads with pickle.
- **NonGpuContextShm**: shared-memory transport (planned/TODO).

This split keeps server protocol stable while allowing transport-specific behavior
behind one interface contract.

### 2.4 Transport Comparison

**Store (worker → server storage):**

| Transport | Copies | Data flow |
|---|---|---|
| Handle (CUDA IPC) | 2 | GPU KV → GPU staging buffer → CPU memory object |
| Pickle | 4 | GPU KV → CPU chunk → serialize → deserialize → CPU memory object |
| SHM (TODO) | 1 | GPU KV → CPU memory object (SHM mapped) |

**Retrieve (server storage → worker):**

| Transport | Copies | Data flow |
|---|---|---|
| Handle (CUDA IPC) | 2 | CPU memory object → GPU staging buffer → GPU KV |
| Pickle | 4 | CPU memory object → serialize → deserialize → CPU chunk → GPU KV |
| SHM (TODO) | 1 | CPU memory object (SHM mapped) → GPU KV |

| Transport | Pros | Cons | Best fit |
|---|---|---|---|
| Handle (CUDA IPC) | Mature path, good async overlap | CUDA-only | NVIDIA CUDA deployments |
| Pickle | Works everywhere, no SHM setup | Extra serialization + copy overhead | Universal fallback |
| SHM (TODO) | Lowest copy count, no serialization | Requires enough `/dev/shm` and synchronization | High-throughput non-CUDA setups |

## 3. Protocol & Data Flow

### 3.1 MQ Request Types Used by Non-GPU Path

The non-GPU path uses five request types:

1. `REGISTER_KV_CACHE_NON_GPU_CONTEXT`  
   Worker registers non-CUDA KV layout metadata so the server can reconstruct
   the worker KV memory layout for store/retrieve operations.

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

### 3.3 Data Flow: SHM Path (TODO)

Store:
1. Worker `prepare_store` obtains SHM slot/offset.
2. Worker gathers directly into SHM-backed buffers.
3. Worker `commit_store` notifies server to consume SHM data.

Retrieve:
1. Worker `prepare_retrieve` asks server to populate SHM.
2. Server writes retrieved chunks into SHM.
3. Worker scatters from SHM-backed buffers into paged KV.
4. Worker `commit_retrieve` releases/read-completes SHM state.
