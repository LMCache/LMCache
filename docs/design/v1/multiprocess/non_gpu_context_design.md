# Non-GPU Context Design (MP mode, non-CUDA)

## 1. Motivation

LMCache multiprocess mode relies on **CUDA IPC** to transfer KV cache data
between vLLM worker processes and the LMCache cache server. The existing
path wraps GPU tensors in `CudaIPCWrapper`, exchanges IPC handles via ZMQ
messages, and uses CUDA events for cross-process synchronisation.

This design is fundamentally tied to the CUDA programming model:

| CUDA IPC dependency | Why it blocks non-CUDA devices |
|---|---|
| `CudaIPCWrapper` / `cudaIpcGetMemHandle` | Only works on NVIDIA CUDA tensors |
| `torch.cuda.Event(interprocess=True)` | CUDA-specific IPC event API |
| `cupy.cuda.ExternalStream` | CUDA stream wrapper |
| GPU pointer arithmetic in C++ kernels | Assumes CUDA device pointers |

For non-CUDA accelerators — **CPU, Intel XPU, Habana HPU**, or any future
device — none of these primitives are available.

The **non-GPU context** path introduces a device-agnostic KV transfer mechanism:

1. Workers **gather** paged KV blocks into contiguous CPU chunk tensors.
2. CPU chunks are **transported** to the server through a pluggable
   serialisation layer (pickle today, shared memory in the future).
3. On retrieve, the server returns CPU chunks and workers **scatter** them
   back into device-local paged KV tensors.

The existing CUDA IPC path is **untouched** — the two paths coexist behind a
polymorphic `TransferContext` abstraction.

### Transport comparison

**Store (worker → server storage):**

| Transport | Copies | Data flow |
|---|---|---|
| CUDA IPC | 2 | GPU KV → GPU staging buffer → CPU memory obj |
| Pickle | 4 | GPU KV → CPU chunk → pickle.dumps → pickle.loads → CPU memory obj |
| SHM (TODO) | 1 | GPU KV → CPU memory obj (SHM mapped) |

**Retrieve (server storage → worker):**

| Transport | Copies | Data flow |
|---|---|---|
| CUDA IPC | 2 | CPU memory obj → GPU staging buffer → GPU KV |
| Pickle | 4 | CPU memory obj → pickle.dumps → pickle.loads → CPU chunk → GPU KV |
| SHM (TODO) | 1 | CPU memory obj (SHM mapped) → GPU KV |

**Applicability:**

| Transport | Platform requirement | Pros | Cons |
|---|---|---|---|
| CUDA IPC | NVIDIA CUDA devices only | Async GPU streams, mature path | CUDA-only |
| Pickle | Any device, no dependencies | Generally available, zero setup | 4 copies + serialisation overhead |
| SHM (TODO) | `/dev/shm` capacity ≥ L1 cache size | Fewest copies (1), no serialisation | Requires sufficient shared memory |

## 2. Architecture Overview

### 2.1 Layered architecture

```
vllm_multi_process_adapter.py    ← Engine adapter, device-agnostic
  └── TransferContext             ← Worker-side transport abstraction (§3)
        ├── CudaTransferContext    ← CUDA IPC + MQ future path
        └── NonCudaTransferContext     ← Synchronous gather/scatter path
              └── NonGpuContext        ← Serialisation abstraction (§4.2)
                    ├── NonGpuContextPickle   ← pickle.dumps/loads (§4.3)
                    └── NonGpuContextShm      ← shared memory (§4.4, TODO)
```

Two layers of abstraction serve different purposes:

- **TransferContext** (§3) — decides **CUDA vs non-CUDA** routing at the
  worker adapter level.
- **NonGpuContext** (§4.2) — decides **how** CPU chunk data is serialised and
  transported (pickle vs SHM). Only used inside `NonCudaTransferContext`.

### 2.2 State machine (worker ↔ server)

```text
                           register_kv_caches()
                                      |
                                      v
                    create_transfer_context(kv_caches)
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
              [device == cuda]                 [device != cuda]
                     |                                 |
                     v                                 v
      CudaTransferContext.register()     NonCudaTransferContext.register()
      → REGISTER_KV_CACHE               → REGISTER_KV_CACHE_NON_GPU_CONTEXT
        (CUDA IPC handles)                 (scalar metadata fields)
                     |                         + create_non_gpu_context()
                     +----------------+----------------+
                                      |
                                      v
                              [READY / SERVING]
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
       transfer_ctx.submit_store()      transfer_ctx.submit_store()
                     |                                 |
                     v                                 v
           STORE (GPU → L1)            gather_paged_kv_to_cpu()
           [async MQ future]           + _non_gpu_context.prepare_store()
                     |                 + _non_gpu_context.commit_store() [sync]
                     v                         _store_done[id] = ok
                 [READY]                               |
                     +----------------+----------------+
                                      |
                                      v
      transfer_ctx.submit_retrieve()  +  poll_finished()
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
          RETRIEVE (L1 → GPU)     _non_gpu_context.prepare_retrieve() [sync]
          [async MQ future]       + scatter_cpu_to_paged_kv()
                     |            + _non_gpu_context.commit_retrieve()
                     v            _retrieve_done[id] = (ok, block_ids)
                     +----------------+----------------+
                                      |
                                      v
                              [READY / SERVING]
                                      |
                                      v
                           unregister_kv_cache()
                                      |
                                      v
                                  [TERMINATED]
```

## 3. Worker-side: TransferContext Abstraction

### 3.1 Problem

Before this refactoring, `vllm_multi_process_adapter.py` contained
non-CUDA-specific branching in every method — `register_kv_caches`,
`submit_store_request`, `submit_retrieve_request`, `get_finished`, and the
unhealthy drain path. Adding a third transport would require touching every
branch.

### 3.2 Solution

`transfer_context.py` defines the `TransferContext` ABC with six methods:
`register`, `submit_store`, `submit_retrieve`, `poll_finished`, `drain_all`,
and `close`. The adapter holds a single `TransferContext` and delegates —
no `if/else` anywhere.

### 3.3 `create_transfer_context()` factory

Inspects device types of all KV cache tensors **exactly once**. CUDA →
`CudaTransferContext`; otherwise → `NonCudaTransferContext`. Mixed device types
are rejected.

### 3.4 `CudaTransferContext`

Wraps the original CUDA IPC path. Sends `REGISTER_KV_CACHE` / `STORE` /
`RETRIEVE` messages with IPC handles, tracks async MQ futures.
`poll_finished` queries futures; `drain_all` marks all pending as finished
for unhealthy shutdown. Semantics identical to pre-refactoring.

### 3.5 `NonCudaTransferContext`

Holds a `NonGpuContext` instance internally. Sends
`REGISTER_KV_CACHE_NON_GPU_CONTEXT` with scalar metadata. Store and retrieve
are **synchronous**: gather → prepare/commit, then record result in
`_store_done` / `_retrieve_done`. `poll_finished` simply drains these dicts.

## 4. Server-side: Non-GPU Context Protocol

### 4.1 Why GPU context and non-GPU context need different protocols

| | GPU context | non-GPU context |
|---|---|---|
| Registration | `REGISTER_KV_CACHE` — IPC handles | `REGISTER_KV_CACHE_NON_GPU_CONTEXT` — scalar fields |
| Store | `STORE` — event handle + block IDs, server reads GPU directly | `STORE_CPU_CHUNKS` — serialised CPU tensors |
| Retrieve | `RETRIEVE` — event handle + block IDs, server writes GPU directly | `RETRIEVE_CPU_CHUNKS` — key lookup, returns CPU tensors |

Registration uses **scalar fields** (`block_size`, `num_layers`,
`hidden_dim_size`, `dtype_str`, `use_mla`) instead of pickled objects
to avoid cross-process pickle security and compatibility concerns. The
server reconstructs `MemoryLayoutDesc` from the scalars internally.

### 4.2 `NonGpuContext` ABC: two-phase prepare/commit

The serialisation layer is abstracted behind `NonGpuContext` so that pickle
and SHM can be swapped without touching `NonCudaTransferContext` or the server.

The ABC defines: `prepare_store`, `commit_store`, `prepare_retrieve`,
`commit_retrieve`, `close`.

Why two phases? Pickle can do everything in one step (prepare serialises,
commit sends). SHM needs prepare to allocate a slot, then the worker writes
into mapped memory, then commit tells the server "ready". The split
accommodates both without forcing unnecessary round-trips on pickle.

| Phase | Pickle | SHM (TODO) |
|---|---|---|
| `prepare_store` | `pickle.dumps(chunks)` → opaque handle | MQ `PREPARE_STORE` → get SHM offset → `memcpy` into SHM |
| `commit_store` | MQ `STORE_CPU_CHUNKS`, block for ack | MQ `COMMIT_STORE` → server reads from SHM |
| `prepare_retrieve` | MQ `RETRIEVE_CPU_CHUNKS` → `pickle.loads` | MQ `PREPARE_RETRIEVE` → server writes to SHM → map tensor views |
| `commit_retrieve` | no-op | MQ `FINISH_READ` → release SHM read lock |

`create_non_gpu_context()` factory currently always returns `NonGpuContextPickle`.
Future: probe `/dev/shm` availability and capacity, fall back to pickle if
insufficient.

## 5. Data Path: Gather / Scatter

### 5.1 Chunk format

- **Non-MLA**: `[2, num_layers, chunk_tokens, hidden_dim]` — dim 0 = `(K, V)`.
- **MLA**: `[num_layers, chunk_tokens, hidden_dim]` — single latent vector.

Where `chunk_tokens = blocks_per_chunk × block_size`.

### 5.2 Supported KV layouts

| Format enum | Layout | Shape per layer |
|---|---|---|
| `NL_X_TWO_NB_BS_NH_HS` | NHD | `[2, NB, BS, NH, HS]` |
| `NL_X_NB_TWO_BS_NH_HS` | NHD (flashinfer) | `[NB, 2, BS, NH, HS]` |
| `NL_X_TWO_NB_NH_BS_HS` | HND | `[2, NB, NH, BS, HS]` |
| `NL_X_NB_TWO_NH_BS_HS` | HND (flashinfer) | `[NB, 2, NH, BS, HS]` |
| `NL_X_NB_BS_HS` | MLA | `[NB, BS, HS]` |

### 5.3 Block-level indexing

Gather and scatter operate at **block granularity** (`tensor[block_ids]`)
rather than per-token `index_select` / `index_copy_`. For HND layouts, a
`permute(0, 2, 1, 3)` converts between head-major and token-major order.

### 5.4 Utility functions

- **`compute_kv_layout`** — extracts `(block_size, num_layers, hidden_dim_size, dtype_str, gpu_kv_format)` from live KV tensors.
- **`gather_paged_kv_to_cpu`** — gathers paged blocks into CPU chunk tensors.
- **`scatter_cpu_to_paged_kv`** — scatters CPU chunks back into device paged KV tensors. Respects `skip_first_n_tokens` for partial-prefix retrieval.

## Non-goals

- No change to existing CUDA IPC path semantics.
- No CPU-specific logic added to shared `gpu_connector/utils.py`.
- No wire-protocol incompatibility between CUDA and non-GPU context workers in
  the same cluster.
