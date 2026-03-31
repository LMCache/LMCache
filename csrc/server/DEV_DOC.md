# Pure C++ LMCache Server — Developer Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Build System](#build-system)
4. [File Reference](#file-reference)
5. [Wire Protocol](#wire-protocol)
6. [CUDA IPC & GPU Context](#cuda-ipc--gpu-context)
7. [L1 Slab Storage](#l1-slab-storage)
8. [Token Hashing & Sessions](#token-hashing--sessions)
9. [ZMQ Message Queue Server](#zmq-message-queue-server)
10. [Cache Engine Orchestration](#cache-engine-orchestration)
11. [Request Handlers](#request-handlers)
12. [Initialization & Startup Ordering](#initialization--startup-ordering)
13. [Debugging History & Lessons Learned](#debugging-history--lessons-learned)
14. [Configuration & Deployment](#configuration--deployment)
15. [Known Limitations & Future Work](#known-limitations--future-work)

---

## Overview

This is a complete pure C++ rewrite of the LMCache multiprocess server (originally `lmcache/v1/multiprocess/server.py`). It replaces the Python server process with a single C++ binary that is wire-compatible with existing Python vLLM clients. No client-side changes are needed.

The server is aligned with the `origin/dev` branch: it uses the block-level kernel (`multi_layer_block_kv_transfer` from `mp_mem_kernels.cu`), affinity/normal thread pools, `QUERY_PREFETCH_LOOKUP_HITS` request type, and batched retrieve (batch_size=4).

**Tested configuration:** DeepSeek-V3.1 on 8× NVIDIA H20 GPUs, TP=8, fp8 KV cache, MLA attention, block_size=64, 61 layers, hidden_dim=576.

**Codebase stats:** 5,427 lines across 20 files (10 .cpp, 10 .h), plus reused `mem_kernels.cu`, `mp_mem_kernels.cu`, `bitmap.cpp`, and `ttl_lock.cpp`.

---

## Architecture

```
vLLM Worker (Python)                     C++ LMCache Server
┌──────────────┐                      ┌──────────────────────────┐
│  TP Worker 0 │─── ZMQ DEALER ──────>│                          │
│  TP Worker 1 │─── ZMQ DEALER ──────>│  MessageQueueServer      │
│  ...         │                      │  (ZMQ ROUTER + eventfd)  │
│  TP Worker 7 │─── ZMQ DEALER ──────>│                          │
└──────────────┘                      ├──────────────────────────┤
                                      │  Main Loop (zmq_poll)    │
                                      │  ├─ SYNC handlers        │
                                      │  └─ Thread pools         │
                                      │     ├─ Affinity (GPU)    │
                                      │     │  STORE, RETRIEVE   │
                                      │     └─ Normal (CPU)      │
                                      │        LOOKUP, FREE, etc.│
                                      ├──────────────────────────┤
                                      │  CacheEngine             │
                                      │  ├─ GPUContext[0..7]     │
                                      │  ├─ L1Store (mmap slab)  │
                                      │  ├─ TokenHasher (BLAKE3) │
                                      │  └─ SessionManager       │
                                      └──────────────────────────┘
```

### Data Flow: Store Operation

```
vLLM Worker (TP rank X, device X):
  1. Encode STORE request: [key, instance_id, gpu_block_ids, event_ipc_handle]
  2. Send via ZMQ DEALER to server

C++ Server (affinity pool worker thread):
  1. Decode msgpack frames → StorePayload
  2. Compute token chunk hashes (BLAKE3 rolling prefix)
  3. Map to ObjectKeys (model_name + chunk_hash + kv_rank)
  4. L1Store::reserve_write() → get slab write slots
  5. Set CUDA device + stream guards for this GPU
  6. Stage all block_ids to pre-allocated GPU buffer
  7. Wait on vLLM's CUDA event (ensure GPU writes complete)
  8. For each chunk:
     a. Slice block_ids for this chunk's blocks
     b. multi_layer_block_kv_transfer(D2H): GPU KV cache → tmp_gpu_buffer
     c. cudaMemcpyAsync(D2H): tmp_gpu_buffer → L1 slab
  9. cudaStreamSynchronize
  10. L1Store::finish_write() → transition slots to Ready
  11. Create + record completion CUDA event
  12. Encode response: [event_handle, true]
```

### Data Flow: Retrieve Operation (Batched)

```
Same as Store but reversed, with batching:
  - Stage all block_ids to GPU once (stage_block_ids)
  - L1Store::reserve_read()
  - Process in batches of 4 chunks:
    a. get_tmp_gpu_buffer_batched(chunk_size, batch_size) → 4 buffer views
    b. cudaMemcpyAsync(H2D): L1 slab → tmp_buffers (one per chunk)
    c. multi_layer_block_kv_transfer(H2D): tmp_buffers → GPU KV cache
       with skip_prefix_n_blocks for APC skip
  - L1Store::finish_read()
  - Uses high-priority CUDA stream
```

### Data Flow: Lookup (Two-Phase Async)

```
Phase 1 — LOOKUP:
  1. Compute chunk hashes → ObjectKeys
  2. L1Store::prefix_lookup() → count leading hits
  3. Register PrefetchJob with hit count
  4. Return job_id

Phase 2 — QUERY_PREFETCH_STATUS or QUERY_PREFETCH_LOOKUP_HITS:
  - QUERY_PREFETCH_LOOKUP_HITS: returns hit count if lookup done, None if pending
  - QUERY_PREFETCH_STATUS: returns hit count and removes job entry (exactly-once)
```

---

## Build System

### CMakeLists.txt

The build uses CMake 3.20+ with C++17 and CUDA 17. Dependencies are resolved via:

| Dependency | Resolution |
|------------|-----------|
| **libtorch** | Auto-discovered via `python3 -c "import torch; print(torch.utils.cmake_prefix_path)"` |
| **CUDA Toolkit** | `find_package(CUDAToolkit)` — system install |
| **libzmq** | System `.so.5` + FetchContent for headers (avoids needing dev packages) |
| **cppzmq** | FetchContent header-only C++ binding |
| **msgpack-cxx** | FetchContent header-only, compiled with `MSGPACK_NO_BOOST` |
| **BLAKE3** | FetchContent, built as static library with SIMD assembly (SSE2/SSE4.1/AVX2/AVX512) |

### CUDA Architectures

```
CMAKE_CUDA_ARCHITECTURES: 80 (Ampere), 86 (Ampere), 89 (Ada), 90 (Hopper)
```

### Reused Sources

Four existing LMCache C++ files are compiled directly (not via pybind11):

```
csrc/mem_kernels.cu              — Token-level CUDA transfer kernels
csrc/mp_mem_kernels.cu           — Block-level CUDA transfer kernel
csrc/storage_manager/bitmap.cpp  — Bitmap bitwise operations
csrc/storage_manager/ttl_lock.cpp — TTL-based lock management
```

### Build Commands

```bash
cd LMCache-repo/csrc/server
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release   # or Debug
make -j$(nproc)
# Binary: ./lmcache-server
```

### Important Linker Flags

- `-rdynamic` — enables readable symbol names in `backtrace_symbols()` for crash handler
- RPATH includes `/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib` for `libnvJitLink.so`

---

## File Reference

### Headers (.h)

| File | Lines | Key Contents |
|------|-------|-------------|
| `types.h` | 278 | `RequestType` enum (1-21, no SYNC_LOOKUP), `DType`, `ObjectKey`+hash, `IPCCacheEngineKey`, `CudaIpcTensorDesc`, `MemorySlabRef`, `PrefetchHandle`, `compute_extra_count()` |
| `wire_protocol.h` | 173 | `Encoder`/`Decoder` classes, payload structs (`StorePayload`, `RetrievePayload`, `LookupPayload`, `QueryPrefetchLookupHitsPayload`, etc.) |
| `gpu_context.h` | 148 | `GPUContext` class — per-GPU IPC handles, streams, `PageBufferShapeDesc`, `stage_block_ids()`, `get_tmp_gpu_buffer_batched()` |
| `cache_engine.h` | 147 | `CacheEngine` class, `PrefetchJob`, `query_prefetch_lookup_hits()` |
| `l1_store.h` | 110 | `L1Store` abstract interface + `L1StoreConfig` |
| `l2_adapter.h` | 111 | `L2Adapter` abstract interface (placeholder) |
| `session_manager.h` | 95 | `Session` + `SessionManager` classes |
| `mq_server.h` | 91 | `IRequestHandler` + `MessageQueueServer` (`add_affinity_thread_pool`, `add_normal_thread_pool`) |
| `tensor_bridge.h` | 84 | `wrap_as_tensor()`, `open_ipc_tensor()`, `open_ipc_event()`, `create_ipc_event()` |
| `token_hasher.h` | 69 | `TokenHasher` class |

### Implementations (.cpp)

| File | Lines | Key Contents |
|------|-------|-------------|
| `wire_protocol.cpp` | 913 | msgpack encode/decode, Python pickle protocol 4 parser, `map_find()` helper |
| `mq_server.cpp` | 593 | `AffinityThreadPool`, `NormalThreadPool`, ZMQ ROUTER, eventfd, `zmq_poll` main loop |
| `l1_store.cpp` | 572 | `SlabL1Store` implementation: mmap, cudaHostRegister, state machine, LRU eviction |
| `cache_engine.cpp` | 534 | `store()`, `retrieve()` (batched, block-level kernel), `lookup()`, `query_prefetch_lookup_hits()`, `free_lookup_locks()` |
| `main.cpp` | 507 | 13 handler classes, CLI parsing, CUDA init, affinity/normal pool wiring |
| `gpu_context.cpp` | 491 | IPC handle opening, format discovery, stream creation, `stage_block_ids()`, `get_tmp_gpu_buffer_batched()` |
| `tensor_bridge.cpp` | 211 | `getIpcDevPtr()`, `from_blob()`, CUDA event IPC helpers |
| `session_manager.cpp` | 133 | Hash caching with TTL cleanup |
| `token_hasher.cpp` | 129 | BLAKE3 rolling prefix hash |
| `types.cpp` | 38 | `ipc_key_to_object_keys()` |

---

## Wire Protocol

### ZMQ Frame Layout

Each request is a multipart ZMQ message:

```
Frame 0: ZMQ routing identity (auto-managed by ROUTER)
Frame 1: RequestUID (int64, msgpack-encoded)
Frame 2: RequestType (int, msgpack-encoded)
Frame 3+: Payload frames (varies by request type)
```

Each response:

```
Frame 0: ZMQ routing identity (echoed back)
Frame 1: RequestUID (echoed back)
Frame 2: RequestType (echoed back)
Frame 3: Encoded response (msgpack)
```

### Payload Formats by Request Type

| RequestType | Payload Frames | Response |
|-------------|---------------|----------|
| `REGISTER_KV_CACHE` (1) | `[instance_id, kv_caches_list, model_name, world_size]` | None |
| `UNREGISTER_KV_CACHE` (2) | `[instance_id]` | None |
| `STORE` (3) | `[key, instance_id, gpu_block_ids, event_ipc_handle]` | `(bytes, bool)` |
| `RETRIEVE` (4) | `[key, instance_id, gpu_block_ids, event_ipc_handle, skip_tokens]` | `(bytes, bool)` |
| `LOOKUP` (5) | `[key, tp_size]` | `int` |
| `QUERY_PREFETCH_STATUS` (6) | `[prefetch_job_id]` | `int \| None` |
| `QUERY_PREFETCH_LOOKUP_HITS` (7) | `[prefetch_job_id]` | `int \| None` |
| `FREE_LOOKUP_LOCKS` (8) | `[key, tp_size]` | None |
| `END_SESSION` (9) | `[request_id]` | None |
| `CLEAR` (10) | (none) | None |
| `GET_CHUNK_SIZE` (11) | (none) | `int` |
| `PING` (12) | (none) | `bool` |
| `NOOP` (13) | (none) | `str` |

### IPCCacheEngineKey Encoding

The Python client encodes `IPCCacheEngineKey` as a **msgpack MAP** (dict with string keys), NOT an array. The C++ decoder handles both formats via `map_find()` helper:

```
{
  "model_name": str,
  "world_size": int,
  "worker_id": int | None,
  "token_ids": [int, ...],
  "start": int,
  "end": int,
  "request_id": str
}
```

### CudaIPCWrapper (Pickle Protocol 4)

CUDA IPC tensor descriptors are serialized by Python as pickle Ext type code 1. The C++ parser extracts:

1. **`ipc_handle_blob`**: First `BINBYTES`/`SHORT_BINBYTES` object ≥64 bytes (this is `handle[1]` from `_share_cuda_()`)
2. **`storage_size_bytes`**: `handle[2]` — extracted from int after the handle blob
3. **`device_uuid`**: `SHORT_BINUNICODE` string after the `'device_uuid'` field name in the pickle stream. The pickle contains a bare UUID; we prepend `"GPU-"` for CUDA device matching.
4. **`dtype`**: Extracted from `STACK_GLOBAL` opcode (e.g., `"uint8"` for fp8)
5. **`shape`/`stride`**: Extracted from `TUPLE2`/`TUPLE3` opcodes (small tuples) or `MARK`...`TUPLE` (large tuples)
6. **`storage_offset`**: Last `int` before final `TUPLE`

### DType Mapping

| Wire string | DType enum | at::ScalarType | Element size |
|-------------|-----------|---------------|-------------|
| `"uint8"` | `Int8` | `Byte` | 1 |
| `"float16"` | `Float16` | `Half` | 2 |
| `"bfloat16"` | `BFloat16` | `BFloat16` | 2 |
| `"float32"` | `Float32` | `Float` | 4 |
| `"float8_e4m3fn"` | `Float8E4M3FN` | `Float8_e4m3fn` | 1 |

Note: fp8 KV cache uses `torch.uint8` on the wire, which maps to `DType::Int8` → `at::ScalarType::Byte`.

---

## CUDA IPC & GPU Context

### GPUContext Class (`gpu_context.h`, `gpu_context.cpp`)

Each vLLM TP worker registers its KV cache via `REGISTER_KV_CACHE`. The server creates one `GPUContext` per instance_id (= per TP worker PID).

#### Construction Flow

```
1. Match device UUID from first tensor descriptor to local CUDA device index
   (iterate cudaGetDeviceProperties, format UUID as "GPU-%02x%02x...")
2. Open IPC tensor handles via c10::cuda::CUDACachingAllocator::getIpcDevPtr()
   → Creates at::Tensor objects that MUST stay alive (stored as class member)
3. Discover GPU KV format from tensor shapes:
   - 5D [2,NB,BS,NH,HS] → NL_X_TWO_NB_BS_NH_HS (flash attn)
   - 5D [NB,2,BS,NH,HS] → NL_X_NB_TWO_BS_NH_HS (flash infer)
   - 3D [NB,BS,HS] → NL_X_NB_BS_HS (MLA)
4. Extract shape parameters: num_blocks, block_size, hidden_dim_size, num_heads, head_size
5. Build PageBufferShapeDesc for the block-level kernel
6. Upload KV cache pointers to GPU as int64 tensor [num_layers]
7. Pre-compute slot mapping: [num_blocks, block_size] where slot[b][s] = b*bs+s
8. Allocate temporary GPU buffer for transfers (sized for 4× chunk_size for batching)
9. Allocate pre-allocated GPU buffer for block IDs (1M int64 elements)
10. Create normal + high-priority CUDA streams
```

#### Key Methods

- **`stage_block_ids(block_ids)`**: Copies int32 block IDs → int64 on pre-allocated GPU buffer. Returns a tensor view. Avoids per-call cudaMalloc.
- **`get_tmp_gpu_buffer_batched(num_tokens, batch_size)`**: Returns `batch_size` non-overlapping tensor views into the pre-allocated tmp buffer (sized for `kMaxBatchSize=4`).
- **`shape_desc()`**: Returns `PageBufferShapeDesc` struct for the block-level kernel.

#### Critical: IPC Tensor Lifetime

```cpp
// WRONG (original bug — dangling pointers after constructor):
std::vector<at::Tensor> kv_tensors;  // local variable, destroyed at end of ctor
for (int i = 0; i < num_layers_; ++i) {
    at::Tensor t = open_ipc_tensor(desc, device_idx);
    kv_cache_ptrs_[i] = t.data_ptr();  // saves raw pointer
    kv_tensors.push_back(std::move(t)); // tensor destroyed when kv_tensors goes out of scope!
}

// CORRECT (fixed):
kv_cache_ipc_tensors_.reserve(num_layers_);  // class member!
for (int i = 0; i < num_layers_; ++i) {
    at::Tensor t = open_ipc_tensor(desc, device_idx);
    kv_cache_ptrs_[i] = t.data_ptr();
    kv_cache_ipc_tensors_.push_back(std::move(t));  // kept alive for GPUContext lifetime
}
```

**Why this matters:** `getIpcDevPtr()` returns `shared_ptr<void>`. PyTorch's IPC cache stores only `weak_ptr`. When the tensor (which holds the `shared_ptr` in its storage) is destroyed, the IPC handle is closed and the GPU memory is unmapped. Any subsequent kernel access to the raw pointer is an illegal memory access.

### Tensor Bridge (`tensor_bridge.h`, `tensor_bridge.cpp`)

#### `open_ipc_tensor()`

```cpp
// 1. Open IPC handle via PyTorch's caching allocator
std::string handle_str(blob.data(), blob.size());
auto dev_ptr = c10::cuda::CUDACachingAllocator::getIpcDevPtr(std::move(handle_str));

// 2. Apply storage_offset
void* offset_ptr = static_cast<char*>(dev_ptr.get()) + storage_offset * elem_size;

// 3. Create tensor with custom deleter holding shared_ptr
auto ref = std::make_shared<std::shared_ptr<void>>(std::move(dev_ptr));
auto deleter = [ref](void*) { /* ref drops when tensor dies */ };
return at::from_blob(offset_ptr, shape, stride, deleter, options);
```

**Why `getIpcDevPtr()` instead of `cudaIpcOpenMemHandle()`:** PyTorch 2.10+ with CUDA 12.x uses an expanded 66-byte handle format (not the raw 64-byte `cudaIpcMemHandle_t`). `getIpcDevPtr()` handles both formats.

### CUDA Device & Stream Scoping

Every GPU operation in `store()` and `retrieve()` uses ATen guards:

```cpp
// Set the correct CUDA device
c10::cuda::CUDAGuard device_guard(gpu_ctx->device_index());

// Direct all ATen + CUDA operations to this GPU's stream
at::cuda::CUDAStream torch_stream =
    at::cuda::getStreamFromExternal(gpu_ctx->stream(), gpu_ctx->device_index());
at::cuda::CUDAStreamGuard stream_guard(torch_stream);
```

This matches the Python server's `with torch.cuda.device(dev), torch.cuda.stream(stream):` pattern.

---

## L1 Slab Storage

### Architecture

```
┌─────────────────────────────────────────┐
│           L1 Slab (mmap'd)              │
│  ┌──────┬──────┬──────┬──────┬────────┐ │
│  │Slot 0│Slot 1│Slot 2│ ...  │Slot N-1│ │
│  │ FREE │READY │WRITE │      │READING │ │
│  └──────┴──────┴──────┴──────┴────────┘ │
│  cudaHostRegister'd → zero-copy DMA     │
├─────────────────────────────────────────┤
│  Metadata (hash map):                   │
│    ObjectKey → { slot_index, state,     │
│                  lock_count, lru_pos }  │
│  TTLLock for read lock management       │
│  LRU eviction when capacity exhausted   │
└─────────────────────────────────────────┘
```

### State Machine

```
 ┌──── Free ◄─── evict()/delete_key()
 │       │
 │  reserve_write()
 │       │
 │       ▼
 │    Writing
 │       │
 │  finish_write()
 │       │
 │       ▼
 └──── Ready ◄─── finish_read()
          │
     reserve_read()
          │
          ▼
       Reading ───► Ready (when lock_count → 0)
```

### MLA Multi-Reader Locking

For MLA models, all TP workers share the same KV cache object (since there's only one KV head). The `extra_count` parameter in `reserve_read()`/`finish_read()` handles this:

```cpp
int extra_count = compute_extra_count(tp_size, world_size);
// Non-MLA: extra_count = 0 (each worker has distinct KV shard)
// MLA: extra_count = tp_size - 1 (all workers share same object)
```

---

## Token Hashing & Sessions

### TokenHasher (`token_hasher.h`, `token_hasher.cpp`)

Computes rolling prefix hashes using BLAKE3:

```
Chunk 0 hash = BLAKE3(none_hash || tokens[0:chunk_size])
Chunk 1 hash = BLAKE3(chunk_0_hash || tokens[chunk_size:2*chunk_size])
Chunk N hash = BLAKE3(chunk_{N-1}_hash || tokens[N*chunk_size:(N+1)*chunk_size])
```

Where `none_hash` is `BLAKE3(b"None")` — the initial prefix context.

Only complete chunks are hashed (trailing tokens < chunk_size are ignored).

### SessionManager (`session_manager.h`, `session_manager.cpp`)

- `Session` stores per-request state: token_ids, cached hashes
- `get_hashes(start, end)` returns chunk hashes for the range, using cached values when possible
- Thread-safe (mutex per session)
- `cleanup_expired()` removes sessions older than TTL

---

## ZMQ Message Queue Server

### Architecture

```
                          ┌──────────────┐
  ZMQ ROUTER socket ─────┤  Main Loop   │
  (tcp://host:port)       │  zmq_poll()  │
                          │  [socket_fd, │
                          │   eventfd]   │
                          └──────┬───────┘
                                 │
             ┌───────────────────┼───────────────────┐
             │                   │                   │
     ┌───────▼───────┐  ┌───────▼───────┐  ┌───────▼───────┐
     │ SYNC handler  │  │ Affinity pool │  │ Normal pool   │
     │ (main thread) │  │ (GPU-bound)   │  │ (CPU-bound)   │
     │               │  │ STORE,RETRIEVE│  │ LOOKUP, FREE  │
     │ GET_CHUNK_SIZE│  │ Identity-     │  │ QUERY, END,   │
     │ REGISTER, etc.│  │ pinned routing│  │ CLEAR, PING   │
     └───────────────┘  └───────┬───────┘  └───────┬───────┘
                                │                   │
                         eventfd write ◄────────────┘
                         (wakes main loop to send ZMQ response)
```

### Thread Pool Types

#### AffinityThreadPool
- Routes requests to workers by `identity_hash % num_workers`
- Same ZMQ client identity → same worker thread (deterministic)
- Eliminates need for per-instance GPU transfer locks
- Each worker has its own task queue (no contention between workers)
- Used for: `STORE`, `RETRIEVE`

#### NormalThreadPool
- Standard shared-queue round-robin dispatch
- Used for: `LOOKUP`, `QUERY_PREFETCH_STATUS`, `QUERY_PREFETCH_LOOKUP_HITS`, `FREE_LOOKUP_LOCKS`, `END_SESSION`, `CLEAR`, `PING`

### Key Design Points

1. **SYNC handlers** run in the main ZMQ poll loop (fast, non-blocking)
2. **BLOCKING handlers** are dispatched to their assigned pool
3. **All blocking handlers must have a pool assigned** — validated at `start()` time
4. **eventfd** bridges thread pool → main loop: worker writes 1 to eventfd, main loop wakes up and sends the ZMQ response
5. Responses are always sent from the main loop (ZMQ sockets are not thread-safe)

### Handler Classification

| Handler Type | Request Types |
|-------------|--------------|
| **SYNC** | `REGISTER_KV_CACHE`, `UNREGISTER_KV_CACHE`, `QUERY_PREFETCH_STATUS`, `GET_CHUNK_SIZE`, `NOOP` |
| **BLOCKING** (affinity) | `STORE`, `RETRIEVE` |
| **BLOCKING** (normal) | `LOOKUP`, `QUERY_PREFETCH_LOOKUP_HITS`, `FREE_LOOKUP_LOCKS`, `END_SESSION`, `CLEAR`, `PING` |

---

## Cache Engine Orchestration

### Store Flow (Block-Level Kernel)

```cpp
std::pair<std::vector<uint8_t>, bool> CacheEngine::store(
    const IPCCacheEngineKey& key,
    int instance_id,
    const std::vector<int32_t>& gpu_block_ids,
    const std::vector<uint8_t>& event_ipc_handle)
{
  // 1. Session management + hash computation
  auto session = session_manager_.get_or_create(key.request_id);
  session->set_tokens(key.token_ids);
  auto chunk_hashes = session->get_hashes(key.start, key.end);
  auto obj_keys = ipc_key_to_object_keys(...);

  // 2. GPU context lookup
  auto& gpu_ctx = gpu_contexts_[instance_id];
  int blocks_per_chunk = chunk_size_ / gpu_ctx->block_size();

  // 3. CUDA device + stream guards
  c10::cuda::CUDAGuard device_guard(gpu_ctx->device_index());
  at::cuda::CUDAStreamGuard stream_guard(...);

  // 4. Stage all block_ids to GPU once (pre-allocated buffer)
  at::Tensor all_block_ids_gpu = gpu_ctx->stage_block_ids(gpu_block_ids);

  // 5. Wait for vLLM to finish writing KV cache
  cudaStreamWaitEvent(gpu_ctx->stream(), vllm_event, 0);

  // 6. Reserve L1 write slots
  auto reserved = l1_store_->reserve_write(obj_keys, layout, "new");

  // 7. Transfer each chunk: GPU → tmp_buffer → L1 (not batched — skip gaps)
  {
    std::lock_guard<std::mutex> lk(gpu_ctx->transfer_lock());
    for (size_t idx = 0; idx < obj_keys.size(); ++idx) {
      if (reserved.find(obj_keys[idx]) == reserved.end()) continue;

      at::Tensor chunk_block_ids_gpu = all_block_ids_gpu.slice(...);
      at::Tensor tmp_buf = gpu_ctx->get_tmp_gpu_buffer(chunk_size_);

      // Block-level kernel: GPU KV cache → tmp_buffer
      multi_layer_block_kv_transfer(
          gpu_ctx->kv_pointers(),
          {tmp_buf.data_ptr()}, chunk_block_ids_gpu,
          device, D2H, gpu_ctx->shape_desc(), chunk_size_,
          gpu_kv_format, 0);

      // tmp_buffer → L1 slab (async memcpy)
      cudaMemcpyAsync(slab_ref.data, tmp_buf.data_ptr(), ...);
    }
  }

  // 8. Sync and finish
  cudaStreamSynchronize(gpu_ctx->stream());
  l1_store_->finish_write(written_keys);
  return {done_event_bytes, true};
}
```

### Retrieve Flow (Batched)

Same structure as Store but:
- Uses **high-priority stream** (for latency-sensitive path)
- **Batched**: processes up to 4 chunks per kernel launch
- Uses `get_tmp_gpu_buffer_batched()` for 4 separate buffer views
- Applies `skip_prefix_n_blocks` (skips blocks already in GPU cache via APC)
- Transfer direction is H2D (host → device)

```cpp
// Process in batches of 4 chunks
for (batch_start = 0; batch_start < obj_keys.size(); batch_start += 4) {
  auto tmp_bufs = gpu_ctx->get_tmp_gpu_buffer_batched(chunk_size_, actual_batch);

  // H2D memcpy for each chunk in batch
  for (bi = 0; bi < actual_batch; ++bi)
    cudaMemcpyAsync(tmp_bufs[bi].data_ptr(), slab_ref.data, ...);

  // Single kernel launch for the whole batch
  multi_layer_block_kv_transfer(
      gpu_ctx->kv_pointers(),
      lmcache_ptrs,  // vector of 1-4 tmp buffer pointers
      batch_block_ids_gpu,
      device, H2D, shape_desc, chunk_size_, format,
      skip_blocks_in_batch);
}
```

### Lookup Flow

```cpp
int CacheEngine::lookup(const IPCCacheEngineKey& key, int tp_size) {
  // 1. Compute chunk hashes
  auto chunk_hashes = token_hasher_.compute_chunk_hashes(key.token_ids);
  auto obj_keys = ipc_key_to_object_keys(...);

  // 2. L1 prefix lookup (find longest prefix of existing keys)
  int64_t l1_hits = l1_store_->prefix_lookup(obj_keys);

  // 3. Register prefetch job
  int job_id = next_prefetch_job_id_++;
  prefetch_jobs_[job_id] = PrefetchJob{handle, world_size, request_id};
  return job_id;
}
```

### Free Lookup Locks

Simplified — just releases L1 read locks directly:

```cpp
void CacheEngine::free_lookup_locks(const IPCCacheEngineKey& key, int tp_size) {
  auto chunk_hashes = token_hasher_.compute_chunk_hashes(key.token_ids, start, end);
  auto obj_keys = ipc_key_to_object_keys(...);
  int extra_count = compute_extra_count(tp_size, key.world_size);
  l1_store_->finish_read(obj_keys, extra_count);
}
```

---

## Request Handlers

All 13 handlers are defined in `main.cpp` as concrete `IRequestHandler` implementations:

| Class | Request Type | Handler Type | Pool | Action |
|-------|-------------|-------------|------|--------|
| `RegisterHandler` | `REGISTER_KV_CACHE` | SYNC | — | `engine.register_kv_cache()` |
| `UnregisterHandler` | `UNREGISTER_KV_CACHE` | SYNC | — | `engine.unregister_kv_cache()` |
| `StoreHandler` | `STORE` | BLOCKING | Affinity | `engine.store()` |
| `RetrieveHandler` | `RETRIEVE` | BLOCKING | Affinity | `engine.retrieve()` |
| `LookupHandler` | `LOOKUP` | BLOCKING | Normal | `engine.lookup()` |
| `QueryPrefetchStatusHandler` | `QUERY_PREFETCH_STATUS` | SYNC | — | `engine.query_prefetch_status()` |
| `QueryPrefetchLookupHitsHandler` | `QUERY_PREFETCH_LOOKUP_HITS` | BLOCKING | Normal | `engine.query_prefetch_lookup_hits()` |
| `FreeLookupLocksHandler` | `FREE_LOOKUP_LOCKS` | BLOCKING | Normal | `engine.free_lookup_locks()` |
| `EndSessionHandler` | `END_SESSION` | BLOCKING | Normal | `engine.end_session()` |
| `ClearHandler` | `CLEAR` | BLOCKING | Normal | `engine.clear()` |
| `GetChunkSizeHandler` | `GET_CHUNK_SIZE` | SYNC | — | `engine.get_chunk_size()` |
| `PingHandler` | `PING` | BLOCKING | Normal | `engine.ping()` |
| `NoopHandler` | `NOOP` | SYNC | — | Returns "OK" |

---

## Initialization & Startup Ordering

**Critical ordering in `main()`:**

```cpp
// 1. Signal handlers (SIGINT, SIGTERM → clean shutdown; SIGSEGV, SIGABRT → backtrace)
std::signal(SIGSEGV, crash_handler);

// 2. Parse CLI arguments
auto cfg = parse_args(argc, argv);

// 3. Initialize CUDA runtime BEFORE CacheEngine
//    This MUST happen before cudaHostRegister (in L1Store) or it silently fails!
for (int i = 0; i < device_count; ++i) {
    cudaSetDevice(i);
    cudaFree(nullptr);  // Force lazy CUDA init
}
cudaSetDevice(0);
c10::cuda::CUDACachingAllocator::init(device_count);  // Required for getIpcDevPtr()

// 4. Create CacheEngine (L1 slab mmap + cudaHostRegister happens here)
CacheEngine engine(cfg.chunk_size, l1_config, nullptr);

// 5. Create ZMQ server + register all 13 handlers
MessageQueueServer server(bind_url, 0);
server.add_handler(RequestType::STORE, std::make_unique<StoreHandler>(engine));
// ... all 13 handlers ...

// 6. Assign thread pools
server.add_affinity_thread_pool({STORE, RETRIEVE}, cfg.max_gpu_workers);
server.add_normal_thread_pool({LOOKUP, QUERY_PREFETCH_LOOKUP_HITS, ...}, cfg.max_cpu_workers);

// 7. Start server (validates all blocking handlers have pools, spawns main loop)
server.start();

// 8. Sleep loop until shutdown signal
while (!g_shutdown) sleep(1);
```

**Why CUDA init must come first:**
- `cudaHostRegister()` requires a CUDA context to exist. Without `cudaFree(nullptr)` first, the register call silently fails.
- Later `cudaMemcpyAsync` from the slab hits illegal memory access because the slab wasn't actually pinned.
- `c10::cuda::CUDACachingAllocator::init()` is required for `getIpcDevPtr()` to work.

---

## Debugging History & Lessons Learned

### Bug 1: PyTorch 2.10+ IPC Handle Format

**Symptom:** `cudaIpcOpenMemHandle` returned `CUDA_ERROR_INVALID_VALUE`
**Root Cause:** PyTorch 2.10+ with CUDA 12.x uses a 66-byte expanded handle format, not the raw 64-byte `cudaIpcMemHandle_t`.
**Fix:** Use `c10::cuda::CUDACachingAllocator::getIpcDevPtr()` which handles both formats.

### Bug 2: CUDA Caching Allocator Not Initialized

**Symptom:** Segfault inside `getIpcDevPtr()`
**Root Cause:** `c10::cuda::CUDACachingAllocator` was not initialized before the first `getIpcDevPtr()` call.
**Fix:** Call `c10::cuda::CUDACachingAllocator::init(device_count)` in `main()` before creating `CacheEngine`.

### Bug 3: Dangling IPC Pointers (The Final Critical Bug)

**Symptom:** First store on one GPU succeeded, then `multi_layer_kv_transfer` crashed with illegal memory access on subsequent stores. CUDA error became "sticky" and broke all subsequent operations (including `cudaIpcOpenEventHandle`).
**Root Cause:** IPC tensors were stored in a **local variable** in the `GPUContext` constructor. When the constructor returned, the tensors were destroyed, releasing the `shared_ptr` from `getIpcDevPtr()`, which closed the IPC handle and unmapped the GPU memory. `kv_cache_ptrs_` then contained dangling pointers.
**Fix:** Store IPC tensors as a **class member** `kv_cache_ipc_tensors_` so they live as long as the `GPUContext`.

### Bug 4: IPCCacheEngineKey MAP vs ARRAY

**Symptom:** Key decode failed — fields were all zeros/empty.
**Root Cause:** Python `msgspec.msgpack.encode()` for `@msgspec.Struct` classes produces msgpack MAP (dict with string keys), not ARRAY. The decoder assumed ARRAY format.
**Fix:** Support both formats with a `map_find()` helper function.

### Bug 5: Pickle Parser Issues

**Symptom:** Empty `device_uuid`, wrong IPC handle extraction, shape parsing failures.
**Root Cause (multiple):**
1. `device_uuid`: Pickle contains bare UUID, not "GPU-" prefixed → prepend "GPU-" after extraction
2. IPC handle: Originally took last 64-byte blob (handle[6] = event handle) → take FIRST ≥64-byte `BINBYTES` (handle[1] = storage handle)
3. Shape: Pickle uses `TUPLE2`/`TUPLE3` opcodes for small tuples, not `MARK`...`TUPLE` → handle both formats

### Bug 6: uint8 DType Not Recognized

**Symptom:** fp8 KV cache dtype failed to parse.
**Root Cause:** fp8 uses `torch.uint8` on the wire, which wasn't in the dtype mapping.
**Fix:** Add `"uint8"` → `DType::Int8` → `at::ScalarType::Byte` mapping.

### Bug 7: ATen Header Namespace Conflicts

**Symptom:** Compilation error — `c10::ScalarType` vs `at::ScalarType` mismatch under NVCC.
**Root Cause:** Per-operator ATen includes (`ATen/ops/*.h`) caused namespace aliasing to break in PyTorch 2.10+.
**Fix:** Use `#include <torch/all.h>` everywhere (sets up proper namespace aliasing).

### Bug 8: CUDA Init Before L1 Slab

**Symptom:** `cudaMemcpyAsync(D2H)` to the L1 slab caused illegal memory access.
**Root Cause:** `cudaHostRegister()` was called before CUDA was initialized. It returned `cudaSuccess` but silently did nothing.
**Fix:** Move CUDA initialization (`cudaFree(nullptr)` per device) before `CacheEngine` construction.

---

## Configuration & Deployment

### Launch Scripts

**C++ LMCache server** (`launch_lmc_cpp_server.sh`):
```bash
LMCache-repo/csrc/server/build/lmcache-server \
    --host 0.0.0.0 \
    --port 15555 \
    --chunk-size 256 \
    --l1-capacity-gib 64 \
    --max-gpu-workers 16 \
    --max-cpu-workers 8
```

**vLLM with C++ LMCache** (`launch_vllm_server.sh`):
```bash
vllm serve "$MODEL" \
    -tp 8 --kv-cache-dtype fp8 --block-size 64 \
    --max-model-len 16384 --enforce-eager \
    --kv-transfer-config '{"kv_connector":"LMCacheMPConnectorDynamic",
        "kv_connector_extra_config":{"lmcache.mp.port":15555}}'
```

### Log Locations

When using nohup:
- LMCache server: `/tmp/lmc_out.log` (or wherever redirected)
- vLLM server: `/disc/data1/riggins/hover/vllm_server.log`

### Process Management

```bash
# Kill all related processes
kill -9 $(pgrep -f lmcache-server) $(pgrep -f "vllm serve") 2>/dev/null

# Verify GPU memory is freed
nvidia-smi
```

---

## Known Limitations & Future Work

### Not Implemented

1. **L2 Adapter** — `l2_adapter.h` is an abstract interface only. No Redis, filesystem, or NitroFS backend.
   - The `run_prefetch_load()` method is a stub returning 0
   - L2 lookup/lock/load paths are not implemented

2. **Telemetry** — No equivalent of the Python server's telemetry system (START/END events, span correlation, JSONL output).

3. **Blend Operations** — `CB_*` request types (14-21) are defined in `types.h` but no handlers are registered.

4. **Hot Reload** — Server must be restarted for any configuration changes.

5. **Graceful Shutdown** — CUDA cleanup on shutdown is best-effort. IPC handles are leaked (cleaned up by OS on exit).

### Potential Improvements

1. **Concurrent stores to different GPUs** — With affinity pools, same-identity requests serialize on one worker. Different identities (different TP workers) run on different workers concurrently.

2. **L1 eviction policy** — Current LRU eviction is simple. Could add frequency-based or cost-aware eviction.

3. **Error recovery** — A CUDA error on one device currently corrupts all CUDA state. Could add per-device error isolation.

4. **Metrics endpoint** — No Prometheus/HTTP metrics. Could add a lightweight HTTP server or ZMQ stats channel.

---

## Testing

### Quick Smoke Test

```bash
# Terminal 1: Start C++ LMCache server
bash launch_lmc_cpp_server.sh

# Terminal 2: Start vLLM
bash launch_vllm_server.sh 2>&1 | tee vllm_server.log

# Terminal 3: Run benchmark (wait for vLLM to be ready)
bash easy_bench.sh

# Check logs
cat /tmp/lmc_out.log   # Should show REGISTER, LOOKUP, STORE operations
```

### Expected Log Output (Success)

```
lmcache-server (pure C++)
  bind: tcp://0.0.0.0:15555
  chunk_size: 256
  L1 capacity: 64 GiB
  max_gpu_workers: 16
  max_cpu_workers: 8
CUDA initialized: 8 device(s)
Starting server...
LMCache C++ server is running on tcp://0.0.0.0:15555
CacheEngine: registered KV cache for instance XXXXX (61 layers, model=..., ws=1)
  [repeated 8 times, one per TP worker]
CacheEngine: stored 23 chunks (5888 tokens)
  [repeated 8 times, one per TP worker]
```
