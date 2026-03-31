# LMCache Pure C++ Server — Implementation Complete

## Status: WORKING (aligned with origin/dev)

The pure C++ LMCache server is **implemented and operational**. It has been tested with DeepSeek-V3.1 on 8x NVIDIA H20 GPUs with TP=8, fp8 KV cache, MLA attention, and successfully handles store/retrieve/lookup operations from vLLM clients. Tested with 20 concurrent requests sent in two rounds.

The server is aligned with the `origin/dev` branch: it uses the block-level kernel (`multi_layer_block_kv_transfer`), affinity/normal thread pools, and the `QUERY_PREFETCH_LOOKUP_HITS` request type (no `SYNC_LOOKUP`).

---

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│           lmcache-server binary             │
│        (LMCache-repo/csrc/server/)          │
├─────────────────────────────────────────────┤
│  main.cpp — CLI, signal handlers, wiring    │
│  ├─ 13 IRequestHandler classes              │
│  ├─ CacheEngine orchestrator                │
│  └─ MessageQueueServer (ZMQ ROUTER)         │
├─────────────────────────────────────────────┤
│  CacheEngine (cache_engine.cpp)             │
│  ├─ GPUContext — per-GPU IPC + streams      │
│  ├─ L1Store — mmap slab + cudaHostRegister  │
│  ├─ TokenHasher — BLAKE3 rolling hashes     │
│  └─ SessionManager — per-request state      │
├─────────────────────────────────────────────┤
│  Wire Protocol (wire_protocol.cpp)          │
│  ├─ msgpack encode/decode                   │
│  └─ Python pickle parser (CudaIPCWrapper)   │
├─────────────────────────────────────────────┤
│  Reused C++ Sources                         │
│  ├─ mem_kernels.cu — token-level kernels    │
│  ├─ mp_mem_kernels.cu — block-level kernel  │
│  ├─ bitmap.cpp — bitwise operations         │
│  └─ ttl_lock.cpp — TTL-based locking        │
├─────────────────────────────────────────────┤
│  Dependencies                               │
│  ├─ libtorch (ATen tensors, CUDA guards)    │
│  ├─ libzmq (ZMQ ROUTER socket)              │
│  ├─ msgpack-cxx (wire serialization)        │
│  ├─ BLAKE3 (token hashing)                  │
│  └─ CUDA runtime + driver APIs              │
└─────────────────────────────────────────────┘
```

---

## File Inventory (5,427 lines total)

| File | Lines | Purpose |
|------|-------|---------|
| `wire_protocol.cpp` | 913 | msgpack encode/decode + Python pickle parser |
| `mq_server.cpp` | 593 | ZMQ ROUTER + eventfd + affinity/normal thread pools |
| `l1_store.cpp` | 572 | Mmap slab storage with state machine |
| `cache_engine.cpp` | 534 | Store/retrieve/lookup orchestration (block-level kernel) |
| `main.cpp` | 507 | Entry point, handlers, CLI |
| `gpu_context.cpp` | 491 | Per-GPU IPC, streams, block_ids staging, batched buffers |
| `types.h` | 278 | All shared type definitions |
| `tensor_bridge.cpp` | 211 | ATen ↔ raw CUDA bridge |
| `wire_protocol.h` | 173 | Encoder/Decoder declarations |
| `gpu_context.h` | 148 | GPUContext class (with PageBufferShapeDesc, stage_block_ids) |
| `cache_engine.h` | 147 | CacheEngine class |
| `session_manager.cpp` | 133 | Per-request hash cache |
| `token_hasher.cpp` | 129 | BLAKE3 rolling prefix hash |
| `l1_store.h` | 110 | L1Store abstract interface |
| `l2_adapter.h` | 111 | L2 interface (placeholder) |
| `session_manager.h` | 95 | SessionManager class |
| `mq_server.h` | 91 | MessageQueueServer class |
| `tensor_bridge.h` | 84 | Tensor bridge declarations |
| `token_hasher.h` | 69 | TokenHasher class |
| `types.cpp` | 38 | ipc_key_to_object_keys() |

---

## Key Technical Details

### 1. Wire Protocol Compatibility

The C++ server is **wire-compatible** with existing Python vLLM clients. No client changes needed.

- ZMQ ROUTER socket (same as Python `MessageQueueServer`)
- msgpack serialization matching `msgspec.msgpack` format
- CudaIPCWrapper parsed from Python pickle protocol 4 (Ext type code 1)
- IPCCacheEngineKey decoded as msgpack MAP (dict with string keys)

### 2. CUDA IPC Handle Management

- Uses `c10::cuda::CUDACachingAllocator::getIpcDevPtr()` (not raw `cudaIpcOpenMemHandle`)
- Supports PyTorch 2.10+ expanded 66-byte handle format
- IPC tensors stored as class members to keep shared_ptr alive (critical for handle lifetime)
- `c10::cuda::CUDAGuard` + `at::cuda::CUDAStreamGuard` for device/stream scoping

### 3. GPU KV Transfer (Block-Level Kernel)

Uses `multi_layer_block_kv_transfer` from `csrc/mp_mem_kernels.cu`:
- **Block-level**: operates on block IDs directly, not slot mappings
- **Batched retrieve**: processes up to 4 chunks per kernel launch
- **PageBufferShapeDesc**: shape descriptor struct passed to the kernel
- **stage_block_ids()**: pre-allocated GPU buffer (1M int64 elements) for block IDs
- Store: GPU KV cache → tmp_buffer → L1 slab (D2H)
- Retrieve: L1 slab → tmp_buffer → GPU KV cache (H2D)
- Supports MLA format (format=3, `NL_X_NB_BS_HS`) and all 5 other vLLM/SGLang formats

### 4. Thread Pool Architecture

Two pool types (matching `origin/dev` Python server):
- **AffinityThreadPool** (STORE, RETRIEVE): routes requests from the same ZMQ identity to the same worker thread (hash identity → worker index), eliminating per-instance GPU transfer locks
- **NormalThreadPool** (LOOKUP, QUERY_PREFETCH_LOOKUP_HITS, FREE_LOOKUP_LOCKS, etc.): standard round-robin worker pool for CPU-bound handlers
- All blocking handlers **must** have a pool assigned before `start()` (validated at startup)

### 5. L1 Slab Storage

- mmap'd memory region + `cudaHostRegister` for zero-copy DMA
- State machine: Free → Writing → Ready → Reading → Ready
- LRU eviction when capacity exhausted
- TTL-based read lock management via existing C++ `TTLLock`

### 6. Token Hashing

- BLAKE3 rolling prefix hash (same algorithm as Python server)
- Session-based hash caching for incremental updates

---

## Build & Run

```bash
# Build
cd LMCache-repo/csrc/server
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run
./lmcache-server \
    --host 0.0.0.0 \
    --port 15555 \
    --chunk-size 256 \
    --l1-capacity-gib 64 \
    --max-gpu-workers 16 \
    --max-cpu-workers 8
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8001` | Bind port |
| `--chunk-size` | `256` | Tokens per chunk |
| `--l1-capacity-gib` | `8` | L1 slab capacity in GiB |
| `--max-workers` | `8` | Thread pool workers (sets both GPU and CPU) |
| `--max-gpu-workers` | `8` | Affinity pool workers for STORE/RETRIEVE |
| `--max-cpu-workers` | `8` | Normal pool workers for LOOKUP etc. |
| `--hugepages` | off | Use huge pages for L1 slab |
| `--no-cuda-host-register` | off | Disable cudaHostRegister |

---

## Dependencies

| Dependency | Version | How Resolved |
|------------|---------|-------------|
| libtorch | from pip PyTorch | `torch.utils.cmake_prefix_path` |
| CUDA Toolkit | 12.x | system install |
| libzmq | 4.3.5 | system .so + FetchContent headers |
| cppzmq | 4.10.0 | FetchContent (header-only) |
| msgpack-cxx | 6.1.1 | FetchContent (header-only, no Boost) |
| BLAKE3 | 1.8.2 | FetchContent (static lib with SIMD) |
| c10_cuda | from libtorch | for `getIpcDevPtr()` |

---

## Known Limitations

1. **L2 adapter not implemented** — only L1 (host memory) caching; L2 (Redis/NitroFS) is a placeholder
2. **No telemetry** — the Python server's telemetry system is not ported
3. **Blend operations not implemented** — CB_* request types are defined but not handled
4. **No hot-reload** — server must be restarted for config changes

---

## Bugs Fixed During Development

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| `cudaIpcOpenMemHandle` invalid argument | PyTorch 2.10+ uses 66-byte expanded handle | Use `getIpcDevPtr()` instead |
| Segfault in `getIpcDevPtr` | CUDA caching allocator not initialized | Call `CUDACachingAllocator::init()` before engine |
| CUDA illegal memory access in kernel | IPC tensor pointers dangled after constructor | Store IPC tensors as class members |
| IPCCacheEngineKey decode failure | msgspec encodes as MAP, not array | Handle both MAP and ARRAY formats |
| `uint8` dtype unknown | fp8 KV cache uses `torch.uint8` | Map `"uint8"` → `DType::Int8` → `at::ScalarType::Byte` |
| Empty `device_uuid` | Pickle doesn't contain "GPU-" prefix | Parse bare UUID, prepend "GPU-" |
| `c10::ScalarType` namespace mismatch | Per-operator ATen includes | Use `#include <torch/all.h>` |
