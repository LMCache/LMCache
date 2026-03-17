# LMCache Architecture Deep-Dive

## 1. What is LMCache?

LMCache is an **LLM serving engine extension** that dramatically reduces **Time-To-First-Token (TTFT)** and increases **throughput** by caching and reusing **KV (Key-Value) caches** across requests and across serving instances. KV caches are the intermediate attention states produced during the "prefill" phase of LLM inference -- the most expensive computation step. By storing these caches across a multi-tier storage hierarchy (GPU, CPU, Disk, Remote/S3, P2P), LMCache avoids redundant recomputation of shared text prefixes.

**Key insight:** Many LLM workloads share common text prefixes (system prompts, RAG documents, multi-round conversation history). LMCache detects this overlap, stores the KV caches once, and reuses them across any request on any serving instance -- achieving **3-10x latency/GPU savings**.

---

## 2. High-Level Architecture Diagram

```
+=====================================================================+
|                        LLM Serving Engine                           |
|                    (vLLM / SGLang / others)                         |
+====+======================+=========================+===============+
     |                      |                         |
     | register_kv_caches   | start_load_kv /         | save_kv_layer /
     | (once at startup)    | wait_for_layer_load     | wait_for_save
     |                      | (retrieve path)         | (store path)
     v                      v                         v
+====+======================+=========================+===============+
|              Integration Layer (Connector/Adapter)                   |
|  +---------------------------+   +-------------------------------+  |
|  | LMCacheConnectorV1Dynamic |   | LMCacheConnectorV1Impl       |  |
|  | (vLLM KVConnectorBase_V1) |-->| (vllm_v1_adapter.py)         |  |
|  +---------------------------+   | - Creates LMCacheEngine       |  |
|                                  | - Maps vLLM structures to     |  |
|  +---------------------------+   |   LMCache API calls           |  |
|  | SGLang Adapter            |   +-------------------------------+  |
|  | (sglang_adapter.py)       |                                      |
|  +---------------------------+                                      |
+====================================================================+
     |                      |                         |
     v                      v                         v
+====+======================+=========================+===============+
|                   LMCacheEngine (cache_engine.py)                   |
|                   [Central Orchestrator]                             |
|                                                                     |
|   Public API:                                                       |
|   - lookup()           Check existence of cached KV for tokens      |
|   - store()            GPU -> CPU/Storage (all layers at once)      |
|   - store_layer()      GPU -> CPU/Storage (layer-by-layer pipeline) |
|   - retrieve()         Storage/CPU -> GPU (all layers at once)      |
|   - retrieve_layer()   Storage/CPU -> GPU (layer-by-layer pipeline) |
|   - async_lookup_and_prefetch()   Async lookup + background fetch   |
|   - move()             Cross-node KV transfer via P2P               |
|   - compress/decompress()   In-place KV cache compression           |
|   - clear()            Remove cached entries                        |
|   - freeze/unfreeze()  Lock hot cache from modifications            |
|                                                                     |
|   +-------------------+   +------------------+   +---------------+  |
|   | Token Database    |   | Event Manager    |   | Stats Monitor |  |
|   | (Chunked/Segment) |   | (async tracking) |   | (observability|  |
|   +-------------------+   +------------------+   +---------------+  |
+===+========================+========================+===============+
    |                        |                        |
    v                        v                        v
+---+------+    +------------+----------+    +--------+--------+
| GPU      |    | Storage Manager       |    | LMCache Worker  |
| Connector|    | (storage_manager.py)  |    | (cache_controller|
| Module   |    |                       |    |  /worker.py)     |
+----------+    | Manages multi-tier    |    | Connects to      |
                | storage backends:     |    | Controller for   |
                | - LocalCPUBackend     |    | cluster-wide     |
                | - LocalDiskBackend    |    | coordination     |
                | - RemoteBackend       |    +---------+--------+
                | - P2PBackend          |              |
                | - PluginBackends      |              v
                +-----------+-----------+    +---------+--------+
                            |                | Cache Controller  |
                            v                | Manager           |
                +-----------+-----------+    | (controller_mgr)  |
                | Memory Allocator      |    | - KVController    |
                | (Lazy/Mixed/Paged/    |    | - RegController   |
                |  CuFile/NUMA-aware)   |    | - ClusterExecutor |
                +-----------------------+    +------------------+
```

---

## 3. Major Components Breakdown

### 3.1 Integration Layer

**Files:** `lmcache/integration/vllm/`, `lmcache/integration/sglang/`

**Role:** Bridge between the serving engine (vLLM or SGLang) and LMCache internals.

| Component | File | Purpose |
|-----------|------|---------|
| `LMCacheConnectorV1Dynamic` | `lmcache_connector_v1.py` | Implements vLLM's `KVConnectorBase_V1` interface. Delegates all calls to `LMCacheConnectorV1Impl`. |
| `LMCacheConnectorV1Impl` | `vllm_v1_adapter.py` | The actual implementation. Creates `LMCacheEngine`, `GPUConnector`, `TokenDatabase`. Maps vLLM scheduler outputs to LMCache `store/retrieve/lookup` calls. Handles layer-by-layer pipelining. |
| `SGLangAdapter` | `sglang_adapter.py` | Equivalent adapter for the SGLang serving engine. |

**How it connects:**
1. vLLM/SGLang instantiates the connector at startup
2. `register_kv_caches()` is called once with the GPU KV cache tensors
3. On each forward pass: `start_load_kv()` triggers retrieval, `save_kv_layer()` triggers storage
4. `wait_for_layer_load()` / `wait_for_save()` synchronize async operations

---

### 3.2 LMCacheEngine (Core Orchestrator)

**File:** `lmcache/v1/cache_engine.py`

**Role:** The central class that coordinates all KV cache operations. It does NOT directly touch GPU memory or storage backends -- it delegates to specialized components.

**Key internal flow:**

```
                         User Request (tokens + mask)
                                    |
                                    v
                         +--------------------+
                         | Token Database     |
                         | process_tokens()   |---> (start, end, CacheEngineKey)
                         | Chunks tokens into |     for each chunk
                         | fixed-size pieces   |
                         | & computes hashes  |
                         +--------------------+
                                    |
                   +----------------+----------------+
                   |                                 |
                   v (Store)                         v (Retrieve)
          +--------+---------+              +--------+---------+
          | StorageManager   |              | StorageManager   |
          | .allocate()      |              | .batched_get()   |
          | Get MemoryObj    |              | Fetch MemoryObj  |
          +--------+---------+              +--------+---------+
                   |                                 |
                   v                                 v
          +--------+---------+              +--------+---------+
          | GPU Connector    |              | GPU Connector    |
          | .batched_from_   |              | .batched_to_gpu()|
          |  gpu()           |              | Copy CPU -> GPU  |
          | Copy GPU -> CPU  |              +--------+---------+
          +--------+---------+                       |
                   |                                 v
                   v                          Return hit mask
          +--------+---------+               (which tokens were
          | StorageManager   |                cached)
          | .batched_put()   |
          | Write to backends|
          +------------------+
```

**Sub-components owned by the engine:**

| Component | Purpose |
|-----------|---------|
| `TokenDatabase` | Chunks token sequences and computes content-based hashes (CacheEngineKey) |
| `EventManager` | Tracks async prefetch/loading operations (ONGOING -> DONE state machine) |
| `StorageManager` | Multi-backend storage with allocation, put, get, contains, eviction |
| `GPUConnector` | Moves KV data between GPU paged memory and CPU MemoryObjs |
| `LMCacheWorker` | (Optional) Communicates with centralized Cache Controller |
| `PinMonitor` | Monitors and manages pinned cache entries |
| `StatsMonitor` | Collects performance metrics (throughput, latency, hit rates) |
| `HealthMonitor` | Periodic health checks with fallback policies |

---

### 3.3 Token Database

**Files:** `lmcache/v1/token_database/` (ChunkedTokenDatabase, SegmentTokenDatabase)

**Role:** Converts raw token sequences into cache-addressable chunks with content-based hashing.

**How it works:**
1. Receives a token sequence (e.g., 1024 tokens)
2. Splits into fixed-size chunks (default `chunk_size=256` tokens)
3. Computes a **rolling hash** for each chunk (content-based, so identical text always produces the same key)
4. Returns `(start_idx, end_idx, CacheEngineKey)` for each chunk

**Two variants:**
- `ChunkedTokenDatabase`: Standard prefix-based chunking. Used for normal KV cache reuse.
- `SegmentTokenDatabase`: Segment-aware chunking for the **CacheBlend** feature. Handles non-prefix (arbitrary substring) reuse by segmenting at special delimiter tokens.

---

### 3.4 GPU Connector Module

**Files:** `lmcache/v1/gpu_connector/`

**Role:** Handles the GPU <-> CPU data transfer, abstracting away the different memory layouts used by vLLM and SGLang.

**Key challenge:** Different serving engines store KV caches in different GPU memory formats:
- vLLM Flash Attention: `List[num_layers] of [2, num_blocks, block_size, num_heads, head_size]`
- vLLM Flash Infer: `List[num_layers] of [num_blocks, 2, block_size, num_heads, head_size]`
- vLLM Cross-Layer: `[num_blocks, num_layers, 2, block_size, num_heads, head_size]`
- vLLM MLA: `List[num_layers] of [num_blocks, block_size, head_size]`
- SGLang MHA: `List[2] -> List[num_layers] of [page_buffer_size, num_heads, head_size]`
- SGLang MLA: `List[num_layers] of [page_buffer_size, 1, head_size]`

```
GPU Connector Hierarchy:
+---------------------------+
| GPUConnectorInterface     |  (Abstract base)
+---------------------------+
    |
    +-- VLLMPagedMemGPUConnectorV2      (vLLM batch, custom CUDA kernels)
    +-- VLLMPagedMemGPUConnectorV3      (vLLM batch, newer kernel version)
    +-- VLLMPagedMemLayerwiseGPUConnector (vLLM layer-by-layer pipeline)
    +-- VLLMBufferLayerwiseGPUConnector  (vLLM layer-by-layer for blending)
    +-- SGLangGPUConnector               (SGLang batch)
    +-- SGLangLayerwiseGPUConnector      (SGLang layer-by-layer)
    +-- MockGPUConnector                 (Testing)
```

**Key operations:**
- `batched_from_gpu(memory_objs, starts, ends, **kwargs)`: Reads KV data from GPU paged buffer into CPU MemoryObjs. Uses custom CUDA kernels (`lmc_ops`) for zero-copy or async DMA.
- `batched_to_gpu(memory_objs, starts, ends, **kwargs)`: Writes KV data from CPU MemoryObjs back to GPU paged buffer.
- `discover_gpu_kv_format()`: Auto-detects the GPU KV cache memory layout at runtime.

---

### 3.5 Storage Manager & Backends

**Files:** `lmcache/v1/storage_backend/storage_manager.py`, `lmcache/v1/storage_backend/`

**Role:** Multi-tier storage system with unified API for storing, retrieving, and managing KV cache chunks.

```
Storage Tier Architecture:
+=========================+
|    Storage Manager      |  Unified API for all backends
|  (storage_manager.py)   |
+=========================+
    |
    |  Tier 1 (Hot, fastest)
    +-- LocalCPUBackend         Pinned CPU memory (NUMA-aware)
    |                           - Fastest access (~microseconds)
    |                           - Limited by system RAM
    |
    |  Tier 2 (Warm)
    +-- LocalDiskBackend        Local SSD/NVMe storage
    |                           - Supports GDS (GPU Direct Storage)
    |                           - Moderate latency
    |
    |  Tier 3 (Cold/Remote)
    +-- RemoteBackend           Remote storage server (Redis, S3, etc.)
    |                           - Serialization via serde (naive/cachegen)
    |                           - Network latency
    |
    |  Special
    +-- P2PBackend              Peer-to-peer between instances
    |                           - Uses NIXL or custom RPC
    |                           - For disaggregated prefill
    |
    +-- PluginBackends          User-provided storage plugins
                                - Loaded dynamically at runtime
```

**Memory Allocator subsystem:**

```
Memory Allocator Hierarchy:
+-------------------------------+
| MemoryAllocatorInterface      |
+-------------------------------+
    |
    +-- MixedMemoryAllocator        Default: manages TensorMemory + PagedTensor pools
    +-- LazyMemoryAllocator         Starts small, background-expands to target size
    +-- CuFileMemoryAllocator       For GDS (GPU Direct Storage) aligned buffers
    +-- PagedTensorMemoryAllocator  For paged memory format
```

**MemoryObj:** The core data unit -- a reference-counted handle to a chunk of KV cache data in CPU memory. Supports pinning (prevents eviction during active use).

**Storage Manager key methods:**
- `allocate()` / `batched_allocate()`: Get MemoryObj from allocator
- `batched_put()`: Store MemoryObj to appropriate backend tier
- `batched_get()`: Retrieve MemoryObj from backends (checks tiers in priority order)
- `batched_contains()`: Check if keys exist (for lookup)
- `async_lookup_and_prefetch()`: Async background fetch
- `layerwise_batched_get()`: Layer-by-layer retrieval generator

---

### 3.6 Cache Controller (Cluster Coordination)

**Files:** `lmcache/v1/cache_controller/`

**Role:** Centralized coordination service for multi-instance LMCache deployments. Enables P2P KV cache sharing and cluster-wide cache awareness.

```
Cluster Architecture:
+------------------+     +------------------+     +------------------+
| vLLM Instance 1  |     | vLLM Instance 2  |     | vLLM Instance 3  |
| + LMCacheEngine  |     | + LMCacheEngine  |     | + LMCacheEngine  |
| + LMCacheWorker  |     | + LMCacheWorker  |     | + LMCacheWorker  |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
         | ZMQ (PUSH)            | ZMQ (PUSH)             | ZMQ (PUSH)
         | Register/Admit/       | Register/Admit/        | Register/Admit/
         | Heartbeat             | Heartbeat              | Heartbeat
         v                        v                        v
+========+========================+========================+==========+
|                    Cache Controller Manager                         |
|                  (controller_manager.py)                            |
|                                                                     |
|  +---------------------+    +---------------------+                 |
|  | Registration         |    | KV Controller       |                |
|  | Controller           |    | - Track which keys  |                |
|  | - Register/deregister|    |   are on which nodes|                |
|  |   worker instances   |    | - P2P lookup routing|                |
|  | - Health check via   |    | - Full sync         |                |
|  |   heartbeats         |    | - Move/compress/    |                |
|  +---------------------+    |   decompress cmds   |                |
|                              +---------------------+                |
|  +---------------------+                                            |
|  | Cluster Executor     |                                           |
|  | - Issue commands to  |                                           |
|  |   workers via ZMQ    |                                           |
|  | - Orchestrate cross- |                                           |
|  |   node operations    |                                           |
|  +---------------------+                                            |
+=======+=========================================================+===+
        |                                                         |
        | ZMQ (REQ/REP, PUB/SUB)                                  |
        v                                                         v
  Command responses                                    P2P lookup results
  (admit/evict/move                                   (which instance has
   confirmations)                                      the requested KV?)
```

**Components:**

| Component | File | Purpose |
|-----------|------|---------|
| `LMCacheControllerManager` | `controller_manager.py` | Main controller process. Receives messages via ZMQ, dispatches to sub-controllers. |
| `RegistrationController` | `controllers/registration_controller.py` | Manages worker instance registration, deregistration, and heartbeat-based health monitoring. |
| `KVController` | `controllers/kv_controller.py` | Tracks which KV cache keys exist on which workers. Handles P2P lookup, full sync, move/compress commands. |
| `LMCacheClusterExecutor` | `executor.py` | Issues commands to workers (move, compress, clear, etc.) via ZMQ. |
| `LMCacheWorker` | `worker.py` | Runs inside each LMCache instance. Sends admit/evict notifications to controller, receives commands. |
| `FullSyncSender` | `full_sync_sender.py` | Batched sync of all local cache keys to controller on registration. |

**Message types** (in `message.py`): RegisterMsg, DeRegisterMsg, HeartbeatMsg, BatchedKVOperationMsg (admit/evict), LookupMsg, MoveMsg, CompressMsg, FullSyncBatchMsg, etc.

---

### 3.7 Compute Module (Attention & Blending)

**Files:** `lmcache/v1/compute/`

**Role:** Implements the **CacheBlend** algorithm -- enabling reuse of KV caches for **non-prefix** text (arbitrary shared substrings, not just common prefixes).

```
CacheBlend Pipeline:
+--------------------------------------------------+
|  Standard KV Cache Reuse (prefix-only):          |
|  "System prompt | User query"                    |
|  [cached prefix]  [recompute]                    |
+--------------------------------------------------+

+--------------------------------------------------+
|  CacheBlend KV Cache Reuse (non-prefix):         |
|  "Doc A | Doc B | Doc C | Query"                 |
|  [cached A] [cache miss] [cached C] [recompute]  |
|                                                   |
|  Problem: Cached chunks have wrong positional     |
|  encodings (they were computed at different        |
|  positions originally)                            |
|                                                   |
|  Solution: CacheBlend uses selective recomputation |
|  to "blend" cached KV with fresh KV, fixing the   |
|  attention quality at chunk boundaries            |
+--------------------------------------------------+
```

**Sub-components:**

| Component | Purpose |
|-----------|---------|
| `AttentionInterface` | Abstract interface for attention computation (forward_contiguous, init_attn_metadata) |
| `LMCFlashAttnBackend` | Implementation wrapping vLLM's Flash Attention (FA2/FA3) |
| `FlashInferSparseAttention` | Implementation using FlashInfer's block-sparse attention |
| `LMCAttnMetadata` | Attention metadata (seq lengths, positions, masks) |
| `Blender` (`blend/blender.py`) | Core CacheBlend algorithm: selectively recomputes attention for boundary tokens between cached and non-cached segments |
| `BlendMetadata` | Tracks which tokens need recomputation vs. can use cached KV |
| `PositionalEncoding` (`positional_encoding.py`) | Applies RoPE (Rotary Position Embedding) corrections to cached KV caches |
| Model implementations (`models/llama.py`, `models/qwen3.py`) | Model-specific layer computation for recomputation during blending |

---

### 3.8 Configuration System

**Files:** `lmcache/v1/config.py`, `lmcache/v1/config_base.py`

**Role:** Comprehensive configuration management with ~80+ parameters.

```
Configuration Loading Priority:
1. Defaults (defined in _CONFIG_DEFINITIONS)
2. YAML config file (LMCACHE_CONFIG_FILE env var or explicit path)
3. Environment variables (LMCACHE_<PARAM_NAME> in uppercase)
4. Remote config server (optional, fetched via HTTP)
5. Programmatic overrides (passed at instantiation)
```

**Major config categories:**
- **Storage:** `local_cpu`, `local_disk`, `remote_url`, `max_local_cpu_size`, `max_local_disk_size`
- **Chunking:** `chunk_size` (default 256 tokens)
- **Features:** `use_layerwise`, `enable_blending`, `enable_p2p`, `enable_pd`, `enable_controller`
- **PD (Prefill-Decode disaggregation):** `pd_role`, `pd_buffer_size`, `pd_peer_host`
- **Lazy Memory:** `enable_lazy_memory_allocator`, `lazy_memory_initial_ratio`
- **Plugins:** `runtime_plugin_locations`, `storage_plugins`

---

### 3.9 Health Monitor

**Files:** `lmcache/v1/health_monitor/`

**Role:** Periodic health checking with automatic fallback when backends become unhealthy.

```
Health Monitor Flow:
+-------------------+
| HealthMonitor     |
| (periodic thread) |
+--------+----------+
         |
         | runs checks periodically
         v
+--------+----------+     +---------------------+
| RemoteBackendCheck|     | (future checks...)  |
| - Pings remote    |     |                     |
|   storage server  |     |                     |
+--------+----------+     +---------------------+
         |
         | unhealthy?
         v
+--------+----------+
| Fallback Policy   |
| - RECOMPUTE: skip |
|   all cache ops   |
| - LOCAL_CPU: use  |
|   only local CPU  |
+-------------------+
```

---

### 3.10 Internal API Server

**Files:** `lmcache/v1/internal_api_server/`

**Role:** FastAPI-based HTTP server for runtime introspection and control.

**API categories:**
- **Common:** metrics, log level control, thread management, environment info
- **vLLM-specific:** cache stats, freeze/unfreeze, hot cache toggle, chunk statistics, bypass mode, inference control
- **Controller-specific:** worker info, key stats

---

### 3.11 Serialization (Serde)

**Files:** `lmcache/storage_backend/serde/`

**Role:** Serializes/deserializes KV cache data for remote storage and network transfer.

| Serializer | Description |
|------------|-------------|
| `NaiveSerde` | Simple tensor serialization (torch.save/load equivalent) |
| `FastSerde` | Optimized binary format with minimal overhead |
| `CacheGenEncoder/Decoder` | Learned compression -- significantly reduces KV cache size for network transfer (from the CacheGen paper) |
| `SafeSerde` | SafeTensors-based serialization |

---

## 4. Workflow Diagrams

### 4.1 Store Workflow (GPU -> Storage)

```
  vLLM Forward Pass Completes
           |
           v
  Connector.save_kv_layer(layer_name, kv_tensor, attn_metadata)
           |
           v
  LMCacheEngine.store(tokens, mask, slot_mapping, ...)
           |
           +---> TokenDatabase.process_tokens(tokens, mask)
           |     Returns: [(start=0, end=256, key=hash_0),
           |               (start=256, end=512, key=hash_1), ...]
           |
           +---> For each chunk:
           |       StorageManager.allocate(kv_shapes, kv_dtypes)
           |       Returns: MemoryObj (pinned CPU buffer)
           |
           +---> GPUConnector.batched_from_gpu(memory_objs, starts, ends)
           |     Custom CUDA kernel copies from GPU paged KV buffer
           |     to CPU MemoryObjs (using slot_mapping for page table lookup)
           |     |
           |     +-- Zero-copy via pinned memory
           |     +-- Async DMA on CUDA stream
           |
           +---> StorageManager.batched_put(keys, memory_objs)
                 |
                 +-- LocalCPUBackend.put()     (keep in RAM)
                 +-- LocalDiskBackend.put()    (async write to SSD)
                 +-- RemoteBackend.put()       (serialize + send)
                 |
                 +-- If controller enabled:
                     LMCacheWorker sends BatchedKVOperationMsg(ADMIT)
                     to Controller via ZMQ
```

### 4.2 Retrieve Workflow (Storage -> GPU)

```
  vLLM Scheduler determines request needs prefill
           |
           v
  Connector.start_load_kv(forward_context)
           |
           v
  LMCacheEngine.lookup(tokens)
           |
           +---> TokenDatabase.process_tokens(tokens)
           |     Returns chunk keys
           |
           +---> StorageManager.batched_contains(keys)
           |     Checks each tier: CPU -> Disk -> Remote -> P2P
           |     Returns: hit_count (prefix length of consecutive hits)
           |
           +---> Returns hit_count to scheduler
           |     Scheduler adjusts num_computed_tokens
           |
  Later, during forward pass:
           |
           v
  LMCacheEngine.retrieve(tokens, mask, slot_mapping, ...)
           |
           +---> TokenDatabase.process_tokens(tokens, mask)
           |
           +---> StorageManager.batched_get(keys, location)
           |     Fetches MemoryObjs from the appropriate backend tier
           |     |
           |     +-- CPU: direct pointer return (fastest)
           |     +-- Disk: async read into allocated MemoryObj
           |     +-- Remote: deserialize network response
           |
           +---> GPUConnector.batched_to_gpu(memory_objs, starts, ends)
           |     Custom CUDA kernel copies from CPU MemoryObjs
           |     to GPU paged KV buffer (using slot_mapping)
           |
           +---> Return boolean mask of retrieved tokens
           |
           +---> MemoryObj.ref_count_down() (release back to allocator)
```

### 4.3 Layer-wise Pipeline (Overlapped Compute + Transfer)

```
  Timeline (layers 0..N-1):

  GPU Compute:  [Layer 0 attn] [Layer 1 attn] [Layer 2 attn] ... [Layer N-1]
                     |              |              |                  |
  Store Pipeline:    |              |              |                  |
    from_gpu:   [Copy L0->CPU] [Copy L1->CPU] [Copy L2->CPU]        |
    put:                       [Put L0]       [Put L1]         [Put LN-1]
                     ^              ^              ^
                     |              |              |
                yield             yield          yield    (generator protocol)

  Retrieve Pipeline:
    get:        [Get L0]       [Get L1]       [Get L2]    ... [Get LN-1]
    to_gpu:          [Copy L0->GPU] [Copy L1->GPU]        [Copy LN-1->GPU]
                     ^              ^              ^
                     |              |              |
                yield             yield          yield
```

This pipelining overlaps the data transfer of layer i with the GPU computation of layer i+1, hiding most of the CPU<->GPU transfer latency.

### 4.4 Async Lookup + Prefetch Workflow

```
  Scheduler Thread                    Background Async Loop
       |                                      |
       v                                      |
  async_lookup_and_prefetch(req_id, tokens)   |
       |                                      |
       +---> TokenDatabase.process_tokens()   |
       |     Compute all chunk keys           |
       |                                      |
       +---> StorageManager.async_lookup_     |
       |     and_prefetch(keys)        ------>|
       |                                      v
       |     EventManager.add_event(     [Background: check each
       |       LOADING, req_id, future)   tier for each key,
       |                                  start prefetch from
       |     (returns immediately)        remote/disk backends]
       |                                      |
       v                                      v
  [Scheduler continues                  [Prefetch completes]
   scheduling other                     EventManager.update_event_status(
   requests]                              LOADING, req_id, DONE)
       |
       v (later, when request is scheduled for execution)
  retrieve(tokens, req_id=req_id)
       |
       +---> EventManager.pop_event(LOADING, req_id)
       |     Gets the completed future with all MemoryObjs
       |
       +---> GPUConnector.batched_to_gpu(memory_objs)
       |
       v
  [Tokens served with cached KV -- minimal wait]
```

### 4.5 P2P Cross-Instance KV Sharing via Controller

```
  Instance A (has KV cache)              Controller              Instance B (needs KV cache)
       |                                     |                          |
       | Register + FullSync                 |                          | Register
       | (sends all local keys)              |                          |
       +------------------------------------>|<-------------------------+
       |                                     |                          |
       | BatchedKVOp(ADMIT, keys)            |                          |
       +------------------------------------>|                          |
       |                                     |                          |
       |                                     |   lookup(tokens)         |
       |                                     |<-------------------------+
       |                                     |                          |
       |                                     | P2P Lookup: "Instance A  |
       |                                     | has these keys"          |
       |                                     +------------------------->|
       |                                     |                          |
       |    [P2P direct transfer via NIXL/RPC]                          |
       |<-------------------------------------------------------------->|
       |    KV data transferred directly between instances              |
       |    (bypasses controller for data plane)                        |
```

### 4.6 CacheBlend Non-Prefix Reuse Workflow

```
  Input: "Doc_A tokens | Doc_B tokens | Doc_C tokens | Query tokens"

  Step 1: Token Database segments input by special delimiters
           Segment 0: Doc_A (tokens 0-255)    -> hash_A
           Segment 1: Doc_B (tokens 256-511)  -> hash_B
           Segment 2: Doc_C (tokens 512-767)  -> hash_C
           Segment 3: Query (tokens 768-1023) -> hash_Q

  Step 2: Lookup each segment independently
           hash_A: HIT  (cached from previous request)
           hash_B: MISS
           hash_C: HIT  (cached from different request)
           hash_Q: MISS

  Step 3: Retrieve cached segments
           Load KV for Doc_A and Doc_C from storage

  Step 4: CacheBlend recomputation
           Problem: Cached KV for Doc_C was originally at positions 0-255,
                    but now it needs to be at positions 512-767.
                    The positional encodings (RoPE) are wrong!

           Solution:
           a) Apply positional encoding correction (undo old RoPE, apply new)
           b) Selectively recompute attention at segment boundaries
              - Check a few "indicator layers" to measure attention quality
              - If quality is below threshold, recompute that token's KV
              - Typically only ~10-20% of boundary tokens need recomputation
           c) Blend: merge recomputed KV with cached KV

  Step 5: Forward pass uses blended KV cache
           [cached_A | recomputed_B | blended_C | recomputed_Q]
```

---

## 5. Data Flow Through the Full System

```
+--------+      +--------+      +--------+      +--------+      +--------+
| Tokens |----->| Token  |----->| Cache  |----->| Memory |----->| Storage|
| (input |      | Data-  |      | Engine |      | Alloc- |      | Back-  |
| sequence)     | base   |      | Key    |      | ator   |      | ends   |
|        |      |        |      | (hash) |      | (CPU   |      | (CPU/  |
|        |      | chunk  |      |        |      | buffer)|      | Disk/  |
|        |      | & hash |      |        |      |        |      | Remote)|
+--------+      +--------+      +--------+      +--------+      +--------+
                                    |                                |
                                    v                                v
                              +--------+                       +--------+
                              | GPU    |<--------------------->| Event  |
                              | Connect|   (async prefetch)    | Manager|
                              | or     |                       |        |
                              | (CUDA  |                       +--------+
                              | kernels|
                              +--------+
                                    ^
                                    |
                              +--------+
                              | GPU KV |
                              | Cache  |
                              | (paged |
                              | buffer)|
                              +--------+
```

---

## 6. Key Design Decisions & Patterns

### 6.1 Content-Based Hashing
KV caches are addressed by **content hash** of the token sequence, not by position or request ID. This means identical text always maps to the same cache key, enabling cross-request and cross-instance sharing without coordination.

### 6.2 Chunk-Based Granularity
Token sequences are split into fixed-size chunks (default 256 tokens). This provides:
- Efficient prefix matching (compare chunk-by-chunk)
- Bounded memory allocation units
- Natural eviction granularity

### 6.3 Multi-Tier Storage with Automatic Tiering
The StorageManager checks backends in priority order (CPU -> Disk -> Remote -> P2P). Writes go to the fastest available tier; reads cascade through tiers.

### 6.4 Zero-Copy GPU Transfer
Custom CUDA kernels (`lmc_ops`) handle the GPU<->CPU transfer with:
- Direct page table mapping (understands vLLM/SGLang paged memory layout)
- Pinned CPU memory for DMA transfers
- Async CUDA streams for non-blocking copies

### 6.5 Layer-wise Pipelining
Instead of waiting for all layers to transfer, the layer-wise mode overlaps:
- GPU computation of layer i+1 with CPU transfer of layer i
- This hides ~90% of transfer latency behind useful computation

### 6.6 Reference-Counted Memory Objects
`MemoryObj` uses reference counting to safely share buffers between the engine and storage backends. Pinning prevents eviction while data is in-flight.

### 6.7 Singleton Engine Pattern
`LMCacheEngineBuilder` ensures only one engine instance per `instance_id`, preventing resource conflicts in multi-process serving engines.

### 6.8 Generator/Coroutine Pattern for Pipelining
Layerwise connectors use Python generators (`yield`) to interleave GPU compute with data transfer. The engine drives the generator, sending one layer's data at a time. The `VLLMBufferLayerwiseGPUConnector` even uses ping-pong double buffering (two GPU intermediate buffers) to overlap layer i CPU->GPU copy with layer i-1 GPU buffer->paged memory copy.

### 6.9 TTL-Based Read/Write Locks (Distributed Module)
The `L1Manager` in the distributed module uses TTL-based locks for object lifecycle management. Write locks auto-expire after 600s, read locks after 300s. Read locks are re-entrant (count-based). This prevents deadlocks from crashed processes while maintaining correctness.

### 6.10 Observer Pattern for Eviction
The eviction system uses an observer/listener pattern: `L1Manager` notifies `EvictionPolicy` (via `L1ManagerListener` interface) about all object lifecycle events (create, read, write, delete). The LRU policy tracks access order in an `OrderedDict`. A background `EvictionController` thread checks memory watermarks every second and triggers eviction when usage exceeds the threshold.

---

## 7. Detailed CacheBlend Algorithm Walkthrough

The CacheBlend algorithm is LMCache's most sophisticated feature, enabling reuse of KV caches
for non-prefix text. Here is the step-by-step process:

```
Step 1: SEGMENTATION
  Input: "Doc_A tokens | Doc_B tokens | Doc_C tokens | Query tokens"
  SegmentTokenDatabase splits input at special delimiter tokens (" # # " by default)
  Each segment gets an independent content hash

Step 2: LOOKUP
  hash_A: HIT  (cached from previous request)
  hash_B: MISS
  hash_C: HIT  (cached from a DIFFERENT request, at DIFFERENT positions)
  hash_Q: MISS (new query)

Step 3: RETRIEVE (Layer-wise)
  Load cached KV for Doc_A and Doc_C from storage into GPU buffer
  Problem: Doc_C's KV was computed at positions [0-255] originally,
           but now needs to serve as positions [512-767]

Step 4: POSITIONAL ENCODING CORRECTION
  For each cached segment at wrong positions:
  a) Use FusedRope CUDA kernel to undo original RoPE encoding
  b) Apply new RoPE encoding for correct positions
  This fixes Key cache's positional information

Step 5: SELECTIVE RECOMPUTATION (the "Blending" step)
  For each transformer layer:
    a) Run QKV projection on ALL tokens (both cached and uncached)
    b) At "check layers" (e.g., layers 0, 8, 16, 24):
       - Compute L2 difference: diff_k[t] = ||K_new[t] - K_cached[t]||^2
       - Select top-k% most-different tokens (recomp_ratio, e.g., 10-20%)
       - These are the "important" tokens that need fresh computation
    c) At other layers:
       - Reuse the same important token indices from the last check layer
    d) Run attention ONLY on important tokens against full K/V sequence
       - Flash Attention: causal attention on full sequence
       - FlashInfer Sparse: block-sparse attention on selected blocks only
    e) Merge: overwrite cached K/V at important positions with recomputed values

Step 6: FORWARD PASS
  The blended KV cache (cached + selectively recomputed) is used for the
  standard forward pass. Only ~10-20% of tokens needed recomputation,
  saving 80-90% of prefill compute.
```

---

## 8. Cache Controller Communication Protocol

The controller uses 4 ZMQ communication channels:

```
                    +---------------------------+
                    |    Controller Manager     |
                    |                           |
                    |  +---------+ +---------+  |
                    |  |  PULL   | | ROUTER  |  |  +----------+
  Workers =========>|  | Socket  | | Socket  |<===>| ROUTER   |
  (fire-and-forget) |  |(port A) | |(port B) |  |  | Heartbeat|
                    |  +---------+ +---------+  |  | (port C) |
                    +---------------------------+  +----------+

  Channel 1: Worker -> Controller (PUSH -> PULL)
    - Fire-and-forget messages: KV admit/evict batches, full sync data
    - High throughput, no response needed

  Channel 2: Worker <-> Controller (DEALER <-> ROUTER)
    - Request-reply: Register, P2P lookup, full sync start/status
    - Worker needs response before proceeding

  Channel 3: Worker <-> Controller (DEALER <-> ROUTER, dedicated)
    - Heartbeat only, avoids head-of-line blocking
    - Includes piggyback commands (e.g., FullSyncCommand) in responses

  Channel 4: Controller -> Workers (REQ -> REP)
    - Control commands: Clear, Pin, Move, Compress, Health
    - Issued by ClusterExecutor, fan-out to all workers in parallel
```

**Full Sync Protocol (detailed):**

```
  Worker                              Controller
    |                                     |
    |  [1] Random delay (anti-thundering) |
    |  [2] Enter FREEZE mode              |
    |  [3] Collect all hot_cache keys     |
    |                                     |
    |--FullSyncStartMsg (DEALER)--------->|
    |                                     | Clear old keys for this worker
    |<-FullSyncStartRetMsg (sync_id)------|
    |                                     |
    |--FullSyncBatchMsg #1 (PUSH)-------->| Record batch, add keys
    |--FullSyncBatchMsg #2 (PUSH)-------->| Record batch, add keys
    |  ... (up to 2000 keys per batch)    |
    |--FullSyncBatchMsg #N (PUSH)-------->|
    |                                     |
    |--FullSyncEndMsg (PUSH)------------->| Mark worker as COMPLETED
    |                                     |
    |--FullSyncStatusMsg (DEALER)-------->|
    |<-FullSyncStatusRetMsg---------------|
    |  (is_complete, progress,            |
    |   can_exit_freeze,                  |
    |   missing_batches)                  |
    |                                     |
    |  [If missing batches: resend them]  |
    |  [If can_exit_freeze (>=80% done):  |
    |     exit FREEZE mode]               |
    |                                     |
```

---

## 9. vLLM Integration Request Lifecycle

This shows the complete lifecycle of a single request through vLLM + LMCache:

```
  Phase 1: SCHEDULING
  ====================
  vLLM Scheduler receives new request
        |
        v
  LMCacheConnectorV1Impl.get_num_new_matched_tokens(request, num_computed)
        |
        +-- Apply multimodal hashes to token IDs (if applicable)
        +-- LMCacheEngine.lookup(tokens, pin=True)
        |     +-- TokenDatabase.process_tokens() -> chunk keys
        |     +-- StorageManager.batched_contains() -> hit count
        +-- Return: number of tokens with external cache hit
        |
        v
  Scheduler allocates GPU blocks (skipping externally cached tokens)
        |
        v
  LMCacheConnectorV1Impl.update_state_after_alloc()
        +-- Clear lookup state, prepare load/save specs
        |
        v
  LMCacheConnectorV1Impl.build_connector_meta(scheduler_output)
        +-- For each scheduled request, create ReqMeta with:
        |     LoadSpec (which tokens to load from cache)
        |     SaveSpec (which tokens to save after compute)
        |     slot_mapping (GPU page table mapping)
        +-- Return LMCacheConnectorMetadata

  Phase 2: FORWARD PASS (Worker)
  ==============================
  LMCacheConnectorV1Impl.start_load_kv(forward_context)
        |
        +-- For each request with load_spec:
        |     Non-layerwise: LMCacheEngine.retrieve(tokens, mask, slot_mapping)
        |     Layerwise: LMCacheEngine.retrieve_layer() -> generator
        |     Blending: LMCBlender.blend(tokens, mask)
        |
        v
  [Forward pass begins]
        |
        +-- For each transformer layer:
        |     LMCacheConnectorV1Impl.wait_for_layer_load(layer_name)
        |       +-- Advance layerwise retriever generator by one step
        |     [Run attention with loaded KV cache]
        |     LMCacheConnectorV1Impl.save_kv_layer(layer_name, kv_tensor)
        |       +-- At layer 0: create store_layer() generators
        |       +-- Advance all store generators by one step
        |
        v
  [Forward pass ends]
        |
        v
  LMCacheConnectorV1Impl.wait_for_save()
        +-- Non-layerwise: LMCacheEngine.store(tokens, mask, slot_mapping)
        +-- Layerwise: advance store generators to completion
        +-- Unpin all lookup pins

  Phase 3: CLEANUP
  ================
  LMCacheConnectorV1Impl.request_finished(request, block_ids)
        +-- Remove request tracker
        +-- Cancel any pending async lookups
```

---

## 10. Observability & Metrics Architecture

```
+------------------------------------------------------------------+
|  Every LMCache Operation (store/retrieve/lookup/remote I/O/...)  |
+----------------------------------+-------------------------------+
                                   |
                                   v
                    +--------------+---------------+
                    |      LMCStatsMonitor          |
                    |  (Thread-safe Singleton)       |
                    |                                |
                    |  Accumulates per-interval:     |
                    |  - Request counts & hit rates  |
                    |  - Timing breakdowns           |
                    |  - Remote I/O bytes & latency  |
                    |  - P2P transfer stats          |
                    |  - Memory usage                |
                    |  - Health check results        |
                    +--------------+---------------+
                                   |
                        periodic flush (background thread)
                                   |
                    +--------------v---------------+
                    |     LMCacheStatsLogger         |
                    |  (Daemon Thread)               |
                    |                                |
                    |  Every N seconds:              |
                    |  1. get_stats_and_clear()      |
                    |  2. Push to Prometheus          |
                    |  3. Report to usage logger     |
                    +--------------+---------------+
                                   |
                    +--------------v---------------+
                    |     PrometheusLogger           |
                    |  (Singleton)                   |
                    |                                |
                    |  20+ Counters                  |
                    |  10+ Gauges                    |
                    |  15+ Histograms                |
                    |  Dynamic per-backend metrics   |
                    +-------------------------------+

  Prometheus scrape endpoint: /metrics (via Internal API Server)

  Key Metrics:
  - lmcache:num_retrieve_requests, lmcache:num_hit_tokens
  - lmcache:retrieve_hit_rate, lmcache:time_to_retrieve
  - lmcache:store_speed, lmcache:retrieve_speed
  - lmcache:remote_time_to_get, lmcache:remote_time_to_put
  - lmcache:local_cache_usage, lmcache:remote_cache_usage
  - lmcache:remote_ping_latency, lmcache:lmcache_is_healthy
  - lmcache:p2p_transfer_speed, lmcache:forced_unpin_count
```

---

## 11. Component File Map

| Component | Key Files | Lines |
|-----------|----------|-------|
| **Core Engine** | `v1/cache_engine.py` | ~2000 |
| **Configuration** | `v1/config.py`, `v1/config_base.py` | ~1600 |
| **Token Database** | `v1/token_database/` | ~500 |
| **GPU Connector** | `v1/gpu_connector/gpu_connectors.py` | ~1800 |
| **GPU Connector Utils** | `v1/gpu_connector/utils.py`, `gpu_ops.py` | ~500 |
| **Storage Manager** | `v1/storage_backend/storage_manager.py` | ~800 |
| **Local CPU Backend** | `v1/storage_backend/local_cpu_backend.py` | ~400 |
| **Remote Backend** | `v1/storage_backend/remote_backend.py` | ~600 |
| **P2P Backend** | `v1/storage_backend/p2p_backend.py` | ~500 |
| **Memory Management** | `v1/memory_management/` | ~1500 |
| **Lazy Allocator** | `v1/lazy_memory_allocator.py` | ~270 |
| **Cache Controller** | `v1/cache_controller/controller_manager.py` | ~530 |
| **Controller Worker** | `v1/cache_controller/worker.py` | ~630 |
| **Controller Executor** | `v1/cache_controller/executor.py` | ~460 |
| **Controller Messages** | `v1/cache_controller/message.py` | ~830 |
| **Controller Registry** | `v1/cache_controller/utils.py` | ~680 |
| **Full Sync** | `v1/cache_controller/full_sync_sender.py` | ~475 |
| **KV Controller** | `v1/cache_controller/controllers/kv_controller.py` | ~440 |
| **Registration** | `v1/cache_controller/controllers/registration_controller.py` | ~280 |
| **Full Sync Tracker** | `v1/cache_controller/controllers/full_sync_tracker.py` | ~470 |
| **Blender** | `v1/compute/blend/blender.py` | ~170 |
| **Model Wrappers** | `v1/compute/models/base.py` | ~140 |
| **Attention Backends** | `v1/compute/attention/` | ~550 |
| **Positional Encoding** | `v1/compute/positional_encoding.py` | ~200 |
| **vLLM Integration** | `integration/vllm/vllm_v1_adapter.py` | ~1630 |
| **vLLM Connector** | `integration/vllm/lmcache_connector_v1.py` | ~210 |
| **SGLang Integration** | `integration/sglang/sglang_adapter.py` | ~330 |
| **Health Monitor** | `v1/health_monitor/base.py` | ~590 |
| **Observability** | `observability.py` | ~1910 |
| **Event Manager** | `v1/event_manager.py` | ~130 |
| **KV Layer Groups** | `v1/kv_layer_groups.py` | ~210 |
| **API Server** | `v1/api_server/__main__.py` | ~540 |
| **Internal API** | `v1/internal_api_server/` | ~1200 |
| **Serialization** | `storage_backend/serde/` | ~500 |
