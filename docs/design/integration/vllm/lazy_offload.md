# Lazy Offload in LMCache MP Connector

## 1. vLLM SimpleCPUOffloadConnector Lazy Mode

### 1.1 Overview

vLLM implements a `SimpleCPUOffloadConnector` at
`vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py`
that supports two modes: **eager** and **lazy**.

In **eager mode**, every newly computed KV block is stored to CPU
immediately after it is confirmed. In **lazy mode**, stores are deferred
until GPU memory pressure is detected — the connector walks the GPU
BlockPool's free queue (LRU order) and offloads blocks that are about to
be evicted.

### 1.2 Architecture

The connector is split into three components:

| Component | Role |
|-----------|------|
| `SimpleCPUOffloadConnector` | Entry point; routes to scheduler or worker based on `KVConnectorRole` |
| `SimpleCPUOffloadScheduler` | Scheduler-side: decides *which* blocks to offload and *when* |
| `SimpleCPUOffloadWorker` | Worker-side: executes GPU↔CPU DMA copies on low-priority CUDA streams |

Data flows via `SimpleCPUOffloadMetadata` (scheduler → worker) and
`SimpleCPUOffloadWorkerMetadata` (worker → scheduler).

### 1.3 Lazy Mode: Core Algorithm

The lazy mode is implemented in `SimpleCPUOffloadScheduler._prepare_lazy_store_specs()`:

```python
def _prepare_lazy_store_specs(self) -> tuple[list[int], list[int], list[str]]:
    """Single-pass cursor walk: offload cached GPU blocks near eviction."""
    free_queue = gpu_pool.free_block_queue

    for covered, node in enumerate(free_queue.iter_blocks_after(self._cursor)):
        if covered >= self._target_free or len(gpu_ids) >= num_cpu_free:
            break

        self._cursor = node

        # Skip if: no hash, null, or already cached in CPU
        if (bhash is not None
            and not node.is_null
            and cpu_pool.cached_block_hash_to_block.get_one_block(bhash) is None):
            gpu_ids.append(node.block_id)
            block_hashes.append(bhash)

    # Batch-allocate CPU blocks and stamp hashes
    if gpu_ids:
        cpu_blocks = cpu_pool.get_new_blocks(len(gpu_ids))
        cpu_ids = [blk.block_id for blk in cpu_blocks]
        for cpu_blk, bhash in zip(cpu_blocks, block_hashes):
            cpu_blk._block_hash = bhash

    return gpu_ids, cpu_ids, []
```

Key points:
- A **cursor** tracks the last scanned position in the GPU free queue.
- `target_free` is estimated from `max_num_batched_tokens` to ensure
  enough free blocks remain for new requests.
- GPU blocks are **touched** (moved to MRU end) during in-flight copies
  to prevent eviction.
- The CPU side maintains its own `KVCacheCoordinator` with a `BlockPool`
  for prefix-cache matching.

### 1.3.1 GPU Block Protection: Touch and Free

The `SimpleCPUOffloadConnector` uses a simple but effective strategy:
`request_finished()` **always returns `False`**, meaning vLLM immediately
frees the request's blocks. In-flight DMA copies are protected by
**explicit `touch()`/`free_blocks()` ref_cnt pairs** managed by the
connector itself.

#### 1.3.1.1 `request_finished` Always Returns `False`

```python
# vllm/v1/simple_kv_offload/manager.py — SimpleCPUOffloadScheduler
def request_finished(self, request, block_ids) -> tuple[bool, dict | None]:
    """Always returns (False, None). GPU blocks are protected by ref_cnt,
    so the scheduler can free blocks immediately."""
    # ... cleanup pending CPU hits and load/store state ...
    return False, None
```

This means vLLM's scheduler immediately decrements `ref_cnt` on all the
request's blocks when it finishes. The connector does **not** hold blocks
hostage — it relies on its own `touch()` calls to keep in-flight blocks
alive.

#### 1.3.1.2 How `touch()` and `free_blocks()` Work

The `BlockPool` uses `ref_cnt` to track block liveness:

```python
# vllm/v1/core/block_pool.py
def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
    """Increment ref_cnt; remove from free queue if it was there."""
    for block in blocks:
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.remove(block)
        block.ref_cnt += 1

def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
    """Decrement ref_cnt; return to free queue when it reaches 0."""
    for block in ordered_blocks:
        block.ref_cnt -= 1
        if block.ref_cnt == 0 and not block.is_null:
            # Blocks with hash go to MRU end (prefix cache candidates)
            # Blocks without hash go to LRU end (evict first)
            ...
    free_block_queue.prepend_n(blocks_without_hash)
    free_block_queue.append_n(blocks_with_hash)
```

Key invariant: **A block with `ref_cnt > 0` is never in the free queue
and cannot be allocated to other requests.**

#### 1.3.1.3 Store Path: Touch on Submit, Free on Completion

When blocks are selected for offloading (both lazy and eager modes),
the connector **touches** them to prevent eviction during the async DMA:

```python
# Lazy mode — _prepare_lazy_store_specs()
if gpu_ids:
    # Touch GPU blocks to prevent eviction during async copy
    gpu_pool.touch([gpu_pool.blocks[bid] for bid in gpu_ids])

# Eager mode — _prepare_eager_store_specs()
if cpu_block_ids:
    # Touch GPU blocks to prevent freeing during async copy
    gpu_block_pool.touch([gpu_block_pool.blocks[bid] for bid in gpu_block_ids])
```

When the DMA completes (reported by worker via `update_connector_output`),
the connector **frees** the touched blocks:

```python
# _process_store_completion() — called when store event is fully done
def _process_store_completion(self, gpu_block_ids, cpu_block_ids):
    # Register CPU blocks in prefix cache
    for cpu_block in cpu_blocks:
        cpu_block_pool.cached_block_hash_to_block.insert(bhash, cpu_block)

    # Free CPU blocks' ref_cnt (they become prefix cache entries)
    self.cpu_block_pool.free_blocks(cpu_blocks)
    # Free GPU blocks' ref_cnt (the extra touch ref from submit time)
    self._gpu_block_pool.free_blocks(
        self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids
    )
```

This creates a **balanced ref_cnt lifecycle**:

```
[Submit store]  gpu_pool.touch(blocks)     → ref_cnt += 1
[Request ends]  vLLM scheduler frees       → ref_cnt -= 1  (from request ownership)
[DMA completes] gpu_pool.free_blocks(...)  → ref_cnt -= 1  (from touch)
                                             ref_cnt == 0 → block enters free queue
```

Even if the request finishes before the DMA completes, the block stays
protected because the connector's `touch()` added an extra ref_cnt.


#### 1.3.1.4 Summary: ref_cnt Lifecycle

```
                    Store (GPU→CPU)                    
                    ───────────────                    
Submit:             gpu.touch() [+1]                  
                    (ESSENTIAL: only protection)       

Request finishes:   vLLM frees request blocks [-1]    
                    (block still alive: touch ref)    
                                                       

DMA completes:      gpu.free_blocks() [-1]            
                    cpu registered in prefix cache    
                    → block enters free queue          
```

### 1.4 Worker-Side: DMA Copy

The worker uses a `DmaCopyBackend` that runs a background thread:

1. Stores wait for a **compute-done event** (ensures KV data is written).
2. Copies are launched via `cuMemcpyBatchAsync` on dedicated low-priority
   CUDA streams.
3. Completion is tracked via `torch.Event` with monotonic event indices.

### 1.5 Key Design: 1:1 GPU/CPU Block Correspondence

vLLM's block hash is a prefix-chained hash — block N's hash depends on
all preceding blocks:

```python
# vllm/v1/core/kv_cache_utils.py
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, ...):
    """Hash depends on parent block hash + current block tokens."""
    if not parent_block_hash:
        parent_block_hash = NONE_HASH
    return BlockHash(
        hash_function((parent_block_hash, tuple(curr_block_token_ids), extra_keys))
    )
```

So vLLM's block hash is **NOT independent** — it uses rolling prefix
hashing just like LMCache. However, the critical property that makes
lazy offload work is:

> **GPU blocks and CPU blocks have a 1:1 correspondence.** Each CPU
> block maps directly to one GPU block, so the CPU `BlockPool` can
> directly reuse the `block_hash` already stamped on each GPU block
> without any recomputation.

When a GPU block enters the free queue, its `block_hash` is already
computed and stored on the `KVCacheBlock` object. The lazy offload
scheduler simply:
1. Reads `node.block_hash` from the GPU free queue block.
2. Checks if the CPU `BlockPool` already has a block with the same hash.
3. If not, allocates a CPU block and stamps the **same hash** on it.
4. Copies the KV data GPU→CPU.

No hash recomputation is needed. No token IDs are needed. The hash is
just carried over from GPU block to CPU block as-is, because they are
1:1 corresponding units.

### 1.6 Why This Approach Is Incompatible with LMCache MP Connector

LMCache's store granularity (**chunk**) differs from vLLM's block
granularity, and this mismatch is the root cause of incompatibility.

#### 1.6.1 Granularity Mismatch

| Property | vLLM SimpleCPUOffload | LMCache MP Connector |
|----------|----------------------|---------------------|
| GPU unit | vLLM block (e.g. 16 tokens) | vLLM block (same) |
| CPU/store unit | CPU block (**same size** as GPU block) | LMCache chunk (e.g. 256 tokens = **16 vLLM blocks**) |
| Hash reuse | ✅ CPU block directly reuses GPU block's hash | ❌ Chunk hash must be computed from token_ids over chunk_size tokens |
| Free queue unit | 1 block → 1 CPU block (1:1 mapping) | 1 block ≠ 1 chunk (need 16 blocks to form 1 chunk) |

Example with `vllm_block_size=16`, `lmcache_chunk_size=256`:
- One LMCache chunk = 16 vLLM blocks.
- When one GPU block enters the free queue, it is only 1/16 of a chunk.
- The other 15 blocks of the same chunk may still be in use by active
  requests, or may have entered the free queue at different times.

#### 1.6.2 Hash Cannot Be Reused

Even if all 16 blocks of a chunk happen to be in the free queue
simultaneously, LMCache **cannot reuse their vLLM block hashes**:

- vLLM block hash: `hash(parent_block_hash, tokens[0:16])`
- LMCache chunk hash: `rolling_hash(prev_chunk_hash, tokens[0:256])`

These are completely different hash functions operating at different
granularities. LMCache must recompute its own chunk hash from the raw
token IDs, which requires:
1. The full token sequence (to compute the rolling prefix hash).
2. Knowledge of chunk boundaries (which 256-token range this chunk
   covers).

Both are only available while the request is still active.

#### 1.6.3 Chunk Assembly Problem

Even ignoring the hash issue, assembling a complete chunk from free-queue
blocks is problematic:

```
Request A: [block0][block1]...[block15] [block16]...[block31] ...
                    chunk 0                     chunk 1

Free queue (LRU order): block3, block17, block0, block22, block1, ...
```

- Blocks from the same chunk may be scattered across the free queue.
- Some blocks of a chunk may still be referenced (ref_cnt > 0) by other
  requests sharing the same prefix (APC).
- There is no efficient way to detect "all blocks of chunk N are now
  free" from the free queue's LRU iteration.

#### 1.6.4 Additional Architectural Differences

| Aspect | vLLM SimpleCPUOffload | LMCache MP Connector |
|--------|------------------|-------------------|
| CPU storage | Local pinned tensors (same process) | Remote LMCache server (separate process via ZMQ + CUDA IPC) |
| Copy mechanism | `cuMemcpyBatchAsync` (direct DMA) | Server-side `transfer_kv_per_object_group` (IPC + kernel + D2H) |
| Latency | same-process | cross-process IPC |
| Block pool access | Direct (scheduler owns it) | Indirect (scheduler has access, but worker/server do not) |
| Dedup | Local CPU `BlockPool` hash map | Server-side `StorageManager` (cross-process query needed) |

#### 1.6.5 Summary

vLLM's lazy offload works because **CPU and GPU use the same block size**,
allowing direct hash reuse without recomputation or token ID access. In
LMCache, the chunk size is larger than the vLLM block size, so:

1. **Hash cannot be reused** — LMCache's chunk hash operates at a
   different granularity and uses a different hash function.
2. **Hash must be recomputed** — which requires the full token sequence
   (only available while the request is active).
3. **Free queue granularity mismatch** — one free-queue block ≠ one
   LMCache chunk; assembling complete chunks from scattered free blocks
   is impractical.

This makes a direct port of vLLM's lazy offload approach infeasible for
the LMCache MP connector.

## 2. LMCache MP Connector: Threshold-Triggered Lazy Offload

### 2.1 Design Overview

Since LMCache cannot reuse vLLM block hashes and requires token IDs for
key construction, the lazy offload strategy must **buffer store metadata
while the request is still active**, and defer the actual store
submission until GPU memory pressure is detected.

The core idea:
1. In `build_connector_meta`, instead of immediately submitting stores,
   buffer the store metadata (token_ids, block_ids, start/end, etc.)
   into a **pending store queue** in the scheduler adapter.
2. Track the total number of GPU blocks consumed by pending (buffered)
   stores.
3. When the accumulated GPU blocks reach a configurable **threshold**
   (percentage of total GPU blocks), trigger offload by draining entries
   from the buffer queue into the actual store submission path.

```mermaid
flowchart TB
    subgraph Scheduler["build_connector_meta (each step)"]
        GM["GetStoreMetadata(tracker)"]
        BQ["Buffer Queue<br/>(pending store entries)"]
        TH{"GPU blocks in queue<br/>≥ threshold?"}
        GM -->|"store metadata"| BQ
        BQ --> TH
        TH -->|"Yes"| DRAIN["Drain queue → emit store ops"]
        TH -->|"No"| SKIP["Skip store this step"]
    end

    DRAIN -->|"store ops in<br/>connector_metadata"| Worker
    Worker -->|"submit_store_request"| Server["LMCache Server"]
```

### 2.2 Buffer Queue Design

```python
@dataclass
class PendingStoreEntry:
    request_id: str
    token_ids: list[int]
    block_ids: list[list[int]]   # per engine group
    start: int
    end: int
    cache_salt: str
    num_gpu_blocks: int          # total GPU blocks this entry occupies

class PendingStoreQueue:
    def __init__(self, threshold_ratio: float, total_gpu_blocks: int):
        self._queue: deque[PendingStoreEntry] = deque()
        self._total_buffered_blocks: int = 0
        self._threshold_blocks: int = int(total_gpu_blocks * threshold_ratio)

    def enqueue(self, entry: PendingStoreEntry) -> None:
        self._queue.append(entry)
        self._total_buffered_blocks += entry.num_gpu_blocks

    @property
    def should_offload(self) -> bool:
        return self._total_buffered_blocks >= self._threshold_blocks

    def drain(self, max_entries: int | None = None) -> list[PendingStoreEntry]:
        """Drain entries from the queue for store submission."""
        ...
```

### 2.3 Integration with `build_connector_meta`

The modification is in the scheduler-side `build_connector_meta` logic:

```python
# Current eager behavior (simplified):
for tracker in active_trackers:
    meta = GetStoreMetadata(tracker, ...)
    if meta:
        store_metas.append(meta)  # immediately emit

# New lazy behavior:
for tracker in active_trackers:
    meta = GetStoreMetadata(tracker, ...)
    if meta:
        entry = PendingStoreEntry(
            request_id=meta.request_id,
            token_ids=meta.op.token_ids,
            block_ids=meta.op.block_ids,
            start=meta.op.start,
            end=meta.op.end,
            cache_salt=meta.cache_salt,
            num_gpu_blocks=sum(len(g) for g in meta.op.block_ids),
        )
        self._pending_store_queue.enqueue(entry)

# Check threshold and drain
if self._pending_store_queue.should_offload:
    entries = self._pending_store_queue.drain()
    for entry in entries:
        store_metas.append(to_request_metadata(entry))
```

### 2.4 Threshold Trigger Mechanism

The threshold determines when buffered stores are flushed:

```
threshold_ratio = configured percentage (e.g. 0.8)
threshold_blocks = total_gpu_blocks * threshold_ratio

Trigger condition:
    sum(num_gpu_blocks for all entries in queue) >= threshold_blocks
```

When triggered, the queue is drained (partially or fully) and the
resulting store ops are emitted in the connector metadata for the worker
to execute.

Configuration:
```
lmcache.mp.lazy_offload = true/false          (default: false)
lmcache.mp.lazy_offload_threshold = 0.8       (trigger when 80% of GPU blocks are buffered)
lmcache.mp.lazy_offload_drain_ratio = 0.5     (drain 50% of queue when triggered)
```

### 2.5 Drain Strategy

When the threshold is reached, not all entries need to be drained at
once. Options:

- **FIFO drain**: Drain the oldest entries first (they hold the oldest
  GPU blocks, most likely to be evicted soon).
- **Partial drain**: Drain a fixed ratio (e.g. 50%) of the queue to
  bring the buffered block count below the threshold.
- **Priority drain**: Drain entries with the longest prefix first (higher
  reuse probability).

FIFO with partial drain is the simplest and most predictable:

```python
def drain(self, target_ratio: float = 0.5) -> list[PendingStoreEntry]:
    target_blocks = int(self._total_buffered_blocks * target_ratio)
    drained = []
    drained_blocks = 0
    while self._queue and drained_blocks < target_blocks:
        entry = self._queue.popleft()
        drained.append(entry)
        drained_blocks += entry.num_gpu_blocks
    self._total_buffered_blocks -= drained_blocks
    return drained
```

### 2.6 Interaction with `request_finished`

When a request finishes:
1. `request_finished` returns `True` — vLLM holds the blocks.
2. The pending store entries for this request remain in the buffer queue.
3. Eventually the threshold triggers, the entries are drained and
   submitted as store ops.
4. The worker's `get_finished` reports the request_id only after all
   stores complete.
5. vLLM then releases the blocks.

No special handling is needed at `request_finished` time — the existing
async store lifecycle already guarantees blocks are held until stores
complete.

### 2.7 Edge Case: Queue Overflow Without Trigger

If many short requests finish quickly but the threshold is never reached
(e.g. low load), entries could accumulate indefinitely. Mitigations:

- **Time-based fallback**: If an entry has been in the queue longer than
  N seconds (or N steps), force-drain it regardless of threshold.
- **Request-finished flush**: When `request_finished` is called, if the
  request still has pending entries in the queue, mark them as
  high-priority for the next drain cycle.

### 2.8 Worker-Side: No Changes Required

The worker receives standard `LoadStoreOp` objects in the connector
metadata and calls `submit_store_request` as usual. It is unaware of
whether the store was triggered eagerly or lazily — the interface is
identical.

### 2.9 Summary

| Aspect | Description |
|--------|-------------|
| **When to buffer** | Every step in `build_connector_meta`, when new storable chunks are detected |
| **What to buffer** | `PendingStoreEntry` containing token_ids, block_ids, start/end (all info needed for key construction) |
| **When to flush** | When total buffered GPU blocks ≥ threshold (configurable ratio of total GPU blocks) |
| **How to flush** | FIFO partial drain → emit as store ops in connector metadata |
| **Worker changes** | None — receives standard store ops |
| **Server changes** | None — receives standard STORE requests |
