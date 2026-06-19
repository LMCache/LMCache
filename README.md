# LMCache Engine-Driven Multi-Group Fork

**Repository:** https://github.com/efschu/LMCache  
**Branch:** `dev-engine-driven-multigroup`

## Overview

This fork implements **engine-driven KV cache transfer** for **hybrid multi-group models** (e.g., Qwen3.6-27B with GDN/Mamba + Attention groups) in LMCache v1.

## Updates
- [2026/05] 🔥 Agentic workload benchmark on AMD MI300X ([blog](https://blog.lmcache.ai/en/2026/05/12/benchmarking-lmcache-for-multi-turn-agentic-workloads-on-amd-mi300x/)).
- [2026/04] 🔥 LMCache's new multiprocess (MP) architecture release ([blog](https://blog.lmcache.ai/en/2026/04/03/lmcaches-new-architecture-boosts-moe-inference-performance-by-10x/)).
- [2026/03] LMCache at GTC 2026 ([post](https://www.linkedin.com/posts/lmcache-lab_llm-opensource-nvidiagtc-activity-7442721875664826369-pMAu?utm_source=share&utm_medium=member_desktop&rcm=ACoAADkIIvQBTyG53kXXX70OZdE5rhpllYQqmIA)).
- [2026/01] LMCache multi-node P2P CPU memory sharing, from experimental feature to production ([blog](https://blog.lmcache.ai/en/2026/01/21/p2p-1/)).
### Problem Statement

The original LMCache `lmcache server` process allocated **666 MB VRAM per GPU** even though it should be a pure CPU-caching daemon. This happens because:

```
vLLM Worker (GPU)
  └─ sends CUDA-IPC handles → LMCache Server
  └─ Server opens CUDA-IPC → creates CUDA Primary Context (~550 MB/GPU)
  └─ Server executes multi_layer_block_kv_transfer CUDA kernels
```

Additionally, the engine-driven path was **blocked for hybrid models** due to a hard check:

```python
def _single_group_block_ids(block_ids: list[list[int]]) -> list[int]:
    if len(block_ids) != 1:
        raise RuntimeError(
            "engine-driven transfer does not support hybrid KV cache groups"
        )
```

Qwen3.6-27B has at least 2 groups (Attention + GDN-State), so this check blocked engine-driven mode entirely.

## Solution: Engine-Driven Multi-Group Transfer

### Architecture

```
vLLM Worker (GPU)
  └─ executes GPU→CPU copy itself (gather_paged_kv_to_cpu)
  └─ sends CPU bytes via COMMIT_STORE

LMCache Server (CPU-only, NO VRAM!)
  └─ receives CPU bytes, deserializes
  └─ stores in L1-RAM / L2-Disk
```

### Key Changes

#### 1. Per-Group Layout Metadata (`custom_types.py`)

Added `GroupLayoutInfo` to track per-group metadata:
- `block_size`, `num_layers`, `hidden_dim_size`, `dtype_str`
- `use_mla` (Modified Local Attention flag)
- `tokens_per_block` (from EngineGroupInfo)

Extended `RegisterEngineDrivenContextPayload` with optional `group_layouts: list[GroupLayoutInfo]`.

#### 2. Multi-Group Gather/Scatter Functions (`transfer_context/base.py`)

- `slice_kv_caches_for_group()`: Extract layer subset for one group
- `gather_paged_kv_multi_group_to_cpu()`: Gather all groups to CPU tensors
- `scatter_cpu_multi_group_to_paged_kv()`: Scatter CPU tensors back to GPU paged KV

#### 3. Worker-Side Multi-Group Support (`transfer_context/worker_transfer.py`)

- `EngineDrivenTransferContext.register()`: Sends per-group layout metadata
- `EngineDrivenTransferContext.submit_store()`: Gathers all groups, serializes as pickle blob
- `EngineDrivenTransferContext.submit_retrieve()`: Deserializes blob, scatters to all groups

#### 4. Server-Side Multi-Group Support (`modules/engine_driven_transfer.py`)

- `register_kv_cache_engine_driven_context()`: Creates per-group `MemoryLayoutDesc` entries
- `_commit_store_multi_group()`: Stores to all Object Groups
- `_prepare_retrieve_multi_group()`: Loads all groups, returns serialized blob

#### 5. Multi-Group Prefetch Awaiting (`modules/lookup.py`)

Fixed a race condition where multi-group prefetch handles (groups 1-3) were submitted but never awaited:

- Added `multi_group_handles: list[PrefetchHandle]` to `_PrefetchJob` dataclass
- In `end_session()`: await primary handle + all `multi_group_handles` via `query_prefetch_status`
- Call `finish_read_prefetched` for **all groups** (group-0 through group-N)
- Touch L1 keys only for group-0 (matching original behavior)

## Build Instructions

### Prerequisites

- Docker
- NVIDIA Docker runtime (for GPU access during build)
- Git

### Build Manylinux Wheel

```bash
# Clone the repository
git clone https://github.com/efschu/LMCache.git
cd LMCache
git checkout dev-engine-driven-multigroup

# Build wheel using manylinux Docker container
docker run --rm \
  -v $(pwd):/io \
  -e BUILDKITE_TOKEN \
  ghcr.io/efschu/lmcache-manylinux-builder:latest \
  /io

# Wheel will be in ./wheelhouse/
ls -la wheelhouse/
```

### Install

```bash
pip install lmcache-*.whl --force-reinstall --no-deps
```

### Usage with vLLM

```bash
# Start LMCache server (CPU-only, no GPU memory!)
CUDA_VISIBLE_DEVICES="" lmcache server \
  --max-workers 1 \
  --max-gpu-workers 1 \
  --max-cpu-workers 1 \
  --chunk-size 1600 \
  --l1-size-gb 10 \
  --eviction-policy LRU \
  --port 6555 \
  --l2-adapter '{"type":"fs","base_path":"/kv-cache", "max_capacity_gb": 250}' &

# Start vLLM with LMCache connector
vllm serve Qwen/Qwen3.6-27B \
  --kv-transfer-config '{
    "kv_connector": "LMCacheMPConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "lmcache.mp.host": "tcp://localhost",
      "lmcache.mp.port": 6555,
      "lmcache.mp.mp_transfer_mode": "auto"
    }
  }'
```

## Testing

```bash
# Run unit tests
pytest tests/

# Run specific multi-group tests
pytest tests/v1/ -k "multi_group or engine_driven"
```

## Commit History

### Commit: `11a440d` - Add multi-group KV-cache design doc for engine-driven lmcache

This single commit contains the full implementation:

1. **`lmcache/v1/multiprocess/custom_types.py`**
   - Added `GroupLayoutInfo` msgspec.Struct for per-group layout metadata
   - Extended `RegisterEngineDrivenContextPayload` with `group_layouts` field

2. **`lmcache/v1/multiprocess/transfer_context/base.py`**
   - Added `slice_kv_caches_for_group()` helper function
   - Added `gather_paged_kv_multi_group_to_cpu()` for multi-group gather
   - Added `scatter_cpu_multi_group_to_paged_kv()` for multi-group scatter

3. **`lmcache/v1/multiprocess/transfer_context/worker_transfer.py`**
   - Extended `EngineDrivenTransferContext` with multi-group support
   - Added `_serialize_multi_group_chunks()` and `_deserialize_multi_group_chunks()`
   - Modified `register()`, `submit_store()`, `submit_retrieve()` for multi-group

4. **`lmcache/v1/multiprocess/modules/engine_driven_transfer.py`**
   - Extended `register_kv_cache_engine_driven_context()` for multi-group registration
   - Added `_commit_store_multi_group()` server-side storage
   - Added `_prepare_retrieve_multi_group()` server-side retrieval

5. **`lmcache/v1/multiprocess/modules/lookup.py`**
   - Fixed multi-group prefetch handling: await all group handles before `finish_read_prefetched`
   - Added `multi_group_handles` field to `_PrefetchJob` dataclass
   - Modified `end_session()` to call `finish_read_prefetched` for all groups

6. **`lmcache/v1/multiprocess/group_view.py`**
   - Added `EngineGroupInfo` msgspec.Struct for engine-side group metadata

7. **`lmcache/v1/protocols/engine.py`**
   - Extended `PrepareRetrieveResponse` with `cpu_data: bytes` field

## Design Document

See [Lmcache_engine_driven_multigroup.md](./Lmcache_engine_driven_multigroup.md) for the full design document including:
- Detailed data flow analysis
- Protocol specifications
- API contracts
- Migration guide

## License

Apache 2.0 (same as upstream LMCache)
