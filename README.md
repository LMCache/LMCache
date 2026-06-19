# LMCache Engine-Driven Multi-Group Fork

**Repository:** https://github.com/efschu/LMCache  
**Branch:** `dev`  
**Latest Release:** [v0.4.8rc2-dev15](https://github.com/efschu/LMCache/releases/tag/v0.4.8rc2-dev15)

---

## Downloads

| Asset | SHA256 |
|-------|--------|
| `lmcache-0.4.8rc2.dev15-cp312-cp312-linux_x86_64.whl` | `ebbf8be0...` |

**Quick Install:**
```bash
curl -L -O https://github.com/efschu/LMCache/releases/download/v0.4.8rc2-dev15/lmcache-0.4.8rc2.dev15-cp312-cp312-linux_x86_64.whl
pip install lmcache-0.4.8rc2.dev15-cp312-cp312-linux_x86_64.whl --force-reinstall --no-deps
```

---

## Overview

This fork implements **engine-driven KV cache transfer** for **hybrid multi-group models** (e.g., Qwen3.6-27B with GDN/Mamba + Attention groups) in LMCache v1.

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

---

## Quick Start

### Download Wheel

```bash
# Download from GitHub Releases or build from source
# Latest build: wheelhouse/lmcache-0.4.8rc2.dev15-cp312-cp312-linux_x86_64.whl
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

**Important:** Use `CUDA_VISIBLE_DEVICES=""` (not `NVIDIA_VISIBLE_DEVICES=""`) to prevent the server from allocating GPU memory.

---

## Documentation
| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file - Overview and quick start |
| [BUILD_GUIDE.md](docs/BUILD_GUIDE.md) | **Step-by-step build instructions** |
| [BUILD.md](docs/BUILD.md) | Technical build details and CI/CD |
| [Lmcache_engine_driven_multigroup.md](Lmcache_engine_driven_multigroup.md) | Full design document with protocol specs |

See [BUILD.md](docs/BUILD.md) for detailed build instructions.

### Quick Build

```bash
# Clone repository
git clone https://github.com/efschu/LMCache.git
cd LMCache

# Build wheel with Docker (GPU required)
docker run --rm \
    --gpus all \
    --security-opt apparmor=unconfined \
    -v $(pwd):/lm \
    -v $(pwd)/wheelhouse:/whl \
    ghcr.io/efschu/lmcache-manylinux-builder-gpu \
    bash -c '
        export TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"
        export ENABLE_CXX11_ABI=1
        cd /lm
        /opt/python/cp312-cp312/bin/pip wheel . --no-deps -w /whl
    '

# Install
pip install wheelhouse/lmcache-*.whl --force-reinstall --no-deps
```

---

## Commit History

### Commit: `26d830d` - Add multi-group KV-cache design document for engine-driven lmcache

This commit contains the full implementation:

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

---

## Testing

```bash
# Run unit tests
pytest tests/

# Run specific multi-group tests
pytest tests/v1/ -k "multi_group or engine_driven"
```

---

## Design Document

See [Lmcache_engine_driven_multigroup.md](./Lmcache_engine_driven_multigroup.md) for the full design document including:

- **Detailed data flow analysis** - How data flows between vLLM and LMCache
- **Protocol specifications** - Wire format for engine-driven transfer
- **API contracts** - Function signatures and expected behaviors
- **Migration guide** - How to migrate from old transfer mode

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          vLLM Worker                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │ Paged KV    │───▶│ gather_paged │───▶│ EngineDrivenTransfer  │  │
│  │ Cache       │    │ _kv_to_cpu() │    │ .submit_store()       │  │
│  └─────────────┘    └──────────────┘    └───────────┬───────────┘  │
│                                                      │              │
│  ┌─────────────┐    ┌──────────────┐               │              │
│  │ Paged KV    │◀───│ scatter_cpu  │◀──────────────┘              │
│  │ Cache       │    │ _to_paged_kv │    ┌───────────────────────┐  │
│  └─────────────┘    └──────────────┘    │ EngineDrivenTransfer  │  │
│                                         │ .submit_retrieve()    │  │
│                                         └───────────┬───────────┘  │
└─────────────────────────────────────────────────────┼─────────────┘
                                                      │ IPC
                                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       LMCache Server (CPU-only)                      │
│  ┌──────────────┐    ┌─────────────────┐    ┌───────────────────┐ │
│  │ Lookup       │───▶│ finish_read     │───▶│ ObjectGroupStore  │ │
│  │ Manager      │    │ _prefetched()   │    │ (all groups)      │ │
│  └──────────────┘    └─────────────────┘    └───────────────────┘ │
│         │                                                       │   │
│         │    ┌─────────────────┐                               │   │
│         └───▶│ PrefetchJob     │                               │   │
│              │ (handles for    │                               │   │
│              │  groups 0..N)   │                               │   │
│              └─────────────────┘                               │   │
│                                                                     │
│  ┌──────────────┐    ┌─────────────────┐                          │
│  │ L1 Memory    │◀──▶│ LRU Eviction    │                          │
│  │ (RAM)        │    │ Policy          │                          │
│  └──────────────┘    └─────────────────┘                          │
│         │                                                             │
│         ▼                                                             │
│  ┌──────────────┐                                                     │
│  │ L2 Disk      │                                                     │
│  │ (KV-Cache)   │                                                     │
│  └──────────────┘                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## GPU Memory Comparison


Savings: **~668 MB per GPU** for LMCache server overhead elimination.

---

## License

Apache 2.0 (same as upstream LMCache)

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Build and test: `pip wheel . --no-deps -w wheelhouse/`
5. Submit a pull request

---

## References

- [LMCache upstream repository](https://github.com/LMCache/LMCache)
- [vLLM KV Transfer documentation](https://docs.vllm.ai/en/latest/features/kv_transfer.html)
- [PyTorch CUDA extensions](https://pytorch.org/docs/stable/cpp_extension.html)
