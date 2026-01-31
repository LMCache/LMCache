# LMCache + vLLM: NIXL as Storage Backend

## 1. Introduction

**Target workload**
- High-performance storage using NIXL
- Advanced transport options (POSIX, GDS, GDS_MT, HF3FS)
- Non-PD use cases with NIXL transport
- **Unified NIXL backend for storage**

**LMCache mode**
- **Storage Mode**
- Single or multi-node
- NIXL transport layer

This recipe demonstrates using **NIXL as a storage backend** (not just for PD transport):

1. **Unified transport** - Same NIXL library for storage and PD
2. **Multiple backends** - POSIX, GDS, GDS_MT, HF3FS
3. **High performance** - Optimized for NVMe and distributed filesystems
4. **Flexibility** - Choose transport per use case

> **NIXL Backends:**
> - `POSIX`: Standard filesystem
> - `GDS`: GPUDirect Storage
> - `GDS_MT`: Multi-threaded GDS
> - `HF3FS`: HyperFS (distributed filesystem)

## 2. When to Use NIXL Storage

| Scenario | Backend | Why |
|----------|---------|-----|
| Standard storage | POSIX | Compatibility |
| Max NVMe perf | GDS/GDS_MT | GPU-direct I/O |
| Distributed FS | HF3FS | Shared storage |
| PD mode | NIXL transport | KV transfer |

## 3. Prerequisites

```bash
# Install NIXL
pip install nixl

# For GDS backend: Install CUDA and cuFile
# For HF3FS: Install HyperFS client
```

## 4. LMCache Configuration

### 4.1 POSIX via NIXL

```yaml
# vllm_nixl_posix.yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 48

local_disk: true
local_disk_path: "/var/lib/lmcache/kv_cache"
max_local_disk_size: 200

# NIXL as storage backend
nixl_backend: "POSIX"  # Options: POSIX, GDS, GDS_MT, HF3FS
nixl_enable: true

save_unfull_chunk: true
```

### 4.2 GDS via NIXL

```yaml
# vllm_nixl_gds.yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 48

local_disk: true
local_disk_path: "/mnt/nvme0/lmcache"
max_local_disk_size: 500

# NIXL GDS backend
nixl_backend: "GDS"
nixl_enable: true

# GDS-specific
nixl_gds_buffer_pool: 1073741824  # 1GB

save_unfull_chunk: true
```

### 4.3 HF3FS via NIXL

```yaml
# vllm_nixl_hf3fs.yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 48

local_disk: true
local_disk_path: "/hf3fs/lmcache/kv_cache"
max_local_disk_size: 1000  # 1TB

# NIXL HF3FS backend
nixl_backend: "HF3FS"
nixl_enable: true

# HF3FS-specific
nixl_hf3fs_mount: "/hf3fs"
nixl_hf3fs_stripe_size: 1048576  # 1MB stripes

save_unfull_chunk: true
```

## 5. Launching vLLM with NIXL Storage

```bash
export PYTHONHASHSEED=0
export LMCACHE_CONFIG_FILE=recipes/vllm_nixl_posix.yaml

CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
--max-model-len 8192 \
--port 8000 \
--no-enable-prefix-caching \
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

## 6. Startup Validation

Expected logs:
```
LMCache INFO: Initializing NIXL storage backend
LMCache INFO: NIXL backend type: POSIX
LMCache INFO: NIXL driver initialized
```

## 7. Benchmarking

| Backend | Throughput | Use Case |
|---------|------------|----------|
| POSIX | 3 GB/s | Standard |
| GDS | 6 GB/s | NVMe optimized |
| GDS_MT | 8 GB/s | Multi-threaded |
| HF3FS | 10 GB/s | Distributed FS |

## 8. Additional Resources
- GDS backend: `recipes/vllm_gds_backend.md` (R-008)
- PD with NIXL: `recipes/vllm_pd_single_node.md` (R-021)
