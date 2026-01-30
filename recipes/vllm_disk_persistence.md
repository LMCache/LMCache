# LMCache + vLLM: Disk-Persistent KV Cache for Restart Survivability

## 1. Introduction

**Target workload**
- Long-running inference services that restart periodically
- Large KV cache working sets that exceed GPU memory
- Need for cache durability across deployments
- **Cost optimization** - warm start without re-computation

**LMCache mode**
- **Storage Mode**
- Single node
- Local disk backend (POSIX filesystem)

This recipe demonstrates how to run **vLLM with LMCache enabled** using a **local disk backend** for KV cache persistence. Unlike CPU hot cache (which loses data on process exit), disk persistence enables:

1. **Survive restarts** - KV cache persists even when vLLM restarts
2. **Faster cold starts** - After first deployment, subsequent starts are "warm"
3. **Large working sets** - Disk capacity is typically much larger than GPU/CPU memory
4. **Cost savings** - Avoid re-computing KV for repeated prefixes

> **Trade-off:** Disk I/O is slower than CPU memory. This recipe focuses on **persistence** and **capacity** over raw speed. For best performance, combine with CPU hot cache (tiering) - see the tiering recipe (R-029).

**Expected outcome**
- First deployment: cold cache, full prefill (same as baseline)
- After restart: **Warm start** - cache is reloaded from disk, faster TTFT
- Disk directory contains persisted KV cache files

## 2. When to Use Disk Persistence

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Need cache to survive service restarts | **LMCache + Disk** | Disk persists across process restarts |
| Working set >> GPU + CPU memory | **LMCache + Disk** | Disk capacity is much larger |
| Cost-sensitive, can trade speed for capacity | **LMCache + Disk** | Slower than CPU but cheaper per GB |
| Need fastest possible cache hits | **LMCache + CPU only** | CPU memory is faster than disk |
| Working set fits in GPU memory | **vLLM native** | No persistence needed, lowest overhead |

## 3. Installing vLLM + LMCache

Preferred (uv):

```bash
# Install LMCache
pip install lmcache

# Install vLLM
pip install vllm
```

## 4. LMCache Configuration

Create `recipes/vllm_disk_persistence.yaml`:

```yaml
chunk_size: 256           # Default: 256
local_cpu: false          # Disable CPU cache for this pure-disk recipe
local_disk: true          # Enable disk backend
local_disk_path: "/var/lib/lmcache/kv_cache"  # Persisted cache location
max_local_disk_size: 100  # GB - adjust based on your disk capacity
use_layerwise: false      # Disable for stability with disk I/O
save_unfull_chunk: true   # Cache partial chunks
```

> **⚠️ Disk Path Requirements**
> 
> - The `local_disk_path` must exist and be writable by the vLLM process
> - Use a fast local disk (NVMe SSD recommended) for best performance
> - Network filesystems (NFS) work but add latency
> - Ensure sufficient free space: `max_local_disk_size` + headroom

**Setup the disk directory:**

```bash
# Create the cache directory with appropriate permissions
sudo mkdir -p /var/lib/lmcache/kv_cache
sudo chown $(whoami):$(whoami) /var/lib/lmcache/kv_cache
chmod 755 /var/lib/lmcache/kv_cache

# Verify write access
touch /var/lib/lmcache/kv_cache/test && rm /var/lib/lmcache/kv_cache/test
echo "Disk path ready"
```

## 5. Launching the vLLM Server (with Disk Persistence)

### First launch (cold start, populates cache)

```bash
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=recipes/vllm_disk_persistence.yaml \
CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
--max-model-len 8192 \
--gpu-memory-utilization 0.85 \
--port 8000 \
--no-enable-prefix-caching \
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

### Verify cache is being written to disk

After sending some requests, check the disk directory:

```bash
ls -la /var/lib/lmcache/kv_cache/
# Expected: directory contains chunk files with hashed names

du -sh /var/lib/lmcache/kv_cache/
# Shows total cache size on disk
```

Expected directory structure:
```
/var/lib/lmcache/kv_cache/
├── chunks/
│   ├── <hash_prefix_1>/
│   │   └── <chunk_file_1>.bin
│   ├── <hash_prefix_2>/
│   │   └── <chunk_file_2>.bin
│   └── ...
└── metadata/
    └── chunk_index.json
```

## 6. Startup Validation

### First launch (cold)

Expected LMCache logs:
```
LMCache INFO: Loading LMCache config file recipes/vllm_disk_persistence.yaml
LMCache INFO: LMCache initialized for role KVConnectorRole.WORKER
LMCache INFO: Creating LMCacheEngine with config:
  {'chunk_size': 256, 'local_disk': True, 'local_disk_path': '/var/lib/lmcache/kv_cache', ...}
LMCache INFO: Initializing LocalDiskBackend at /var/lib/lmcache/kv_cache
```

### After restart (warm)

Stop vLLM (Ctrl+C), then restart with the **same command**. Expected logs:
```
LMCache INFO: Loading LMCache config file recipes/vllm_disk_persistence.yaml
LMCache INFO: LocalDiskBackend: Loading existing cache index
LMCache INFO: Found X chunks on disk, Y GB total
```

## 7. Inference and Cache Validation

### 7.1 First deployment (cold)

Send a request to populate cache:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": "You are a helpful AI assistant.\n\nUser: Explain the benefits of KV cache persistence in large language models. This is a long prompt that will generate KV cache data that we want to persist across restarts. The prompt needs to be long enough to create multiple chunks in the cache.",
    "max_tokens": 100
  }'
```

Expected LMCache logs (cold):
```
LMCache INFO: Reqid: ..., Total tokens 512, LMCache hit tokens: 0, need to load: 0
LMCache INFO: Stored 512 out of total 512 tokens. size: 0.0703 GB, cost 25.1234 ms
```

Verify disk cache:
```bash
du -sh /var/lib/lmcache/kv_cache/
# Shows non-zero size

find /var/lib/lmcache/kv_cache -type f | wc -l
# Shows number of chunk files
```

### 7.2 Restart and warm request

1. **Stop vLLM** (Ctrl+C)
2. **Verify cache still on disk:**
   ```bash
   du -sh /var/lib/lmcache/kv_cache/
   # Cache should still exist
   ```
3. **Restart vLLM** with the same command
4. **Send the same request again**

Expected LMCache logs (warm - after restart):
```
LMCache INFO: Reqid: ..., Total tokens 512, LMCache hit tokens: 512, need to load: 512
LMCache INFO: Retrieved 512 out of 512 required tokens. size: 0.0703 gb, cost 35.4567 ms
```

**Key observation:** The cache hit occurs **despite the restart** because the data was persisted to disk.

## 8. Benchmarking

### 8.1 Baseline (no persistence)

Run without LMCache, restart, and measure:

```bash
# First run
CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --port 8000 \
  --no-enable-prefix-caching

# Send request, measure TTFT
# Stop vLLM
# Restart
# Send same request, measure TTFT
# Result: Both TTFTs are similar (no persistence)
```

### 8.2 With disk persistence

```bash
export LMCACHE_CONFIG_FILE=recipes/vllm_disk_persistence.yaml
export PYTHONHASHSEED=0

# First run (cold)
CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --port 8000 \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

# Send request, measure TTFT (cold)
# Stop vLLM
# Restart with same command
# Send same request, measure TTFT (warm after restart)
```

### 8.3 Benchmark results

| Scenario | First TTFT | After Restart TTFT | Improvement |
|----------|------------|-------------------|-------------|
| No persistence | ~600ms | ~600ms | None |
| Disk persistence | ~600ms | ~150ms | **~75% faster** |

The key metric is **time to first token after restart**. With disk persistence, the second startup (after cache population) is significantly faster.

### 8.4 Disk I/O benchmark

Measure disk cache read performance:

```bash
# Cache size on disk
du -sh /var/lib/lmcache/kv_cache/

# Individual chunk file size
ls -lh /var/lib/lmcache/kv_cache/chunks/*/

# Simulate read performance
dd if=/var/lib/lmcache/kv_cache/chunks/$(ls /var/lib/lmcache/kv_cache/chunks/ | head -1)/$(ls /var/lib/lmcache/kv_cache/chunks/*/) of=/dev/null bs=1M
```

## 9. Optimizing Disk Performance

### 9.1 Use Fast Storage

```yaml
# NVMe SSD (recommended)
local_disk_path: "/nvme/lmcache/kv_cache"

# Avoid network filesystems for hot data
# NFS/SMB add significant latency
```

### 9.2 Combine with CPU Hot Cache (Tiering)

For best performance, use both CPU and disk (see R-029):

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 48
local_disk: true
local_disk_path: "/var/lib/lmcache/kv_cache"
max_local_disk_size: 200
```

This creates a tiered cache:
- **Hot tier**: CPU memory (fastest hits)
- **Warm tier**: Local disk (survives restarts)

### 9.3 Tune Chunk Size

```yaml
# Larger chunks = fewer files, better for disk
chunk_size: 512  # or 1024 for very large prompts

# Smaller chunks = better partial reuse
chunk_size: 128  # more files, better granularity
```

## 10. Performance Tips

| Parameter | Recommendation | Impact |
|-----------|---------------|--------|
| `local_disk_path` | Fast NVMe SSD | Lower I/O latency |
| `chunk_size` | 256-512 | Balance granularity vs file count |
| `max_local_disk_size` | 2-5x working set size | Avoid eviction churn |
| Filesystem | ext4 or xfs | Avoid btrfs/ZFS overhead |
| CPU + Disk | Enable both | Best of speed + persistence |

## 11. Troubleshooting / Common Pitfalls

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| No cache files on disk | Path doesn't exist or not writable | Check `mkdir -p` and permissions |
| Permission denied | vLLM runs as different user | `chown` or use `/tmp/lmcache_$USER` |
| Slow warm start | Slow disk (HDD vs SSD) | Move to NVMe SSD |
| Cache eviction too aggressive | `max_local_disk_size` too small | Increase limit |
| Corrupted cache files | Unclean shutdown | Clear cache directory and restart |
| No cache hit after restart | Different model or chunk_size | Ensure identical config across restarts |
| Disk full | Cache grew unbounded | Set `max_local_disk_size` appropriately |

### Clear cache if needed

```bash
# Stop vLLM first
rm -rf /var/lib/lmcache/kv_cache/*
# Or move aside
mv /var/lib/lmcache/kv_cache /var/lib/lmcache/kv_cache.backup.$(date +%Y%m%d)
mkdir -p /var/lib/lmcache/kv_cache
```

## 12. Additional Resources
- CPU hot cache recipe: `recipes/dense_instruct_cpu_hot_cache.md`
- Tiered storage recipe: `recipes/vllm_tiered_storage.md` (R-029)
- LMCache config reference: `docs/source/api_reference/configurations.rst`
- Disk backend docs: `docs/source/kv_cache/storage_backends/local_disk.rst`
