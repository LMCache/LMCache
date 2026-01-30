# LMCache + vLLM: Tiered Storage (CPU Hot + Disk Warm) for Single Node

## 1. Introduction

**Target workload**
- Production single-node deployments
- Large working sets exceeding GPU memory
- Need for both speed and persistence
- **Best of both worlds: fast local hits + durable storage**

**LMCache mode**
- **Storage Mode**
- Single node
- Two-tier storage: CPU (hot) + Disk (warm)

This recipe demonstrates **tiered storage**, the most common production configuration for LMCache. It combines:

1. **CPU RAM tier** (hot) - Fast access for frequently used KV
2. **Local disk tier** (warm) - Persistent storage for larger working sets
3. **Automatic tiering** - LMCache manages movement between tiers

> **Why tiering?** Pure CPU cache is fast but limited by memory. Pure disk is large but slower. Tiering gives you both: frequently accessed KV stays in fast CPU memory, while the full working set persists to disk.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                     vLLM Instance                            │
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Request   │───▶│  GPU Cache  │───▶│   Miss?     │     │
│  │             │    │  (Limited)  │    │             │     │
│  └─────────────┘    └─────────────┘    └──────┬──────┘     │
│                                                │            │
│                     ┌──────────────────────────┘            │
│                     │                                       │
│           ┌─────────▼──────────┐    ┌─────────────┐         │
│           │   CPU Hot Cache    │    │   Miss?     │         │
│           │  (Fast, volatile)  │◀───┤             │         │
│           │  max_local_cpu: 48 │    └──────┬──────┘         │
│           └─────────┬──────────┘           │                │
│                     │                      │                │
│                     │            ┌─────────▼──────────┐     │
│                     │            │   Disk Warm Cache  │     │
│                     └───────────▶│ (Slower, persistent│     │
│                                  │  max_local_disk: 200│    │
│                                  └─────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Expected outcome**
- Hot data: Retrieved from CPU cache (~10-20ms)
- Warm data: Retrieved from disk (~50-100ms)
- Cold data: Computed from scratch (~500-800ms)
- Full working set persists across restarts

## 2. When to Use Tiered Storage

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Working set > GPU memory, < CPU memory | **CPU only** (R-001) | Simpler, no disk needed |
| Working set > GPU + CPU memory | **Tiered** (this recipe) | Disk extends capacity |
| Need persistence across restarts | **Tiered or Disk** | Disk provides durability |
| Maximum speed, no persistence needed | **CPU only** | No disk I/O overhead |
| Very large working set (TB+) | **Tiered + Remote** (R-030) | Add Redis/S3 for scale |

**Tiering vs Single Tier:**

| Configuration | Speed | Capacity | Persistence |
|--------------|-------|----------|-------------|
| CPU only | ⭐⭐⭐ Fast | ⭐ Limited | ❌ No |
| Disk only | ⭐ Slow | ⭐⭐ Large | ✅ Yes |
| **Tiered** | ⭐⭐ Fast | ⭐⭐ Large | ✅ Yes |

## 3. Installing vLLM + LMCache

```bash
# Install LMCache
pip install lmcache

# Install vLLM
pip install vllm
```

## 4. LMCache Configuration

Create `recipes/vllm_tiered_storage.yaml`:

```yaml
chunk_size: 256

# Tier 1: CPU Hot Cache
local_cpu: true
max_local_cpu_size: 48  # GB - size for ~70% of working set

# Tier 2: Disk Warm Cache
local_disk: true
local_disk_path: "/var/lib/lmcache/kv_cache"
max_local_disk_size: 200  # GB - size for full working set

# Performance tuning
use_layerwise: false      # Can enable after stability verified
save_unfull_chunk: true   # Important for partial chunk caching

# Eviction policy (applied to both tiers)
# LRU is default - recently used stays in CPU, older moves to disk
```

> **⚠️ Critical Sizing Guidance**
> 
> **CPU tier:** Should hold your "hot" working set (frequently accessed data)
> - Recommended: 1.5x GPU KV cache budget, or ~70% of expected working set
> 
> **Disk tier:** Should hold your full working set
> - Recommended: 2-3x expected working set size
> 
> **Example:** If you expect 100GB of unique KV data:
> - `max_local_cpu_size: 70` (hot 70%)
> - `max_local_disk_size: 300` (full set + headroom)

### Setup disk directory

```bash
# Create cache directory
sudo mkdir -p /var/lib/lmcache/kv_cache
sudo chown $(whoami):$(whoami) /var/lib/lmcache/kv_cache
chmod 755 /var/lib/lmcache/kv_cache

# Verify
ls -ld /var/lib/lmcache/kv_cache
```

## 5. Launching vLLM with Tiered Storage

```bash
export PYTHONHASHSEED=0
export LMCACHE_CONFIG_FILE=recipes/vllm_tiered_storage.yaml

CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
--max-model-len 8192 \
--gpu-memory-utilization 0.85 \
--port 8000 \
--no-enable-prefix-caching \
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

## 6. Startup Validation

Expected LMCache logs:
```
LMCache INFO: Loading LMCache config file recipes/vllm_tiered_storage.yaml
LMCache INFO: Creating LMCacheEngine with config:
  {
    'chunk_size': 256,
    'local_cpu': True,
    'max_local_cpu_size': 48.0,
    'local_disk': True,
    'local_disk_path': '/var/lib/lmcache/kv_cache',
    'max_local_disk_size': 200.0,
    ...
  }
LMCache INFO: Initializing LocalCPUBackend
LMCache INFO: Initializing LocalDiskBackend at /var/lib/lmcache/kv_cache
```

Verify both tiers are active:
```bash
# CPU memory should be allocated
ps aux | grep vllm | grep -i lmcache

# Disk directory should exist
ls -la /var/lib/lmcache/kv_cache/
```

## 7. Inference and Tiering Validation

### 7.1 Populate cache with mixed workload

```bash
# Request 1: Populate cache (will store to CPU and disk)
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": "System: You are a helpful assistant.",
    "max_tokens": 100
  }'

# Request 2-10: More unique prompts to fill cache
for i in {1..10}; do
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Qwen/Qwen3-4B-Instruct-2507\",
    \"prompt\": \"Context $i: This is a unique prompt to populate the cache with varied data.\",
    \"max_tokens\": 50
  }"
done
```

### 7.2 Verify tiering behavior

**Check CPU cache usage:**
```bash
# Watch LMCache CPU memory
watch -n 2 'ps -o pid,rss,comm -p $(pgrep -f "vllm")'
```

**Check disk cache usage:**
```bash
# Check disk cache size
du -sh /var/lib/lmcache/kv_cache/

# List chunk files
find /var/lib/lmcache/kv_cache/ -type f | wc -l
```

### 7.3 Demonstrate tiered retrieval

**Hot hit (from CPU):**
```bash
# Request same prompt immediately (should hit CPU)
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": "System: You are a helpful assistant.",
    "max_tokens": 100
  }'
```

Expected log:
```
LMCache INFO: Retrieved 256 tokens from CPU cache. cost 12.5 ms
```

**Warm hit (from disk after CPU eviction):**
```bash
# Generate enough new requests to evict from CPU
# Then request old prompt again (should hit disk)
```

Expected log:
```
LMCache INFO: Retrieved 256 tokens from disk cache. cost 65.3 ms
```

## 8. Benchmarking

### 8.1 Baseline (no LMCache)

```bash
vllm serve Qwen/Qwen3-4B-Instruct \
  --max-model-len 8192 \
  --port 8000 \
  --no-enable-prefix-caching

vllm bench serve --port 8000 \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 6144 \
  --num-prompts 100
```

### 8.2 With tiered storage

```bash
# With tiered config
vllm serve Qwen/Qwen3-4B-Instruct \
  --max-model-len 8192 \
  --port 8000 \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

vllm bench serve --port 8000 \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 6144 \
  --num-prompts 100
```

### 8.3 Expected results

| Metric | No Cache | Tiered Storage | Improvement |
|--------|----------|----------------|-------------|
| Cold TTFT | ~600ms | ~600ms | - |
| Hot hit TTFT | ~600ms | ~120ms | **~80% faster** |
| Warm hit TTFT | ~600ms | ~200ms | **~65% faster** |
| Cache capacity | GPU only | GPU + CPU + Disk | **10-100x larger** |
| Persistence | No | Yes | **Survives restart** |

## 9. Tiering Performance Tuning

### 9.1 CPU tier sizing

```yaml
# Conservative (more disk hits, lower memory usage)
max_local_cpu_size: 24

# Balanced (recommended)
max_local_cpu_size: 48

# Aggressive (more CPU hits, higher memory usage)
max_local_cpu_size: 96
```

### 9.2 Disk tier optimization

```yaml
# Fast NVMe SSD
local_disk_path: "/nvme/lmcache/kv_cache"

# Enable GDS if available (see R-008)
# local_disk: true with GDS backend
```

### 9.3 Eviction policy

LMCache uses LRU (Least Recently Used) by default:
- Hot data stays in CPU
- Cold data moves to disk
- Evicted from disk when `max_local_disk_size` reached

### 9.4 Monitor tier hit rates

```bash
# Watch for logs showing retrieval source
tail -f vllm.log | grep -E "Retrieved.*from|Stored.*to"

# Expected patterns:
# "Retrieved X tokens from CPU cache" - Hot hit
# "Retrieved X tokens from disk cache" - Warm hit
# "Stored X tokens" - New data cached
```

## 10. Performance Tips

| Optimization | Configuration | Impact |
|--------------|---------------|--------|
| Larger CPU tier | Increase `max_local_cpu_size` | More hot hits |
| Faster disk | Use NVMe SSD | Faster warm hits |
| Chunk size | 256-512 | Balance granularity vs overhead |
| Layerwise | Enable after stable | Hide transfer latency |
| Monitoring | Watch hit rates | Tune tier sizes |

## 11. Troubleshooting / Common Pitfalls

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Only disk hits | CPU tier too small | Increase `max_local_cpu_size` |
| CPU OOM | CPU tier too large | Decrease `max_local_cpu_size` |
| Disk full | `max_local_disk_size` too large | Reduce or clean cache |
| Slow warm hits | Slow disk (HDD vs SSD) | Move to NVMe |
| No persistence | Disk path wrong | Check `local_disk_path` |
| Cache not surviving restart | Wrong disk path | Verify path exists |

### Clear cache if needed

```bash
# Clear both tiers
rm -rf /var/lib/lmcache/kv_cache/*

# Or just clear disk (CPU clears on restart)
rm -rf /var/lib/lmcache/kv_cache/*
```

### Debug tiering

```bash
# Check what's in CPU vs disk
# CPU: Watch process RSS
ps -o pid,rss,vsz,comm -p $(pgrep -f vllm)

# Disk: Check directory size
du -sh /var/lib/lmcache/kv_cache/
find /var/lib/lmcache/kv_cache/ -type f | wc -l
```

## 12. Extending to 3-Tier (CPU + Disk + Remote)

For even larger working sets, add a remote tier:

```yaml
# 3-tier configuration
chunk_size: 256

# Tier 1: CPU
local_cpu: true
max_local_cpu_size: 48

# Tier 2: Disk
local_disk: true
local_disk_path: "/var/lib/lmcache/kv_cache"
max_local_disk_size: 200

# Tier 3: Remote (Redis)
remote_url: "redis://localhost:6379"
```

See R-030 (Multi-node shared cache) for full 3-tier configuration.

## 13. Production Checklist

- [ ] CPU tier sized to ~70% of working set
- [ ] Disk tier sized to 2-3x working set
- [ ] Disk path on fast NVMe SSD
- [ ] Permissions set on disk directory
- [ ] Monitoring for hit rates
- [ ] Log rotation for cache logs
- [ ] Backup strategy for disk cache (if needed)

## 14. Additional Resources
- CPU hot cache only: `recipes/dense_instruct_cpu_hot_cache.md` (R-001)
- Disk persistence only: `recipes/vllm_disk_persistence.md` (R-007)
- Multi-node with remote: `recipes/vllm_multi_instance_sharing.md` (R-018)
- Full 3-tier: `recipes/vllm_three_tier_storage.md` (R-030)
