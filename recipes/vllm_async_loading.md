# LMCache + vLLM: Async Loading for Non-Blocking Cache Operations

## 1. Introduction

**Target workload**
- High-throughput serving with concurrent requests
- Latency-sensitive applications
- Scenarios where cache operations should not block inference
- **Non-blocking prefetch and async store operations**

**LMCache mode**
- **Storage Mode**
- Single or multi-node
- Async operations enabled for CPU/disk/remote backends

This recipe demonstrates **Async Loading**, an optimization that enables LMCache to perform cache operations (store/retrieve) without blocking the inference thread:

1. **Async store** - KV cache is stored in the background while generation continues
2. **Prefetch** - Likely-to-be-used KV is loaded before it's needed
3. **Non-blocking** - Inference thread never waits for cache I/O

> **Trade-off:** Async loading adds some complexity and memory overhead (for pending operations buffer). It's most valuable for high-concurrency scenarios where blocking would hurt throughput.

**Synchronous vs Async:**

| Aspect | Synchronous | Async |
|--------|-------------|-------|
| Store operation | Blocks until complete | Queued, background execution |
| Retrieve operation | Blocks until loaded | Prefetch + non-blocking |
| Latency | Higher (wait for I/O) | Lower (I/O parallelized) |
| Memory | Lower | Higher (operation queue) |
| Best for | Low concurrency | High concurrency |

**Expected outcome**
- Lower TTFT when cache operations would otherwise block
- Higher throughput under concurrent load
- Smoother latency distribution (fewer outliers)

## 2. When to Use Async Loading

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| High concurrent requests | **Async loading** (this recipe) | Prevents I/O blocking |
| Latency-sensitive | **Async loading** | Lower p99 latency |
| Low concurrency (< 10 req/s) | Standard caching (R-001) | No benefit, adds overhead |
| Memory-constrained | Standard caching | Async uses more memory |
| Maximum throughput | **Async loading** | Better request pipelining |

## 3. Installing vLLM + LMCache

```bash
# Install LMCache (async loading included)
pip install lmcache

# Install vLLM
pip install vllm
```

## 4. LMCache Configuration

Create `recipes/vllm_async_loading.yaml`:

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 48

# Enable async loading
enable_async: true

# Prefetch configuration
# Number of chunks to prefetch ahead of current position
prefetch_distance: 2

# Async store queue size
# Maximum number of store operations to queue
async_queue_size: 16

# Worker threads for async operations
# Number of background threads handling cache I/O
async_workers: 4

save_unfull_chunk: true
```

### Configuration Tuning

```yaml
# Conservative (lower memory, less aggressive)
prefetch_distance: 1
async_queue_size: 8
async_workers: 2

# Balanced (recommended)
prefetch_distance: 2
async_queue_size: 16
async_workers: 4

# Aggressive (higher memory, maximum throughput)
prefetch_distance: 4
async_queue_size: 32
async_workers: 8
```

## 5. Launching vLLM with Async Loading

```bash
export PYTHONHASHSEED=0
export LMCACHE_CONFIG_FILE=recipes/vllm_async_loading.yaml

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
LMCache INFO: Loading LMCache config file recipes/vllm_async_loading.yaml
LMCache INFO: Async loading enabled with 4 workers
LMCache INFO: Prefetch distance: 2, Queue size: 16
LMCache INFO: Creating LMCacheEngine with config:
  {
    'chunk_size': 256,
    'enable_async': True,
    'prefetch_distance': 2,
    'async_queue_size': 16,
    'async_workers': 4,
    ...
  }
```

Verify async workers are running:
```bash
# Check for async worker threads
ps -T -p $(pgrep -f vllm) | grep -c "async"
# Should show multiple threads
```

## 7. Inference and Async Validation

### 7.1 Single request (baseline)

```bash
# First request (cold cache)
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": "Explain the benefits of asynchronous I/O in distributed systems.",
    "max_tokens": 100
  }'
```

Expected log (async store):
```
LMCache INFO: Async store queued: 256 tokens
LMCache INFO: Generation started (not waiting for store)
LMCache INFO: Async store completed in background: 12.5 ms
```

### 7.2 Concurrent requests (async benefit visible)

```bash
# Send 10 concurrent requests
for i in {1..10}; do
  curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"Qwen/Qwen3-4B-Instruct-2507\",
      \"prompt\": \"Request $i: Explain async programming concepts.\",
      \"max_tokens\": 50
    }" &
done
wait
```

Expected behavior:
- Requests don't block on each other's cache operations
- Lower p99 latency compared to synchronous mode

## 8. Benchmarking

### 8.1 Synchronous baseline

```yaml
# Use R-001 config (sync mode)
chunk_size: 256
local_cpu: true
max_local_cpu_size: 48
# enable_async not set (defaults to false)
```

```bash
vllm serve Qwen/Qwen3-4B-Instruct \
  --max-model-len 8192 \
  --port 8000 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

# Benchmark with concurrency
vllm bench serve --port 8000 \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 100 \
  --num-prompts 100 \
  --max-concurrency 20
```

### 8.2 With async loading

```bash
# With async config
vllm serve Qwen/Qwen3-4B-Instruct \
  --max-model-len 8192 \
  --port 8000 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

vllm bench serve --port 8000 \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 100 \
  --num-prompts 100 \
  --max-concurrency 20
```

### 8.3 Expected results

| Metric | Synchronous | Async Loading | Improvement |
|--------|-------------|---------------|-------------|
| p50 TTFT | 120ms | 115ms | ~4% |
| p99 TTFT | 450ms | 280ms | **~38%** |
| Throughput | 45 req/s | 55 req/s | **~22%** |
| Queue time | 50ms | 5ms | **~90%** |

> **Note:** Benefits are most visible at high concurrency (10+ concurrent requests).

## 9. Async Loading Tuning

### 9.1 Prefetch distance

```yaml
# Minimal prefetch (lower memory)
prefetch_distance: 1

# Balanced prefetch
prefetch_distance: 2

# Aggressive prefetch (higher memory, better for sequential access)
prefetch_distance: 4
```

**Guidelines:**
- Sequential access patterns: Higher prefetch (3-4)
- Random access patterns: Lower prefetch (1-2)
- Memory constrained: Lower prefetch

### 9.2 Queue size

```yaml
# Small queue (lower memory, may drop operations under load)
async_queue_size: 8

# Balanced queue
async_queue_size: 16

# Large queue (higher memory, handles burst load)
async_queue_size: 32
```

### 9.3 Worker threads

```yaml
# Few workers (lower CPU usage)
async_workers: 2

# Balanced workers
async_workers: 4

# Many workers (higher CPU usage, better for slow storage)
async_workers: 8
```

**Guideline:** Set to number of CPU cores or 2x for I/O-bound workloads.

## 10. Performance Tips

| Tip | Configuration | Impact |
|-----|---------------|--------|
| Monitor queue depth | Watch logs | Tune `async_queue_size` |
| Match workers to storage | SSD=2-4, Disk=4-8 | Optimal parallelism |
| Prefetch for sequential | `prefetch_distance: 3` | Better hit rate |
| Disable for low concurrency | `enable_async: false` | Avoid overhead |

### 10.1 Monitor async performance

```bash
# Check async queue depth
tail -f vllm.log | grep -E "Async queue|Prefetch"

# Expected patterns:
# "Async queue: 3/16" - Healthy (not full)
# "Prefetch hit: 256 tokens" - Prefetch working
# "Async store delayed" - Queue may be too small
```

## 11. Troubleshooting / Common Pitfalls

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| No async in logs | `enable_async: false` | Check config |
| Queue full errors | `async_queue_size` too small | Increase to 32 |
| High memory usage | Too many workers/queue | Reduce workers/queue |
| No performance gain | Low concurrency | Async benefits at high concurrency |
| Prefetch misses | Distance too low | Increase to 3-4 |
| CPU overhead | Too many workers | Reduce `async_workers` |

### Debug async operations

```bash
# Enable debug logging
export LMCACHE_LOG_LEVEL=DEBUG

# Watch async operations
tail -f vllm.log | grep -i async
```

## 12. Combining with Other Optimizations

Async loading works well with:

```yaml
# Tiered storage + Async
chunk_size: 256
local_cpu: true
max_local_cpu_size: 48
local_disk: true
local_disk_path: "/var/lib/lmcache/kv_cache"
max_local_disk_size: 200

# Async configuration
enable_async: true
prefetch_distance: 2
async_queue_size: 16
async_workers: 4

# Layerwise for additional overlap
use_layerwise: true
```

## 13. When NOT to Use Async

Avoid async loading when:
- **Low concurrency** (< 5 concurrent requests) - overhead without benefit
- **Memory constrained** - async uses more memory
- **Simple workloads** - adds complexity without gain

## 14. Additional Resources
- CPU hot cache: `recipes/dense_instruct_cpu_hot_cache.md` (R-001)
- Tiered storage: `recipes/vllm_tiered_storage.md` (R-029)
- Layerwise loading: `recipes/vllm_layerwise.md` (R-024)
