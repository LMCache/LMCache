# LMCache + SGLang: CPU-Extended KV Cache for Large Working Sets

## 1. Introduction

**Target workload**
- Multi-turn chat
- Repeated system prompt
- Small repeated RAG-style context
- **Large working sets that may exceed GPU KV cache capacity**

**LMCache mode**
- **Storage Mode**
- Single node
- CPU hot cache only

This recipe demonstrates how to run **SGLang with LMCache enabled** on a single GPU using a **CPU pinned-memory hot cache**. LMCache extends SGLang's KV cache management to CPU memory, enabling:

1. **Larger working sets** - Cache more prefixes than GPU memory allows
2. **Cache persistence** - KV cache survives SGLang restarts
3. **Cross-instance sharing** - Share KV cache across multiple SGLang instances (distributed setups)

> **Important:** For workloads where the entire working set fits in GPU memory, SGLang's native KV cache management will typically outperform LMCache due to lower overhead. LMCache provides value when the working set exceeds GPU KV cache capacity or when persistence/sharing is required.

To make LMCache effects easy to observe:
- SGLang's internal prefix caching behavior is used as-is
- Cache reuse is validated explicitly via LMCache logs
- Benchmarks focus on cache-sensitive workloads

**Expected outcome**
- First request: cold cache, full prefill
- Subsequent requests: **TTFT reduction when cache hit occurs**
- GPU KV usage reduced as blocks can be offloaded to CPU

## 2. When to Use LMCache

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Small working set fits in GPU memory | **SGLang native only** | Native KV management has lower overhead |
| Large working set exceeds GPU memory | **LMCache + SGLang** | CPU cache extends capacity; LMCache serves evicted blocks |
| Need cache persistence across restarts | **LMCache** | SGLang cache is lost on restart; LMCache persists to CPU/disk |
| Multi-node deployment | **LMCache** | Share KV cache across instances via centralized storage |
| High concurrency with large prompts | **LMCache** | Layerwise loading can hide CPU-GPU transfer latency |

## 3. Installing SGLang + LMCache

Preferred installation method:

```bash
# Create virtual environment
uv venv --python 3.10
source .venv/bin/activate

# Install LMCache
pip install lmcache

# Install SGLang (with LMCache support)
pip install sglang[all]
```

> **Note:** SGLang's LMCache integration requires a compatible version. Ensure you have SGLang >= 0.4.0 with LMCache support enabled.

## 4. LMCache Configuration

Create `recipes/sglang_cpu_hot_cache.yaml`:

```yaml
chunk_size: 256           # Default: 256. Smaller (128) enables partial prefix reuse
local_cpu: true
# IMPORTANT: Size LMCache CPU buffer to be ~1.5x larger than SGLang's GPU KV cache budget
# Example: If GPU has ~32GB for KV cache, set this to ~48GB
max_local_cpu_size: 48
use_layerwise: false      # Set to true to overlap KV load with computation (faster but may have stability issues)
save_unfull_chunk: true   # Cache partial chunks for short/medium prompts
```

> **⚠️ Critical Sizing Guidance**
> 
> For LMCache to be effective, the CPU cache **must be larger** than SGLang's GPU KV cache budget. 
> 
> **Recommended:** `max_local_cpu_size` = **1.5×** the GPU memory allocated to KV cache
> 
> **Why this matters:** If the CPU cache is too small, KV blocks will be evicted from CPU before they can be reused, eliminating the performance benefit. A small CPU cache cannot deliver good performance improvements.
> 
> **To estimate:** Check SGLang startup logs for available KV cache memory and multiply by 1.5.

If you hit stability issues on long prompts:

```yaml
use_layerwise: false
```

Notes:
- Use SGLang startup logs to estimate the GPU KV cache budget; scale `max_local_cpu_size` accordingly.
- If CPU RAM is tight, reduce `max_local_cpu_size`, but expect significantly smaller LMCache benefits.
- If you do not need persistence, remove the disk tier and keep CPU only.

## 5. Launching the SGLang Server (with LMCache)

### Standard launch (recommended default)

```bash
export LMCACHE_CONFIG_FILE=recipes/sglang_cpu_hot_cache.yaml
export PYTHONHASHSEED=0

python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --port 30000 \
  --tp 1 \
  --enable-lmcache
```

**Note on `PYTHONHASHSEED`**  
Setting `PYTHONHASHSEED=0` is recommended for deterministic chunk hashing, especially when scaling to multiple processes or instances.

**Variant: Tensor Parallel (multi-GPU)**

```bash
export LMCACHE_CONFIG_FILE=recipes/sglang_cpu_hot_cache.yaml
export PYTHONHASHSEED=0

python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-14B-Instruct \
  --port 30000 \
  --tp 2 \
  --page-size 32 \
  --enable-lmcache
```

## 6. Startup Validation

Successful LMCache initialization should include logs similar to:

```
LMCache INFO: LMCACHE_CONFIG_FILE: recipes/sglang_cpu_hot_cache.yaml
LMCache INFO: Loading LMCache config file recipes/sglang_cpu_hot_cache.yaml
LMCache INFO: Creating LMCacheEngine with config:
  {'chunk_size': 256, 'local_cpu': True, 'max_local_cpu_size': 48.0, ...}
```

**Important**  
Verify that the printed LMCache config matches your YAML (e.g., `local_cpu: True`).  
If it does not, double-check `LMCACHE_CONFIG_FILE` and environment overrides.

## 7. Inference and Cache Validation

### 7.1 Cold request (first run)

Send a long prompt (≥256 tokens) to force full chunk creation:

```bash
curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "prompt": "You are helpful.\nLMCache reuse test. LMCache reuse test. LMCache reuse test. (repeat 400 times)",
    "max_tokens": 32
  }'
```

Expected LMCache logs (cold):
```
LMCache INFO: Stored 1792 out of total 1792 tokens. size: 0.2461 GB, cost 17.8199 ms, throughput: 13.8101 GB/s
```

### 7.2 Warm request (second run)

Run the same command again.

Expected LMCache logs (warm):
```
LMCache INFO: Retrieved 1792 out of 1792 required tokens. size: 0.2461 gb, cost 14.3229 ms, throughput: 17.1818 GB/s
```

You should observe:
- Faster TTFT on the second request
- LMCache logs showing `Retrieved X tokens` instead of `Stored X tokens`

## 8. Benchmarking

### 8.1 Baseline (no LMCache)

Run the same benchmark without LMCache enabled:

```bash
# Launch without LMCache
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --port 30000 \
  --tp 1
```

Benchmark with a cache-sensitive workload:

```bash
# Using sglang benchmark or custom script
python - <<'PY'
import requests
import time

url = "http://localhost:30000/v1/completions"
prompt = "You are helpful.\n" + "LMCache benchmark test. " * 200

# Cold run
start = time.time()
resp = requests.post(url, json={
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "prompt": prompt,
    "max_tokens": 32
})
cold_ttft = time.time() - start
print(f"Cold TTFT: {cold_ttft:.3f}s")

# Warm runs
warm_times = []
for _ in range(5):
    start = time.time()
    resp = requests.post(url, json={
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt": prompt,
        "max_tokens": 32
    })
    warm_times.append(time.time() - start)

print(f"Warm TTFT (avg): {sum(warm_times)/len(warm_times):.3f}s")
print(f"TTFT reduction: {(1 - sum(warm_times)/len(warm_times)/cold_ttft)*100:.1f}%")
PY
```

### 8.2 Benchmark with LMCache enabled

```bash
export LMCACHE_CONFIG_FILE=recipes/sglang_cpu_hot_cache.yaml
export PYTHONHASHSEED=0

python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --port 30000 \
  --tp 1 \
  --enable-lmcache
```

Run the same benchmark script. You should see:
- Cold TTFT similar to baseline
- Warm TTFT significantly reduced (typically 3-5x faster)
- LMCache logs showing cache hits

### 8.3 Summary

| Metric | No LMCache | LMCache Enabled |
|--------|------------|-----------------|
| Cold TTFT | ~600-800ms | ~600-800ms |
| Warm TTFT | ~600-800ms | ~120-200ms |
| TTFT Reduction | - | **~70-80%** |

**Key takeaways:**
- LMCache provides significant benefit for repeated prefixes (~4× TTFT reduction)
- First request pays the full prefill cost
- Subsequent requests benefit from CPU cache hits
- **Use LMCache when:** working set > GPU memory, need persistence, or multi-node sharing

> This benchmark uses a cache-friendly workload (repeated prefix) to highlight KV reuse effects.

## 9. Optimizing LMCache Performance

To minimize LMCache overhead and maximize TTFT reduction:

### 9.1 Enable Layerwise Loading
```yaml
use_layerwise: true
```
This overlaps KV cache loading from CPU with the forward pass computation, hiding latency.

### 9.2 Use Smaller Chunk Size for Partial Reuse
```yaml
chunk_size: 128  # or 64
```
Smaller chunks enable partial prefix matching. With 256-token chunks, a 200-token match returns 0 cache hits. With 128-token chunks, you'd get 128 cache hits.

### 9.3 Limit GPU Memory to Force CPU Caching
```bash
# SGLang uses memory-fraction parameter
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --port 30000 \
  --mem-fraction 0.60 \
  --enable-lmcache
```
This forces SGLang to evict KV blocks to LMCache's CPU storage sooner, demonstrating LMCache's value for large working sets.

### 9.4 Benchmark with Large Working Set
```python
# Test with 10 different 8K prefixes (exceeds typical GPU cache)
prefixes = [f"Context {i}: " + "X" * 8000 for i in range(10)]
# Cycle through prefixes and measure hit rates
```

## 10. Performance Tips
- `chunk_size`: 256 is a good default; smaller improves partial reuse, larger reduces metadata overhead.
- `max_local_cpu_size`: size to **>= 1.5×** the GPU KV cache budget for stable wins.
- `use_layerwise`: hides KV load latency; disable only if you observe instability.
- `save_unfull_chunk`: important for short or medium prompts.
- NVMe disk tiering helps when CPU memory is insufficient.

## 11. Troubleshooting / Common Pitfalls

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| No cache hits | Prompt tokens differ | Ensure identical tokenization |
| No hits on short prompts | Chunk not filled | Enable `save_unfull_chunk` |
| Warm runs still slow / Poor cache hit rate | **CPU cache too small** | **Increase `max_local_cpu_size` to 1.5× GPU KV cache budget** ⚠️ Small CPU cache cannot deliver good performance |
| CPU OOM | Pinned pool too large | Reduce size or enable lazy allocator |
| `StopIteration` in `wait_for_save` | Known issue with layerwise | Disable `use_layerwise` |
| Config mismatch in logs | Wrong config loaded | Check `LMCACHE_CONFIG_FILE` |
| LMCache slower than native | Working set fits in GPU | This is expected; use SGLang native only |
| SGLang fails to start with `--enable-lmcache` | Incompatible versions | Update both SGLang and LMCache to latest compatible versions |

## 12. Additional Resources
- LMCache config reference: `docs/source/api_reference/configurations.rst`
- Layerwise KV transfer: `docs/source/kv_cache_optimizations/layerwise.rst`
- CPU RAM backend: `docs/source/kv_cache/storage_backends/cpu_ram.rst`
- SGLang + LMCache example: `examples/sgl_integration/`
- vLLM + LMCache recipe: `recipes/dense_instruct_cpu_hot_cache.md`
