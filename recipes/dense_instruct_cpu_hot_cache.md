# LMCache + vLLM: CPU-Extended KV Cache for Large Working Sets

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

This recipe demonstrates how to run **vLLM with LMCache enabled** on a single GPU using a **CPU pinned-memory hot cache**. LMCache extends vLLM's prefix caching to CPU memory, enabling:

1. **Larger working sets** - Cache more prefixes than GPU memory allows
2. **Cache persistence** - KV cache survives vLLM restarts
3. **Cross-instance sharing** - Share KV cache across multiple vLLM instances (distributed setups)

> **Important:** For workloads where the entire working set fits in GPU memory, vLLM's native prefix caching will typically outperform LMCache due to lower overhead. LMCache provides value when the working set exceeds GPU KV cache capacity or when persistence/sharing is required.

To make LMCache effects easy to observe:
- vLLM's internal prefix caching is disabled for measurement only
- cache reuse is validated explicitly via LMCache logs
- benchmarks focus on cache-sensitive workloads

**Expected outcome**
- First request: cold cache, full prefill
- Subsequent requests: **TTFT reduction when cache hit occurs**
- GPU KV usage reduced as blocks can be offloaded to CPU

## 2. When to Use LMCache

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Small working set fits in GPU memory | **vLLM native only** | Native prefix caching has lower overhead (~10ms faster TTFT) |
| Large working set exceeds GPU memory | **LMCache + vLLM** | CPU cache extends capacity; LMCache serves evicted blocks |
| Need cache persistence across restarts | **LMCache** | vLLM cache is lost on restart; LMCache persists to CPU/disk |
| Multi-node deployment | **LMCache** | Share KV cache across instances via centralized storage |
| High concurrency with large prompts | **LMCache** | Layerwise loading can hide CPU-GPU transfer latency |

## 3. Installing vLLM + LMCache

Preferred (uv):

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install lmcache vllm
```

## 4. LMCache Configuration

Create `recipes/dense_instruct_cpu_hot_cache.yaml`:

```yaml
chunk_size: 256           # Default: 256. Smaller (128) enables partial prefix reuse
local_cpu: true
max_local_cpu_size: 12    # Size in GB. Should be >=1.5x your GPU KV cache budget
use_layerwise: false      # Set to true to overlap KV load with computation (faster but may have stability issues)
save_unfull_chunk: true   # Cache partial chunks for short/medium prompts
```

If you hit `StopIteration` in `wait_for_save` on long prompts, disable layerwise:

```yaml
use_layerwise: false
```

Notes:
- In practice, `max_local_cpu_size` should be **>= 1.5×** the GPU KV cache budget to see consistent gains.
- Use vLLM startup logs to estimate the GPU KV cache budget; scale `max_local_cpu_size` accordingly.
- If CPU RAM is tight, reduce `max_local_cpu_size`, but expect smaller LMCache benefits.
- If you do not need persistence, remove the disk tier and keep CPU only.

## 5. Launching the vLLM Server (with LMCache)

### Why disable vLLM prefix caching?

Prefix caching is disabled so that **all reuse comes from LMCache**, making cache hits and TTFT deltas easier to interpret. For real deployments, keep vLLM prefix caching enabled and size the CPU cache appropriately.

### Standard connector (recommended default)

```bash
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=recipes/dense_instruct_cpu_hot_cache.yaml \
CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
--max-model-len 8192 \
--gpu-memory-utilization 0.85 \
--port 8000 \
--no-enable-prefix-caching \
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

**Note on `PYTHONHASHSEED`**  
Setting `PYTHONHASHSEED=0` is recommended for deterministic chunk hashing, especially when scaling to multiple processes or instances.

**Variant: Dynamic connector**
- use when you pin a newer LMCache build or need module override.
```bash
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1Dynamic","kv_role":"kv_both","kv_connector_module_path":"lmcache.integration.vllm.lmcache_connector_v1"}'
```

## 6. Startup Validation

Successful LMCache initialization should include logs similar to:

`LMCache INFO: Loading LMCache config file recipes/dense_instruct_cpu_hot_cache.yaml LMCache INFO: LMCache initialized for role KVConnectorRole.WORKER LMCache INFO: Creating LMCacheEngine with config:   {'chunk_size': 256, 'local_cpu': True, 'max_local_cpu_size': 12.0, ...}`

**Important**  
Verify that the printed LMCache config matches your YAML (e.g., `use_layerwise: false`).  
If it does not, double-check `LMCACHE_CONFIG_FILE` and environment overrides.

## 7. Inference and Cache Validation

### 7.1 Cold request (first run)

Send a long prompt (≥256 tokens) to force full chunk creation:

```bash
python - <<'PY' | curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d @-
import json
prompt = "You are helpful.\n" + ("LMCache reuse test. " * 400)
payload = {
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": prompt,
    "max_tokens": 32,
}
print(json.dumps(payload))
PY
```

As you may expected LMCache logs (cold):
```bash
(EngineCore_DP0 pid=480732) [2026-01-20 21:20:40,951] LMCache INFO: Reqid: cmpl-bd1ae3af9e4f2568-0, Total tokens 2005, LMCache hit tokens: 0, need to load: 0 (vllm_v1_adapter.py:1284:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=480732) [2026-01-20 21:20:41,133] LMCache INFO: Stored 1792 out of total 1792 tokens. size: 0.2461 GB, cost 17.8199 ms, throughput: 13.8101 GB/s; offload_time: 17.7556 ms, put_time: 0.0643 ms (cache_engine.py:488:lmcache.v1.cache_engine)
(APIServer pid=480469) INFO:     127.0.0.1:40212 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=480469) INFO 01-20 21:20:49 [loggers.py:248] Engine 000: Avg prompt throughput: 200.5 tokens/s, Avg generation throughput: 3.2 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%, External prefix cache hit rate: 0.0%

```

### 7.2 Warm request (second run)

Run the same command again.

Expected LMCache logs (warm):
```bash
(EngineCore_DP0 pid=480732) [2026-01-20 21:21:58,694] LMCache INFO: Reqid: cmpl-beba3876c5b2d373-0, Total tokens 2005, LMCache hit tokens: 1792, need to load: 1792 (vllm_v1_adapter.py:1284:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=480732) [2026-01-20 21:21:58,711] LMCache INFO: Retrieved 1792 out of 1792 required tokens (from 1792 total tokens). size: 0.2461 gb, cost 14.3229 ms, throughput: 17.1818 GB/s; (cache_engine.py:812:lmcache.v1.cache_engine)
(APIServer pid=480469) INFO:     127.0.0.1:36102 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=480469) INFO 01-20 21:21:59 [loggers.py:248] Engine 000: Avg prompt throughput: 200.5 tokens/s, Avg generation throughput: 3.2 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%, External prefix cache hit rate: 44.7%
```

```bash
External prefix cache hit rate: 44.7%
```
This is expected: vLLM labels KV supplied by external connectors (LMCache) as **external prefix cache**.

## 8. Benchmarking

### 8.1 Baseline (no LMCache)

Run the same benchmark twice: once without LMCache and once with LMCache. Keep
vLLM prefix caching disabled in both runs to isolate LMCache effects (benchmark-only).

Baseline launch (no LMCache):

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --port 8000 \
  --no-enable-prefix-caching
```

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
--max-model-len 8192 \
--gpu-memory-utilization 0.85 \
--port 8000 \
--no-enable-prefix-caching
```

Benchmark result
```bash
============ Serving Benchmark Result ============
Successful requests:                     50
Failed requests:                         0
Request rate configured (RPS):           1.00
Benchmark duration (s):                  50.86
Total input tokens:                      12800
Total generated tokens:                  3200
Request throughput (req/s):              0.98
Output token throughput (tok/s):         62.91
Peak output token throughput (tok/s):    186.00
Peak concurrent requests:                4.00
Total token throughput (tok/s):          314.57
---------------Time to First Token----------------
Mean TTFT (ms):                          42.04
Median TTFT (ms):                        40.54
P99 TTFT (ms):                           53.00
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          13.09
Median TPOT (ms):                        13.03
P99 TPOT (ms):                           13.87
---------------Inter-token Latency----------------
Mean ITL (ms):                           13.09
Median ITL (ms):                         12.94
P99 ITL (ms):                            26.19
==================================================

```

### 8.2 Cache-sensitive benchmark (prefix repetition)
For cache-sensitive TTFT deltas, use prefix repetition (repeats a long prefix):

```bash
vllm bench serve --model Qwen/Qwen3-4B-Instruct-2507 \
--dataset-name prefix_repetition \
--prefix-repetition-prefix-len 6144 \
--prefix-repetition-suffix-len 128 \
--prefix-repetition-num-prefixes 1 \
--prefix-repetition-output-len 32 \
--num-prompts 100 --request-rate 0.5 --max-concurrency 1
```

Benchmark result (no caching)
```bash
============ Serving Benchmark Result ============
Successful requests:                     100
Failed requests:                         0
Maximum request concurrency:             1
Request rate configured (RPS):           0.50
Benchmark duration (s):                  203.21
Total input tokens:                      627200
Total generated tokens:                  3200
Request throughput (req/s):              0.49
Output token throughput (tok/s):         15.75
Peak output token throughput (tok/s):    32.00
Peak concurrent requests:                2.00
Total token throughput (tok/s):          3102.24
---------------Time to First Token----------------
Mean TTFT (ms):                          633.39
Median TTFT (ms):                        633.61
P99 TTFT (ms):                           659.72
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          13.98
Median TPOT (ms):                        13.96
P99 TPOT (ms):                           14.31
---------------Inter-token Latency----------------
Mean ITL (ms):                           13.98
Median ITL (ms):                         13.95
P99 ITL (ms):                            14.75
==================================================
```

**Without LMCache**
- Mean TTFT: **633.39 ms**

### 8.3 Benchmark with LMCache enabled

```bash
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=recipes/dense_instruct_cpu_hot_cache.yaml \
CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
--max-model-len 8192 \
--gpu-memory-utilization 0.85 \
--port 8000 \
--no-enable-prefix-caching \
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

Benchmark
```bash
vllm bench serve --model Qwen/Qwen3-4B-Instruct-2507 \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 6144 \
  --prefix-repetition-suffix-len 128 \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len 32 \
  --num-prompts 100 --request-rate 0.5 --max-concurrency 1
```

Benchmark result (LMCache, no native prefix caching)
```bash
============ Serving Benchmark Result ============
Successful requests:                     100
Failed requests:                         0
Maximum request concurrency:             1
Request rate configured (RPS):           0.50
Benchmark duration (s):                  201.08
Total input tokens:                      627200
Total generated tokens:                  3200
Request throughput (req/s):              0.50
Output token throughput (tok/s):         15.91
Peak output token throughput (tok/s):    64.00
Peak concurrent requests:                3.00
Total token throughput (tok/s):          3135.02
---------------Time to First Token----------------
Mean TTFT (ms):                          120.49
Median TTFT (ms):                        120.35
P99 TTFT (ms):                           151.11
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          13.98
Median TPOT (ms):                        13.97
P99 TPOT (ms):                           14.29
---------------Inter-token Latency----------------
Mean ITL (ms):                           13.98
Median ITL (ms):                         13.94
P99 ITL (ms):                            14.96
==================================================

```

**With LMCache**
- Mean TTFT: **120.49 ms**

### 8.4 Benchmark with vLLM native prefix caching

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --port 8000
```

Benchmark result (vLLM native prefix caching)
```bash
============ Serving Benchmark Result ============
Successful requests:                     20
Failed requests:                         0
Maximum request concurrency:             1
Request rate configured (RPS):           0.50
Benchmark duration (s):                  40.50
Total input tokens:                      125440
Total generated tokens:                  640
Request throughput (req/s):              0.49
Output token throughput (tok/s):         15.80
Peak output token throughput (tok/s):    50.00
Peak concurrent requests:                2.00
Total token throughput (tok/s):          3113.20
---------------Time to First Token----------------
Mean TTFT (ms):                          106.65
Median TTFT (ms):                        80.10
P99 TTFT (ms):                           567.35
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          13.37
Median TPOT (ms):                        13.37
P99 TPOT (ms):                           13.41
---------------Inter-token Latency----------------
Mean ITL (ms):                           13.37
Median ITL (ms):                         13.33
P99 ITL (ms):                            14.90
==================================================
```

**With vLLM native prefix caching**
- Mean TTFT: **106.65 ms**

### 8.5 Summary

**Benchmark context:** This comparison shows results with prefix caching disabled to isolate LMCache's effects. In production, you should keep vLLM prefix caching enabled.

| Metric | No Caching | LMCache Only | vLLM Native Only | vLLM + LMCache |
|--------|------------|--------------|------------------|----------------|
| Mean TTFT | 633.39 ms | 120.49 ms | **106.65 ms** | 116.70 ms |
| Median TTFT | 633.61 ms | 120.35 ms | **80.10 ms** | 87.06 ms |
| P99 TTFT | 659.72 ms | 151.11 ms | **567.35 ms** | 615.70 ms |

**Key takeaways:**
- LMCache provides significant benefit vs no caching (~4× TTFT reduction)
- vLLM native prefix caching is fastest for GPU-fitting workloads (~6× TTFT reduction)
- Running both adds ~10ms overhead vs native alone
- **Use LMCache when:** working set > GPU memory, need persistence, or multi-node sharing

> This benchmark uses a cache-friendly workload (`prefix_repetition`) to highlight KV reuse effects.

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
vllm serve ... --gpu-memory-utilization 0.50
```
This forces vLLM to evict KV blocks to LMCache's CPU storage sooner, demonstrating LMCache's value for large working sets.

### 9.4 Benchmark with Large Working Set
```bash
# 10 different 8K prefixes (exceeds typical GPU cache)
vllm bench serve --model Qwen/Qwen3-4B-Instruct-2507 \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 8192 \
  --prefix-repetition-suffix-len 128 \
  --prefix-repetition-num-prefixes 10 \
  --prefix-repetition-output-len 32 \
  --num-prompts 100 \
  --request-rate 1.0 \
  --max-concurrency 5
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
| Warm runs still slow | CPU cache too small | Increase `max_local_cpu_size` (>=1.5× GPU KV cache budget) |
| CPU OOM | Pinned pool too large | Reduce size or enable lazy allocator |
| `StopIteration` in `wait_for_save` | Known issue | Disable `use_layerwise` |
| Config mismatch in logs | Wrong config loaded | Check `LMCACHE_CONFIG_FILE` |
| LMCache slower than native | Working set fits in GPU | This is expected; use vLLM native only |

## 12. Additional Resources
- LMCache config reference: `docs/source/api_reference/configurations.rst`
- Layerwise KV transfer: `docs/source/kv_cache_optimizations/layerwise.rst`
- CPU RAM backend: `docs/source/kv_cache/storage_backends/cpu_ram.rst`
- vLLM + LMCache quickstart: `docs/source/getting_started/quickstart.rst`
- Example launch patterns: `examples/cache_with_configs/README.md`
