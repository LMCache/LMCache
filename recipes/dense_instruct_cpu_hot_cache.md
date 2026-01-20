# LMCache + vLLM: Qwen3-4B-Instruct on 1xGPU for Multi-Turn Chat

## 1. Introduction
**Target workload**
- Multi-turn chat
- Repeated system prompt
- Small repeated RAG-style context

**LMCache mode**
- **Storage Mode**
- Single node
- CPU hot cache only
This recipe demonstrates how to run **vLLM with LMCache enabled** on a single GPU using a **CPU pinned-memory hot cache**. KV blocks generated during the first request are cached and reused on subsequent requests, reducing **time-to-first-token (TTFT)** and GPU KV pressure.

To make LMCache effects easy to observe:
- vLLM’s internal prefix caching is disabled
- cache reuse is validated explicitly via LMCache logs
- benchmarks focus on cache-sensitive workloads

**Expected outcome**
- First request: cold cache, full prefill
- Subsequent requests with identical token chunks: **large TTFT reduction**
- GPU KV usage remains near zero on warm runs

## 2. Installing vLLM + LMCache
Preferred (uv):

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install lmcache vllm
```

## 3. LMCache Configuration

Create `recipes/dense_instruct_cpu_hot_cache.yaml`:

```yaml
chunk_size: 256           # Default chunk size
local_cpu: true
max_local_cpu_size: 8     # GB of pinned CPU memory
use_layerwise: true       # Overlap KV load with forward pass
```

If you hit `StopIteration` in `wait_for_save` on long prompts, disable layerwise:

```yaml
use_layerwise: false
```

Notes:
- If CPU RAM is tight, reduce `max_local_cpu_size`.
- If you do not need persistence, remove the disk tier and keep CPU only.

## 4. Launching the vLLM Server (with LMCache)

### Why disable vLLM prefix caching?

Prefix caching is disabled so that **all reuse comes from LMCache**, making cache hits and TTFT deltas easier to interpret.

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

## 5. Startup Validation
Successful LMCache initialization should include logs similar to:

`LMCache INFO: Loading LMCache config file recipes/dense_instruct_cpu_hot_cache.yaml LMCache INFO: LMCache initialized for role KVConnectorRole.WORKER LMCache INFO: Creating LMCacheEngine with config:   {'chunk_size': 256, 'local_cpu': True, 'max_local_cpu_size': 8.0, ...}`

**Important**  
Verify that the printed LMCache config matches your YAML (e.g., `use_layerwise: true`).  
If it does not, double-check `LMCACHE_CONFIG_FILE` and environment overrides.

## 6. Inference and Cache Validation

### 6.1 Cold request (first run)

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

### 6.2 Warm request (second run)

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
## 7. Benchmarking

### 7.1 Baseline (no LMCache)

Run the same benchmark twice: once without LMCache and once with LMCache. Keep
vLLM prefix caching disabled in both runs to isolate LMCache effects.

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

### 7.2 Cache-sensitive benchmark (prefix repetition)
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

Benchmark result
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

### 7.3 Benchmark with LMCache enabled

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

Benchmark result
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
### 7.4 Summary
|Metric|Without LMCache|With LMCache|
|---|---|---|
|Mean TTFT|633.39 ms|120.49 ms|
|Absolute reduction|–|**−512.90 ms**|
|Relative reduction|–|**≈81.0%**|
|Speedup|–|**≈5.26×**|

> This benchmark uses a cache-friendly workload (`prefix_repetition`) to highlight KV reuse effects.

## 8. Performance Tips
- `chunk_size`: 256 is a good default; smaller improves partial reuse, larger reduces metadata overhead.
- `max_local_cpu_size`: increase if you have repeated prompts and spare RAM.
- `use_layerwise`: hides KV load latency; disable only if you observe instability.
- `save_unfull_chunk`: important for short or medium prompts.
- NVMe disk tiering helps when CPU memory is insufficient.

## 9. Troubleshooting / Common Pitfalls
|Symptom|Likely cause|Fix|
|---|---|---|
|No cache hits|Prompt tokens differ|Ensure identical tokenization|
|No hits on short prompts|Chunk not filled|Enable `save_unfull_chunk`|
|Warm runs still slow|CPU cache too small|Increase `max_local_cpu_size`|
|CPU OOM|Pinned pool too large|Reduce size or enable lazy allocator|
|`StopIteration` in `wait_for_save`|Known issue|Disable `use_layerwise`|
|Config mismatch in logs|Wrong config loaded|Check `LMCACHE_CONFIG_FILE`

## 10. Additional Resources
- LMCache config reference: `docs/source/api_reference/configurations.rst`
- Layerwise KV transfer: `docs/source/kv_cache_optimizations/layerwise.rst`
- CPU RAM backend: `docs/source/kv_cache/storage_backends/cpu_ram.rst`
- vLLM + LMCache quickstart: `docs/source/getting_started/quickstart.rst`
- Example launch patterns: `examples/cache_with_configs/README.md`
