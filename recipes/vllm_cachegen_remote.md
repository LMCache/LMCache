# LMCache + vLLM: CacheGen Compression for Remote KV

## 1. Introduction

**Target workload**
- Remote KV backends (Redis/Valkey) with network bottlenecks
- Large KV working sets that need reduced transfer size
- Environments where small accuracy loss is acceptable

**LMCache mode**
- **Storage Mode**
- Remote backend with CacheGen compression
- Local CPU hot cache enabled

This recipe shows how to enable **CacheGen** as the remote serde to compress KV before sending to the remote backend. CacheGen reduces KV size and improves transfer efficiency, at the cost of some reconstruction error.

**Expected outcome**
- Remote KV payloads are compressed before transfer
- Lower network bandwidth usage for remote hits
- Improved TTFT for remote retrievals (relative to uncompressed remote)

## 2. When to Use CacheGen

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Remote backend over network | **CacheGen** | Reduces transfer size |
| CPU/disk only | **Optional** | Less network benefit |
| Strict accuracy requirements | **Avoid** | CacheGen is lossy |
| Long contexts, large KV | **CacheGen** | Higher compression impact |

## 3. Installing vLLM + LMCache

Prerequisites:
- Redis or Valkey remote backend running
- vLLM + LMCache installed

Install vLLM and LMCache:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install lmcache vllm
```

## 4. LMCache Configuration

Create `recipes/vllm_cachegen_remote.yaml`:

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 48
local_disk: false
remote_url: "redis://redis.lmcache.svc.cluster.local:6379"
remote_serde: "cachegen"
use_layerwise: false
save_unfull_chunk: true
```

**CacheGen config notes**
- `remote_serde: "cachegen"` enables compression for remote storage.
- CacheGen is **lossy**; some numerical differences are expected.
- For best results, keep `chunk_size` consistent across instances.

## 5. Launching the Server (with LMCache)

```bash
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=recipes/vllm_cachegen_remote.yaml \
CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --port 8000 \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

## 6. Startup Validation

Expected logs:

```
LMCache INFO: Loading LMCache config file recipes/vllm_cachegen_remote.yaml
LMCache INFO: Creating LMCacheEngine with config:
  {'remote_serde': 'cachegen', 'remote_url': 'redis://...', ...}
```

## 7. Inference and Cache Validation

### 7.1 Cold request (first run)

```bash
python - <<'PY' | curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d @-
import json
prompt = "You are helpful.\n" + ("CacheGen compression test. " * 400)
payload = {
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": prompt,
    "max_tokens": 32,
}
print(json.dumps(payload))
PY
```

Expected logs (store to remote):

```
LMCache INFO: Stored 1792 out of total 1792 tokens. size: 0.2461 GB
```

### 7.2 Warm request (remote hit)

Repeat the request and confirm remote hit:

```
LMCache INFO: Retrieved 1792 out of 1792 required tokens. size: 0.2461 gb
```

Note: With CacheGen, exact KV equality is not guaranteed. Expect small numerical differences.

## 8. Benchmarking

### 8.1 Baseline (remote, no compression)

Use the Redis remote recipe with `remote_serde: "naive"` and run:

```bash
vllm bench serve --model Qwen/Qwen3-4B-Instruct-2507 \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 6144 \
  --prefix-repetition-suffix-len 128 \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len 32 \
  --num-prompts 50 --request-rate 0.5 --max-concurrency 1
```

### 8.2 CacheGen enabled

Repeat the same benchmark with `remote_serde: "cachegen"`.

### Example comparison

| Scenario | Remote payload size | TTFT (warm) | Notes |
|----------|----------------------|-------------|-------|
| Remote naive | 1.0x | ~220ms | Full KV transfer |
| CacheGen | 0.5-0.7x | ~150ms | Smaller KV, lossy |

## 9. Optimizing Performance

- Use CacheGen only for remote tiers where network is a bottleneck.
- Keep local CPU hot cache enabled to avoid frequent remote hits.
- For long contexts, increase `chunk_size` to improve compression efficiency.

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| No compression effect | `remote_serde` not set | Set `remote_serde: "cachegen"` |
| Accuracy concerns | Lossy compression | Use `remote_serde: "naive"` |
| Remote errors | Backend unreachable | Verify `remote_url` and network |
| Low hit rate | Prompt mismatch | Ensure identical prompts/tokenization |

## 11. Additional Resources

- Redis remote backend: `recipes/vllm_redis_remote.md`
- LMCache config definitions: `lmcache/v1/config.py`
- CacheGen tests: `tests/benchmarks/test_cachegen.py`
