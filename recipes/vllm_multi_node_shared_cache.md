# LMCache + vLLM: Multi-Node Shared Cache with Redis Tier

## 1. Introduction

**Target workload**
- Horizontally scaled vLLM deployments (multiple nodes or pods)
- Repeated prompts across instances (shared system prompt, RAG prefix, agent tools)
- Multi-tenant serving where prompts overlap across users
- Autoscaling environments where cache persistence matters

**LMCache mode**
- **Storage Mode**
- Multi-node
- Local CPU hot cache + shared Redis remote tier

This recipe shows how to run **many vLLM instances** that share KV caches through a **centralized Redis tier**. Each node keeps a **local CPU hot cache** for fast hits and falls back to Redis for shared hits.

**Expected outcome**
- Instance A warms the Redis tier
- Instance B/C retrieve KV from Redis on first request
- Subsequent requests on B/C hit local CPU for lowest latency

Architecture (example with 3 nodes):

```
          ┌─────────────────────────────┐
          │         Load Balancer       │
          └──────────────┬──────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼───────┐ ┌──────▼──────┐ ┌──────▼──────┐
│ vLLM Node A   │ │ vLLM Node B │ │ vLLM Node C │
│ GPU + CPU hot │ │ GPU + CPU   │ │ GPU + CPU   │
│ LMCache tier  │ │ LMCache tier│ │ LMCache tier│
└───────┬───────┘ └──────┬──────┘ └──────┬──────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
                 ┌───────▼────────┐
                 │ Redis Shared   │
                 │ LMCache Tier   │
                 └────────────────┘
```

## 2. When to Use LMCache Shared Tier

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Multiple serving nodes | **LMCache + Redis** | Share KV across instances, reduce recompute |
| Autoscaling pods | **LMCache + Redis** | Cache persists outside ephemeral pods |
| Low-latency single node | **Local only** | Avoid network overhead |
| Large working set + scale-out | **LMCache + local CPU + Redis** | Local hot hits + shared remote tier |
| Strict data isolation | **Per-tenant Redis** | Avoid cross-tenant cache bleed |

## 3. Installing vLLM + LMCache

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install lmcache vllm
```

Ensure Redis is reachable from all vLLM nodes (same VPC or Kubernetes cluster).

## 4. LMCache Configuration

Create `recipes/vllm_multi_node_shared_cache.yaml`:

```yaml
chunk_size: 256
# IMPORTANT: All instances must use the same chunk_size
local_cpu: true
# Size local CPU cache to ~1.5x GPU KV cache budget per instance
max_local_cpu_size: 48
local_disk: false
# Shared Redis tier (centralized KV cache)
remote_url: "redis://redis.lmcache.svc.cluster.local:6379"
use_layerwise: false
save_unfull_chunk: true
```

**Critical sizing guidance**
- **Local CPU tier**: set `max_local_cpu_size` to ~1.5x the GPU KV cache budget per instance.
- **Redis tier**: size Redis memory to 2-3x the total working set across all nodes.
- **Chunk size**: keep `chunk_size` identical across all instances or cache hits will fail.

## 5. Launching the vLLM Servers (with LMCache)

Launch each node with the same LMCache config and Redis URL. Use different ports per node.

```bash
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=recipes/vllm_multi_node_shared_cache.yaml \
CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --port 8000 \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

Repeat on Node B/C with different ports and GPUs. For production, keep vLLM native prefix caching enabled and drop `--no-enable-prefix-caching`.

## 6. Startup Validation

Each node should show LMCache initialization and Redis backend:

```
LMCache INFO: Loading LMCache config file recipes/vllm_multi_node_shared_cache.yaml
LMCache INFO: Creating LMCacheEngine with config:
  {'chunk_size': 256, 'local_cpu': True, 'remote_url': 'redis://redis.lmcache.svc.cluster.local:6379', ...}
LMCache INFO: Initializing RedisBackend at redis.lmcache.svc.cluster.local:6379
```

From a Redis host/pod, verify connections:

```bash
redis-cli -h redis.lmcache.svc.cluster.local -p 6379 CLIENT LIST | wc -l
```

## 7. Inference and Cache Validation

### 7.1 Warm the shared cache on Node A

```bash
python - <<'PY' | curl http://node-a:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d @-
import json
prompt = "System: You are helpful.\n" + ("Shared prompt. " * 400)
payload = {
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": prompt,
    "max_tokens": 64,
}
print(json.dumps(payload))
PY
```

Expected Node A logs (cold, stores to Redis):

```
LMCache INFO: Reqid: ..., Total tokens 2048, LMCache hit tokens: 0, need to load: 0
LMCache INFO: Stored 1792 out of total 1792 tokens. size: 0.2461 GB
```

### 7.2 Get a shared cache hit on Node B

```bash
python - <<'PY' | curl http://node-b:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d @-
import json
prompt = "System: You are helpful.\n" + ("Shared prompt. " * 400)
payload = {
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": prompt,
    "max_tokens": 64,
}
print(json.dumps(payload))
PY
```

Expected Node B logs (warm, retrieves from Redis):

```
LMCache INFO: Reqid: ..., Total tokens 2048, LMCache hit tokens: 1792, need to load: 1792
LMCache INFO: Retrieved 1792 out of 1792 required tokens. size: 0.2461 gb
```

Optional: repeat on Node B to confirm **local CPU hot hits**.

## 8. Benchmarking

### 8.1 Baseline (no sharing)

Start two nodes **without Redis** (local only) and run the same prompt on both. Both will compute from scratch.

### 8.2 Shared cache (Redis)

Use `prefix_repetition` to warm Node A, then run Node B:

```bash
vllm bench serve --model Qwen/Qwen3-4B-Instruct-2507 \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 6144 \
  --prefix-repetition-suffix-len 128 \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len 32 \
  --num-prompts 50 --request-rate 0.5 --max-concurrency 1
```

Run this once on Node A (warm Redis), then on Node B (shared hit).

### 8.3 Example comparison

| Scenario | Instance A TTFT | Instance B TTFT | Notes |
|----------|------------------|------------------|-------|
| No caching (prefix caching off) | ~600ms | ~600ms | Both cold |
| LMCache shared (prefix caching off) | ~600ms | ~150ms | B hits Redis |
| vLLM native only (per node) | ~120ms | ~120ms | No sharing, GPU-only |

## 9. Optimizing Performance

- **Co-locate Redis** with vLLM nodes (same region or cluster) to minimize latency.
- **Keep local CPU hot cache enabled** to avoid Redis hits on repeated requests per node.
- **Use smaller `chunk_size`** (128) for better partial reuse if prompts vary slightly.
- **Enable `use_layerwise`** after validation to overlap KV load with compute.
- **Size Redis memory** to avoid evictions; use `allkeys-lru` for predictable eviction.
- **Set `PYTHONHASHSEED=0`** on all nodes for deterministic chunk hashing.

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| No cross-node hits | Different `chunk_size` | Use identical YAML on all nodes |
| Redis connection errors | Network/firewall | Open port 6379 and verify DNS |
| Cache hit rate low | Prompts not identical | Normalize system prompts and tokenization |
| Redis OOM | Max memory too small | Increase memory or set eviction policy |
| Hits but slow | Redis far from nodes | Co-locate or add local CPU hot cache |
| Cache collisions across models | Same Redis DB | Use separate DBs or separate Redis instances |

## 11. Additional Resources

- Single-node CPU hot cache: `recipes/dense_instruct_cpu_hot_cache.md`
- Redis backend details: `recipes/vllm_redis_remote.md`
- Multi-instance sharing (2 nodes): `recipes/vllm_multi_instance_sharing.md`
- Tiered storage (CPU + disk): `recipes/vllm_tiered_storage.md`
