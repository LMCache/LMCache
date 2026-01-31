# LMCache + vLLM: Redis Cluster (Sharded) Remote Backend

## 1. Introduction

**Target workload**
- Large shared KV cache that outgrows a single Redis node
- Multi-instance vLLM deployments requiring horizontal cache scaling
- Production environments needing sharding + high availability

**LMCache mode**
- **Storage Mode**
- Remote Redis Cluster backend (sharded)
- Local CPU hot cache for fast hits

This recipe shows how to use **Redis Cluster** as the LMCache remote backend so KV data is **sharded across multiple Redis primaries** for capacity and throughput.

**Expected outcome**
- LMCache connects to Redis Cluster via startup nodes
- Cache keys are distributed across shards
- vLLM instances read/write KV via the cluster

## 2. When to Use Redis Cluster

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Single shared cache node | **Redis standalone** | Lowest operational overhead |
| Need HA only | **Redis Sentinel** | Failover without sharding |
| Need scale-out capacity | **Redis Cluster** | Shards KV across primaries |
| Heavy cross-instance sharing | **Redis Cluster + local CPU** | Local hot hits + shared shard tier |

## 3. Installing vLLM + LMCache

Prerequisites:
- Redis Cluster up and reachable from vLLM nodes
- 3+ Redis primaries (with replicas recommended)
- Network access to all Redis cluster ports (e.g., 7000-7002)

Install vLLM and LMCache:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install lmcache vllm
```

## 4. LMCache Configuration

Create `recipes/vllm_redis_cluster.yaml`:

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 48
local_disk: false
# Redis Cluster (sharded)
remote_url: "redis-cluster://redis-0:7000,redis-1:7000,redis-2:7000"
use_layerwise: false
save_unfull_chunk: true
```

**Redis Cluster URL format**
- `redis-cluster://host1:port1,host2:port2,...`
- Include multiple startup nodes for resilience.

## 5. Launching the vLLM Server (with LMCache)

```bash
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=recipes/vllm_redis_cluster.yaml \
CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --port 8000 \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

## 6. Startup Validation

Expected vLLM logs:

```
LMCache INFO: Loading LMCache config file recipes/vllm_redis_cluster.yaml
LMCache INFO: Creating LMCacheEngine with config:
  {'remote_url': 'redis-cluster://redis-0:7000,redis-1:7000,redis-2:7000', ...}
LMCache INFO: Initializing Redis Cluster backend
```

Verify cluster status:

```bash
redis-cli -c -h redis-0 -p 7000 cluster info
redis-cli -c -h redis-0 -p 7000 cluster nodes
```

## 7. Inference and Cache Validation

### 7.1 Warm cache on vLLM instance

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

Expected logs (cold, stores to cluster):

```
LMCache INFO: Reqid: ..., Total tokens 2000, LMCache hit tokens: 0, need to load: 0
LMCache INFO: Stored 1792 out of total 1792 tokens. size: 0.2461 GB
```

### 7.2 Warm request (cache hit)

Repeat the request to confirm cluster hits:

```
LMCache INFO: Reqid: ..., Total tokens 2000, LMCache hit tokens: 1792, need to load: 1792
LMCache INFO: Retrieved 1792 out of 1792 required tokens. size: 0.2461 gb
```

### 7.3 Validate sharding

Check which node holds a cache key (example):

```bash
redis-cli -c -h redis-0 -p 7000 cluster keyslot "<some-lmcache-key>"
```

Use `cluster nodes` output to map slots to primaries.

## 8. Benchmarking

Use `prefix_repetition` to highlight cache hits:

```bash
vllm bench serve --model Qwen/Qwen3-4B-Instruct-2507 \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 6144 \
  --prefix-repetition-suffix-len 128 \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len 32 \
  --num-prompts 50 --request-rate 0.5 --max-concurrency 1
```

Run twice to compare cold vs warm TTFT.

## 9. Optimizing Performance

- Use multiple startup nodes for faster reconnects.
- Co-locate Redis Cluster with vLLM instances to reduce latency.
- Enable local CPU hot cache for repeated requests per instance.
- Ensure `chunk_size` is identical across all vLLM instances.

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Connection errors | Ports blocked | Open Redis cluster ports (7000+ and bus ports) |
| MOVED errors in logs | Cluster misconfig | Ensure cluster is properly formed |
| No cache hits | Different `chunk_size` | Use identical YAML on all instances |
| Low throughput | Redis far from vLLM | Co-locate or add CPU hot cache |
| Authentication error | Missing credentials | Use `redis-cluster://user:password@host:port,...` |

## 11. Additional Resources

- Redis Cluster docs: https://redis.io/docs/latest/operate/oss_and_stack/management/scaling/
- Redis connector formats: `lmcache/v1/storage_backend/connector/__init__.py`
- Redis standalone recipe: `recipes/vllm_redis_remote.md`
- Redis Sentinel recipe: `recipes/vllm_redis_sentinel.md`
