# LMCache + vLLM: Redis Sentinel (HA) Remote Backend

## 1. Introduction

**Target workload**
- Multi-instance vLLM deployments needing Redis high availability
- Shared KV cache with automatic failover
- Production environments where Redis downtime is unacceptable

**LMCache mode**
- **Storage Mode**
- Remote Redis backend with Sentinel
- Local CPU hot cache for fast hits

This recipe shows how to use **Redis Sentinel** as the LMCache remote backend so vLLM instances can **survive Redis failovers** without losing cache connectivity.

**Expected outcome**
- LMCache connects to Sentinel instead of a single Redis node
- Cache hits continue after Redis master failover
- vLLM instances stay online during sentinel-driven failover

## 2. When to Use Redis Sentinel

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Single Redis node | **Redis standalone** | Simpler and lower operational overhead |
| Redis HA required | **Redis Sentinel** | Auto-failover with minimal client changes |
| Multi-AZ deployment | **Redis Sentinel** | Survive node loss and failover events |
| Heavy sharding needs | **Redis Cluster** | Shards data across multiple primaries |

## 3. Installing vLLM + LMCache

Prerequisites:
- 1 Redis master, 1+ Redis replicas
- 3 Sentinel nodes (recommended) for quorum
- Network access from vLLM hosts to Sentinel ports (default 26379)

Install vLLM and LMCache:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install lmcache vllm
```

## 4. LMCache Configuration

Create `recipes/vllm_redis_sentinel.yaml`:

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 48
local_disk: false
# Redis Sentinel (HA)
remote_url: "redis-sentinel://sentinel-1:26379,sentinel-2:26379,sentinel-3:26379/mymaster"
use_layerwise: false
save_unfull_chunk: true
```

**Sentinel URL format**
- `redis-sentinel://[[username]:[password]@]host1:port1,host2:port2,.../service_name`
- Example with auth:
  - `redis-sentinel://user:password@sentinel-1:26379,sentinel-2:26379/mymaster`

The `service_name` must match the Sentinel master name (e.g., `mymaster`).

## 5. Launching the vLLM Server (with LMCache)

```bash
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=recipes/vllm_redis_sentinel.yaml \
CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --port 8000 \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

## 6. Startup Validation

Expected logs from vLLM:

```
LMCache INFO: Loading LMCache config file recipes/vllm_redis_sentinel.yaml
LMCache INFO: Creating LMCacheEngine with config:
  {'remote_url': 'redis-sentinel://sentinel-1:26379,sentinel-2:26379,sentinel-3:26379/mymaster', ...}
LMCache INFO: Initializing Redis Sentinel backend
```

Validate Sentinel knows the master:

```bash
redis-cli -h sentinel-1 -p 26379 SENTINEL get-master-addr-by-name mymaster
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

Expected logs (cold, stores to Redis):

```
LMCache INFO: Reqid: ..., Total tokens 2000, LMCache hit tokens: 0, need to load: 0
LMCache INFO: Stored 1792 out of total 1792 tokens. size: 0.2461 GB
```

### 7.2 Warm request (cache hit)

Repeat the request and confirm hits:

```
LMCache INFO: Reqid: ..., Total tokens 2000, LMCache hit tokens: 1792, need to load: 1792
LMCache INFO: Retrieved 1792 out of 1792 required tokens. size: 0.2461 gb
```

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

- Keep Sentinel nodes close to Redis primaries (same region/VPC).
- Enable local CPU hot cache to avoid network hits on repeat queries.
- Use consistent `chunk_size` across all vLLM instances.
- Consider Redis Cluster when you need sharding beyond a single master.

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Connection fails | Wrong Sentinel URL | Verify `redis-sentinel://.../service_name` |
| No master found | Wrong service name | Check `SENTINEL get-master-addr-by-name` |
| Failover not detected | Too few sentinels | Use 3 sentinels for quorum |
| Cache hits drop after failover | Client reconnect delay | Wait a few seconds; verify Redis replicas |
| Authentication error | Missing credentials | Use `user:password@` in URL |

### Failover validation (optional)

Trigger failover and verify LMCache continues serving:

```bash
redis-cli -h sentinel-1 -p 26379 SENTINEL failover mymaster
redis-cli -h sentinel-1 -p 26379 SENTINEL get-master-addr-by-name mymaster
```

Re-run the warm request and confirm LMCache hits after the new master is elected.

## 11. Additional Resources

- Redis Sentinel docs: https://redis.io/docs/latest/operate/oss_and_stack/management/sentinel/
- Redis connector formats: `lmcache/v1/storage_backend/connector/__init__.py`
- Redis standalone recipe: `recipes/vllm_redis_remote.md`
- Multi-instance sharing: `recipes/vllm_multi_instance_sharing.md`
