# LMCache + vLLM: Valkey Remote Backend (Redis-Compatible)

## 1. Introduction

**Target workload**
- Redis-compatible deployments that prefer Valkey
- Multi-instance vLLM sharing a remote KV cache
- Environments needing a drop-in Redis alternative

**LMCache mode**
- **Storage Mode**
- Remote Valkey backend
- Local CPU hot cache enabled

This recipe shows how to use **Valkey** as the LMCache remote backend. Valkey is Redis-compatible, so you can swap it in with minimal changes while still enabling cross-instance KV reuse.

**Expected outcome**
- LMCache connects to Valkey via `valkey://` URL
- Cache hits occur on warm requests
- Behavior mirrors Redis backend with compatible URLs and auth

## 2. When to Use Valkey

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Redis already deployed | **Redis backend** | No migration needed |
| Prefer Redis-compatible OSS fork | **Valkey backend** | Drop-in replacement with LMCache support |
| Need sharded cache | **Valkey cluster mode** | Horizontal scale-out |
| Need HA only | **Redis Sentinel** | Built-in failover pattern |

## 3. Installing vLLM + LMCache

Prerequisites:
- Valkey server running and reachable from vLLM hosts
- Network access to Valkey port (default 6379)

Install vLLM and LMCache:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install lmcache vllm
```

## 4. LMCache Configuration

Create `recipes/vllm_valkey_remote.yaml`:

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 48
local_disk: false
# Valkey remote backend (Redis-compatible)
remote_url: "valkey://valkey-1:6379"
use_layerwise: false
save_unfull_chunk: true
extra_config:
  valkey_username: ""
  valkey_password: ""
  valkey_database: 0
  valkey_mode: "standalone"
```

**Valkey config notes**
- `remote_url` uses `valkey://host:port`.
- Authentication and database are configured via `extra_config`.
- For cluster mode, set `valkey_mode: "cluster"` and list multiple nodes in `remote_url`:
  - `valkey://node-1:7000,node-2:7000,node-3:7000`

## 5. Launching the vLLM Server (with LMCache)

```bash
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=recipes/vllm_valkey_remote.yaml \
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
LMCache INFO: Loading LMCache config file recipes/vllm_valkey_remote.yaml
LMCache INFO: Creating LMCacheEngine with config:
  {'remote_url': 'valkey://valkey-1:6379', ...}
LMCache INFO: Initializing Valkey backend
```

## 7. Inference and Cache Validation

### 7.1 Cold request (first run)

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

Expected logs (cold, stores to Valkey):

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

- Keep local CPU hot cache enabled to avoid remote latency on repeat requests.
- Use consistent `chunk_size` across all instances.
- For cluster mode, list multiple startup nodes in `remote_url`.
- Size Valkey memory to fit the shared working set with headroom.

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Connection errors | Wrong URL or port | Verify `valkey://host:port` and connectivity |
| Auth failed | Missing credentials | Set `valkey_username`/`valkey_password` in `extra_config` |
| No cache hits | Different `chunk_size` | Use identical YAML on all instances |
| Cluster mode errors | Wrong mode | Set `valkey_mode: "cluster"` and list nodes |
| High latency | Remote tier only | Enable local CPU hot cache |

## 11. Additional Resources

- Valkey project: https://valkey.io/
- Redis standalone recipe: `recipes/vllm_redis_remote.md`
- Redis Sentinel recipe: `recipes/vllm_redis_sentinel.md`
- Redis Cluster recipe: `recipes/vllm_redis_cluster.md`
