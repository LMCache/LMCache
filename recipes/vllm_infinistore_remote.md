# LMCache + vLLM: InfiniStore Remote Backend

## 1. Introduction

**Target workload**
- High-performance remote KV cache with RDMA
- Accelerator clusters with fast interconnects
- Deployments already using InfiniStore

**LMCache mode**
- **Storage Mode**
- InfiniStore remote backend
- Local CPU hot cache enabled

This recipe shows how to use **InfiniStore** as the LMCache remote backend. InfiniStore provides an RDMA-enabled KV store optimized for high throughput and low latency in GPU clusters.

**Expected outcome**
- LMCache connects to InfiniStore using `infinistore://`
- KV data is stored remotely over RDMA
- Warm requests show LMCache hits

## 2. When to Use InfiniStore

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| RDMA-capable cluster | **InfiniStore** | High throughput, low latency |
| General remote cache | **Redis / Valkey** | Simpler operations |
| Cold persistence | **S3** | Durable storage |
| Single node cache | **Local CPU/Disk** | Lowest latency |

## 3. Installing vLLM + LMCache

Prerequisites:
- InfiniStore server deployed and reachable
- RDMA-capable devices (e.g., mlx5)
- InfiniStore Python client installed

Install vLLM and LMCache:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install lmcache vllm
```

Install InfiniStore dependencies per your cluster setup.

## 4. LMCache Configuration

Create `recipes/vllm_infinistore_remote.yaml`:

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 48
local_disk: false
# InfiniStore remote backend (RDMA)
remote_url: "infinistore://infinistore-master:12345?device=mlx5_0"
use_layerwise: false
save_unfull_chunk: true
extra_config:
  infinistore_link_type: "LINK_ETHERNET"
```

**InfiniStore config notes**
- `remote_url` uses `infinistore://host:port` with optional `?device=`.
- `infinistore_link_type` must match an InfiniStore enum (default `LINK_ETHERNET`).
- Only one host is supported in the URL.

## 5. Launching the vLLM Server (with LMCache)

```bash
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=recipes/vllm_infinistore_remote.yaml \
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
LMCache INFO: Loading LMCache config file recipes/vllm_infinistore_remote.yaml
LMCache INFO: Creating LMCacheEngine with config:
  {'remote_url': 'infinistore://infinistore-master:12345?device=mlx5_0', ...}
LMCache INFO: Initializing InfiniStore connector
```

## 7. Inference and Cache Validation

### 7.1 Cold request (first run)

```bash
python - <<'PY' | curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d @-
import json
prompt = "You are helpful.\n" + ("InfiniStore LMCache test. " * 400)
payload = {
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": prompt,
    "max_tokens": 32,
}
print(json.dumps(payload))
PY
```

Expected logs (cold, stores to InfiniStore):

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

- Ensure RDMA device name matches `?device=`.
- Set `infinistore_link_type` to match network (e.g., `LINK_ROCE`, `LINK_INFINIBAND`).
- Keep local CPU hot cache enabled to mask network latency.
- Use consistent `chunk_size` across vLLM instances.

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Import error | InfiniStore not installed | Install InfiniStore client library |
| Invalid link type | Wrong enum | Set `infinistore_link_type` to valid enum |
| Connection errors | RDMA not ready | Verify RDMA device and network config |
| Low hit rate | Prompt mismatch | Ensure identical prompts/tokenization |
| High latency | Remote-only tier | Enable local CPU hot cache |

## 11. Additional Resources

- LMCache connector formats: `lmcache/v1/storage_backend/connector/__init__.py`
- Redis remote backend: `recipes/vllm_redis_remote.md`
- MooncakeStore backend: `recipes/vllm_mooncake_remote.md`
