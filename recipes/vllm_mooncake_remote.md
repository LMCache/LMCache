# LMCache + vLLM: MooncakeStore Remote Backend

## 1. Introduction

**Target workload**
- Distributed KV persistence across nodes
- High-performance remote cache with RDMA-capable fabrics
- Deployments that already use MooncakeStore

**LMCache mode**
- **Storage Mode**
- MooncakeStore remote backend
- Local CPU hot cache enabled

This recipe shows how to use **MooncakeStore** as the LMCache remote backend. MooncakeStore provides a distributed KV store optimized for high throughput and low latency in accelerator clusters.

**Expected outcome**
- LMCache connects to MooncakeStore using `mooncakestore://`
- KV data is stored remotely with MooncakeStore
- Warm requests show LMCache hits

## 2. When to Use MooncakeStore

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| High-performance distributed cache | **MooncakeStore** | Designed for fast remote KV |
| General-purpose remote cache | **Redis / Valkey** | Simpler operations |
| Cloud cold tier | **S3** | Durable, large-scale persistence |
| Single node cache | **Local CPU/Disk** | Lowest latency |

## 3. Installing vLLM + LMCache

Prerequisites:
- MooncakeStore installed and configured in the cluster
- MooncakeStore client library installed (see Mooncake build docs)
- Network connectivity between vLLM nodes and MooncakeStore endpoints

Install vLLM and LMCache:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install lmcache vllm
```

Install Mooncake dependencies following:

- https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md

## 4. LMCache Configuration

Create `recipes/vllm_mooncake_remote.yaml`:

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 48
local_disk: false
# MooncakeStore remote backend
remote_url: "mooncakestore://mooncake-master:50051?device=mlx5_0"
use_layerwise: false
save_unfull_chunk: true
extra_config:
  # Required Mooncake settings
  local_hostname: "node-a"
  metadata_server: "mooncake-meta:50050"
  master_server_address: "mooncake-master:50051"
  # Optional tuning
  global_segment_size: 3355443200
  local_buffer_size: 1073741824
  protocol: "tcp"
  device_name: ""
  transfer_timeout: 1
  storage_root_dir: "/var/lib/mooncake"
  mooncake_prefer_local_alloc: false
```

**Mooncake config notes**
- `remote_url` uses `mooncakestore://host:port` and optional `?device=`.
- `device_name` in `extra_config` is overridden by the `device` query param if set.
- You can also provide Mooncake settings via `MOONCAKE_CONFIG_PATH` (JSON) instead of `extra_config`.

## 5. Launching the vLLM Server (with LMCache)

```bash
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=recipes/vllm_mooncake_remote.yaml \
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
LMCache INFO: Loading LMCache config file recipes/vllm_mooncake_remote.yaml
LMCache INFO: Creating LMCacheEngine with config:
  {'remote_url': 'mooncakestore://mooncake-master:50051?device=mlx5_0', ...}
LMCache INFO: MooncakeConnector initialized successfully.
```

## 7. Inference and Cache Validation

### 7.1 Cold request (first run)

```bash
python - <<'PY' | curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d @-
import json
prompt = "You are helpful.\n" + ("Mooncake LMCache test. " * 400)
payload = {
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": prompt,
    "max_tokens": 32,
}
print(json.dumps(payload))
PY
```

Expected logs (cold, stores to MooncakeStore):

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

- Use RDMA-capable devices and set `device` in the URL.
- Enable `mooncake_prefer_local_alloc` to favor local segments when possible.
- Keep local CPU hot cache enabled to reduce remote fetches.
- Ensure `chunk_size` is identical across vLLM instances.

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Import error | Mooncake not installed | Install Mooncake client library |
| Missing config | Required extra_config fields absent | Provide `local_hostname`, `metadata_server`, `master_server_address` |
| Connection errors | Wrong endpoints | Verify host/port and network access |
| Low hit rate | Prompt mismatch | Ensure identical prompts/tokenization |
| High latency | Remote-only tier | Enable local CPU hot cache |

## 11. Additional Resources

- Mooncake build docs: https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md
- LMCache connector formats: `lmcache/v1/storage_backend/connector/__init__.py`
- Redis remote backend: `recipes/vllm_redis_remote.md`
