# LMCache + vLLM: S3 Remote Backend (Cold Tier)

## 1. Introduction

**Target workload**
- Very large KV working sets that exceed local disks
- Persistent cache across restarts or node replacement
- Cloud environments with S3-compatible object storage

**LMCache mode**
- **Storage Mode**
- Local CPU hot cache + S3 cold tier
- KV persisted in object storage

This recipe shows how to use **S3 as a cold tier** for LMCache. S3 is slower than local or Redis tiers, but it provides durable, large-scale persistence at relatively low cost.

**Expected outcome**
- LMCache stores chunks to S3 after local CPU hot cache
- Cold requests fetch KV from S3 instead of recomputing
- Cache survives pod or node restarts

## 2. When to Use S3

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Very large cache, persistence required | **S3 backend** | Durable storage with large capacity |
| Low latency required | **Redis / local tiers** | S3 adds higher latency |
| Spot instances or autoscaling | **S3 backend** | KV survives node termination |
| Cost-sensitive cold cache | **S3 backend** | Storage is cheaper than RAM |

## 3. Installing vLLM + LMCache

Prerequisites:
- S3 bucket created
- IAM credentials or instance role with read/write access
- Network access to S3 endpoint

Install vLLM and LMCache:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install lmcache vllm
```

## 4. LMCache Configuration

Create `recipes/vllm_s3_remote.yaml`:

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 48
local_disk: false
# S3 cold tier (persistent)
remote_url: "s3://lmcache-bucket.s3.us-east-1.amazonaws.com"
use_layerwise: false
save_unfull_chunk: true
extra_config:
  # S3 connector settings (required)
  s3_region: "us-east-1"
  # S3 connector settings (optional)
  s3_num_io_threads: 64
  s3_prefer_http2: true
  s3_enable_s3express: false
  disable_tls: false
  # Required: S3 backend does not support chunk metadata
  save_chunk_meta: false
  # Credentials (optional when using IAM role/instance profile)
  aws_access_key_id: ""
  aws_secret_access_key: ""
```

**S3 URL formats**
- Standard: `s3://<bucket>.s3.<region>.amazonaws.com`
- S3 Express: `s3://<bucket>.s3express-<az>.<region>.amazonaws.com`

**Important constraints**
- `save_chunk_meta` must be **false** for S3.
- Use IAM roles where possible; only set access keys if required.

## 5. Launching the vLLM Server (with LMCache)

```bash
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=recipes/vllm_s3_remote.yaml \
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
LMCache INFO: Loading LMCache config file recipes/vllm_s3_remote.yaml
LMCache INFO: Creating LMCacheEngine with config:
  {'remote_url': 's3://lmcache-bucket.s3.us-east-1.amazonaws.com', ...}
LMCache INFO: Initializing S3 backend
```

## 7. Inference and Cold-Tier Validation

### 7.1 Cold request (first run)

```bash
python - <<'PY' | curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d @-
import json
prompt = "You are helpful.\n" + ("LMCache cold tier test. " * 400)
payload = {
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": prompt,
    "max_tokens": 32,
}
print(json.dumps(payload))
PY
```

Expected logs (cold, stores to S3):

```
LMCache INFO: Reqid: ..., Total tokens 2000, LMCache hit tokens: 0, need to load: 0
LMCache INFO: Stored 1792 out of total 1792 tokens. size: 0.2461 GB
```

### 7.2 Restart to prove persistence

Restart the vLLM process, then re-run the same request. You should see LMCache hits even after restart because S3 is persistent.

Expected logs (cold-tier hit):

```
LMCache INFO: Reqid: ..., Total tokens 2000, LMCache hit tokens: 1792, need to load: 1792
LMCache INFO: Retrieved 1792 out of 1792 required tokens. size: 0.2461 gb
```

## 8. Benchmarking

Use `prefix_repetition` and compare cold vs warm:

```bash
vllm bench serve --model Qwen/Qwen3-4B-Instruct-2507 \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 6144 \
  --prefix-repetition-suffix-len 128 \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len 32 \
  --num-prompts 50 --request-rate 0.5 --max-concurrency 1
```

Run twice to compare compute vs S3 retrieval. Expect higher latency than Redis but lower than full recompute.

## 9. Cost and Performance Considerations

- **Latency**: S3 cold hits are slower than Redis or disk; keep a CPU hot tier enabled.
- **Cost**: Request costs apply (PUT/GET). Large cache churn increases cost.
- **Throughput**: Use `s3_num_io_threads` to tune concurrency for higher throughput.
- **Durability**: S3 provides persistence across node or pod restarts.

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Config error on startup | `save_chunk_meta` not false | Set `extra_config.save_chunk_meta: false` |
| Authentication error | Missing credentials | Configure IAM role or keys |
| Slow retrievals | S3 latency | Keep CPU hot cache or add disk tier |
| Region mismatch | Wrong `s3_region` | Ensure region matches bucket |
| TLS errors | Custom endpoint | Set `disable_tls: true` only if required |

## 11. Additional Resources

- S3 connector formats: `lmcache/v1/storage_backend/connector/__init__.py`
- LMCache config guide: `docs/source/api_reference/configurations.rst`
- Redis remote backend: `recipes/vllm_redis_remote.md`
- Disk persistence: `recipes/vllm_disk_persistence.md`
