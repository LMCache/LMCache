# LMCache + vLLM: PD Tuning (Buffer Device + Buffer Size)

## 1. Introduction

**Target workload**
- PD (prefill/decode) deployments with unstable TTFT/ITL
- Long-context workloads that pressure transfer buffers
- GPUs with limited memory headroom

**LMCache mode**
- **Transport Mode (PD)**
- Buffer tuning for sender/receiver
- NIXL transfer channel

This recipe focuses on **PD buffer tuning** to stabilize time-to-first-token (TTFT) and inter-token latency (ITL) by choosing the right **buffer device** (`cuda` vs `cpu`) and **buffer size**.

**Expected outcome**
- Fewer PD transfer timeouts
- Lower variance in TTFT/ITL
- Predictable throughput under long prompts

## 2. When to Use PD Buffer Tuning

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| TTFT spikes under long prompts | **Increase buffer size** | Avoid transfer stalls |
| CUDA OOM on decode | **Use CPU buffer** | Reduce GPU memory pressure |
| Fast interconnect (NVLink/RDMA) | **CUDA buffer** | Best transfer speed |
| Ethernet-only networks | **CPU buffer** | Avoid GPU staging overhead |

## 3. Installing vLLM + LMCache

Prerequisites:
- PD mode already working (see R-021 or R-022)
- NIXL installed and configured

Install vLLM and LMCache:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install lmcache vllm
```

## 4. LMCache Configuration

Create `recipes/vllm_pd_tuning.yaml`:

```yaml
# PD tuning base config (adjust role and ports per instance)
local_cpu: false
local_disk: false

enable_pd: true
transfer_channel: "nixl"
pd_role: "sender"  # change to "receiver" for decode

# Sender (prefill) proxy config
pd_proxy_host: "localhost"
pd_proxy_port: 7500

# Receiver (decode) peer config (uncomment when pd_role: receiver)
# pd_peer_host: "localhost"
# pd_peer_init_port: 7300
# pd_peer_alloc_port: 7400

# Buffer tuning (primary knobs)
pd_buffer_device: "cuda"  # "cuda" (fastest) or "cpu" (lower GPU memory)
pd_buffer_size: 2147483648  # 2GB

# NIXL backend settings
nixl_backends: [UCX]
```

**Key tuning knobs**
- `pd_buffer_device`: choose `cuda` for fastest transfer, `cpu` to reduce GPU memory pressure.
- `pd_buffer_size`: size in bytes; must be large enough to hold the largest KV transfer.

### Buffer sizing formula

Use this rough estimate:

```
buffer_size_bytes = tokens × num_layers × 2 × head_dim × num_heads × bytes_per_element
```

Example (8K tokens, 32 layers, fp16, head_dim 128, 32 heads):

```
8,192 × 32 × 2 × 128 × 32 × 2 bytes ≈ 4.3 GB
```

## 5. Launching the Server (with LMCache)

Use the same launch commands from R-021 or R-022, but swap in the tuning config:

```bash
PYTHONHASHSEED=0 UCX_TLS=cuda_ipc,cuda_copy,tcp \
LMCACHE_CONFIG_FILE=recipes/vllm_pd_tuning.yaml \
VLLM_ENABLE_V1_MULTIPROCESSING=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
CUDA_VISIBLE_DEVICES=0 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 7100 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer"}'
```

## 6. Startup Validation

Expected logs show PD buffer settings:

```
LMCache INFO: PD buffer device: cuda
LMCache INFO: PD buffer size: 2147483648 bytes
```

## 7. Inference and Cache Validation

Send a long prompt through the PD proxy and confirm no transfer errors:

```bash
curl http://localhost:9100/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Explain PD buffer sizing." ,
    "max_tokens": 128
  }'
```

If logs show transfer timeouts, increase `pd_buffer_size` or switch to `cuda`.

## 8. Benchmarking

### Baseline (default buffers)

```bash
vllm bench serve --port 9100 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 6000 \
  --random-output-len 128 \
  --num-prompts 20
```

### Tuned buffers

Repeat after adjusting `pd_buffer_device` and `pd_buffer_size`.

### Example comparison

| Setting | TTFT P50 | ITL P95 | Notes |
|---------|----------|---------|-------|
| 1GB CPU | 520ms | 35ms | OOM-safe, slower |
| 2GB CUDA | 360ms | 22ms | Balanced |
| 4GB CUDA | 330ms | 18ms | Best for 8K+ prompts |

## 9. Optimizing Performance

- Prefer `cuda` buffers when GPU memory allows.
- Increase buffer size for long prompts and high batch sizes.
- Use `UCX_TLS=cuda_ipc,cuda_copy,tcp` for best single-node performance.
- Keep `--enforce-eager` and `--no-enable-prefix-caching` in PD mode.

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Transfer timeout | Buffer too small | Increase `pd_buffer_size` |
| CUDA OOM | GPU buffer too large | Switch to `pd_buffer_device: cpu` |
| High TTFT variance | Buffer device mismatch | Use `cuda` if RDMA/NVLink available |
| Slow transfers | UCX misconfigured | Ensure `UCX_TLS` includes cuda transports |
| Request hangs | Roles mismatched | Verify `pd_role` sender/receiver |

## 11. Additional Resources

- Single-node PD: `recipes/vllm_pd_single_node.md`
- Multi-node PD: `recipes/vllm_pd_multi_node.md`
- PD buffer settings: `recipes/vllm_pd_prefiller.yaml`, `recipes/vllm_pd_decoder.yaml`
