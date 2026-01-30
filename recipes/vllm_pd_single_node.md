# LMCache + vLLM: Disaggregated Prefill-Decode (PD) on Single Node

## 1. Introduction

**Target workload**
- High-throughput serving with long context
- Separation of prefill and decode phases for resource optimization
- Reduced TTFT for concurrent requests
- **Specialized roles: prefill instances compute KV, decode instances generate tokens**

**LMCache mode**
- **Transport Mode (PD)**
- Single node with multiple GPUs
- NIXL channel for KV transfer

This recipe demonstrates **Disaggregated Prefill-Decode (PD)**, a specialized LMCache mode where:

1. **Prefill instances** compute KV cache (process input tokens)
2. **Decode instances** receive KV via NIXL and generate output tokens
3. **NIXL (NVIDIA Inference Xfer Library)** enables fast GPU-to-GPU KV transfer

> **Important:** PD mode is fundamentally different from Storage Mode:
> - Storage Mode (R-001, R-010): KV is stored and retrieved from backends (CPU/disk/Redis)
> - PD Mode: KV is transferred directly from prefill to decode GPUs in real-time
> - PD mode **cannot** be combined with Storage Mode backends

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                         Single Node                          │
│  ┌──────────────┐        NIXL         ┌──────────────┐     │
│  │   Prefill    │  ┌───────────────┐  │    Decode    │     │
│  │   Instance   │──│  GPU-to-GPU   │──│   Instance   │     │
│  │   (GPU 0)    │  │  KV Transfer  │  │   (GPU 1)    │     │
│  │  Port 7100   │  └───────────────┘  │  Port 7200   │     │
│  └──────┬───────┘                     └──────┬───────┘     │
│         │                                     │             │
│         └──────────────┬──────────────────────┘             │
│                        │                                    │
│                 ┌──────▼──────┐                            │
│                 │    Proxy    │                            │
│                 │   Server    │                            │
│                 │  Port 9100  │                            │
│                 └─────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

**Flow:**
1. Client sends request to Proxy Server (port 9100)
2. Proxy routes to Prefill instance for KV computation
3. Prefill transfers KV to Decode instance via NIXL
4. Decode generates tokens and returns response

**Expected outcome**
- Prefill and decode run on separate GPUs
- KV is transferred via NIXL (not stored)
- Lower TTFT for concurrent requests compared to combined serving

## 2. When to Use PD Mode

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Long context + high concurrency | **PD Mode** | Prefill and decode don't block each other |
| GPU resource specialization | **PD Mode** | Different GPUs optimized for different phases |
| Simple/low-latency workloads | **Storage Mode** (R-001) | PD has coordination overhead |
| Need cache persistence | **Storage Mode** (R-007) | PD doesn't store KV |
| Cross-request cache reuse | **Storage Mode** (R-010) | PD is request-to-request only |

**Key differences:**

| Feature | Storage Mode | PD Mode |
|---------|--------------|---------|
| KV storage | CPU/disk/Redis | GPU-to-GPU only |
| Cache reuse | Across requests/time | Request-to-request only |
| Persistence | Yes | No |
| Use case | Cache sharing, persistence | Throughput optimization |
| Backend | Redis, disk, etc. | NIXL only |

## 3. Installing vLLM + LMCache + NIXL

### 3.1 Hardware Prerequisites
- **2+ GPUs** (one for prefill, one for decode)
- Same node (single-node PD)

### 3.2 Software
```bash
# Install LMCache
pip install lmcache

# Install vLLM
pip install vllm

# Install NIXL (required for PD mode)
pip install nixl
```

### 3.3 Verify NIXL installation
```bash
python -c "import nixl; print('NIXL version:', nixl.__version__)"
```

## 4. LMCache Configuration

### 4.1 Prefill Instance Configuration

Create `recipes/vllm_pd_prefiller.yaml`:

```yaml
# Disable storage mode
local_cpu: False
local_disk: False

# Enable PD mode
enable_pd: True
transfer_channel: "nixl"
pd_role: "sender"

# Proxy configuration
pd_proxy_host: "localhost"
pd_proxy_port: 7500

# Buffer configuration
pd_buffer_size: 1073741824  # 1GB
pd_buffer_device: "cuda"     # or "cpu"

# NIXL configuration
nixl_backends: [UCX]
```

### 4.2 Decode Instance Configuration

Create `recipes/vllm_pd_decoder.yaml`:

```yaml
# Disable storage mode
local_cpu: False
local_disk: False

# Enable PD mode
enable_pd: True
transfer_channel: "nixl"
pd_role: "receiver"

# Peer configuration (for multi-node)
pd_peer_host: "localhost"
pd_peer_init_port: 7300
pd_peer_alloc_port: 7400

# Buffer configuration
pd_buffer_size: 2147483648  # 2GB (larger for decode)
pd_buffer_device: "cuda"     # or "cpu"

# NIXL configuration
nixl_backends: [UCX]
```

> **Buffer sizing:** Decode typically needs larger buffers than prefill because it receives all KV at once.

## 5. Launching PD Instances

### 5.1 Start Prefill Instance (Port 7100)

```bash
export PYTHONHASHSEED=0

UCX_TLS=cuda_ipc,cuda_copy,tcp \
LMCACHE_CONFIG_FILE=recipes/vllm_pd_prefiller.yaml \
VLLM_ENABLE_V1_MULTIPROCESSING=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
CUDA_VISIBLE_DEVICES=0 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 7100 \
  --disable-log-requests \
  --enforce-eager \
  --no-enable-prefix-caching \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer1"}}'
```

**Environment variables explained:**
- `UCX_TLS`: NIXL transport layers (cuda_ipc for GPU-to-GPU, tcp for fallback)
- `VLLM_ENABLE_V1_MULTIPROCESSING`: Required for vLLM v1
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`: Required for compatibility

### 5.2 Start Decode Instance (Port 7200)

In a separate terminal:

```bash
export PYTHONHASHSEED=0

UCX_TLS=cuda_ipc,cuda_copy,tcp \
LMCACHE_CONFIG_FILE=recipes/vllm_pd_decoder.yaml \
VLLM_ENABLE_V1_MULTIPROCESSING=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
CUDA_VISIBLE_DEVICES=1 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 7200 \
  --disable-log-requests \
  --enforce-eager \
  --no-enable-prefix-caching \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1", "skip_last_n_tokens": 1}}'
```

### 5.3 Start Proxy Server

The proxy coordinates between prefill and decode:

```bash
# From LMCache examples or create your own
python examples/disagg_prefill/disagg_proxy_server.py \
  --prefillHost localhost \
  --prefillPort 7100 \
  --decodeHost localhost \
  --decodePort 7200 \
  --port 9100
```

Or use a simple FastAPI proxy (see Additional Resources).

## 6. Startup Validation

### Prefill instance logs
```
LMCache INFO: Loading LMCache config file recipes/vllm_pd_prefiller.yaml
LMCache INFO: LMCacheEngine initialized with PD mode (sender)
LMCache INFO: NIXL backend initialized with UCX
LMCache INFO: Connected to proxy at localhost:7500
```

### Decode instance logs
```
LMCache INFO: Loading LMCache config file recipes/vllm_pd_decoder.yaml
LMCache INFO: LMCacheEngine initialized with PD mode (receiver)
LMCache INFO: NIXL backend initialized with UCX
LMCache INFO: Waiting for KV transfer on port 7300/7400
```

### Verify proxy connection
```bash
# Check all services are listening
ss -tlnp | grep -E "7100|7200|9100"

# Expected output shows all three ports
```

## 7. Inference and PD Validation

### 7.1 Send request through proxy

```bash
# Request goes to proxy (port 9100), which coordinates prefill and decode
curl http://localhost:9100/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Explain the benefits of disaggregating prefill and decode phases in large language model serving. This is a long prompt to demonstrate KV transfer via NIXL.",
    "max_tokens": 100
  }'
```

### 7.2 Expected behavior

**Prefill instance logs:**
```
LMCache INFO: Received request for prefill
LMCache INFO: Computing KV cache for 150 tokens
LMCache INFO: Transferring KV to decode instance via NIXL
LMCache INFO: KV transfer complete, 0.123 GB in 15.5 ms
```

**Decode instance logs:**
```
LMCache INFO: Received KV from prefill instance
LMCache INFO: KV shape: [32, 2, 150, 128, 128]
LMCache INFO: Starting token generation
```

## 8. Benchmarking

### 8.1 Baseline (non-PD)

```bash
# Single vLLM instance handling both prefill and decode
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

# Benchmark
vllm bench serve --port 8000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 7500 \
  --random-output-len 200 \
  --num-prompts 30
```

### 8.2 With PD mode

```bash
# Run benchmark through proxy
vllm bench serve --port 9100 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 7500 \
  --random-output-len 200 \
  --num-prompts 30 \
  --burstiness 100
```

### 8.3 Expected results

| Metric | Non-PD | PD Mode | Improvement |
|--------|--------|---------|-------------|
| Mean TTFT | ~800ms | ~310ms | **~60% faster** |
| Throughput | ~5000 tok/s | ~7300 tok/s | **~45% higher** |
| GPU utilization | Single GPU bottleneck | Separated concerns | Better efficiency |

> **Note:** PD benefits are most pronounced with long input contexts and high concurrency.

## 9. PD Configuration Tuning

### 9.1 Buffer device selection

```yaml
# Option A: CUDA buffer (fastest for GPU-to-GPU)
pd_buffer_device: "cuda"

# Option B: CPU buffer (if GPU memory is constrained)
pd_buffer_device: "cpu"
```

### 9.2 Buffer sizing

```yaml
# For long contexts (8K+ tokens)
pd_buffer_size: 4294967296  # 4GB

# For shorter contexts
pd_buffer_size: 1073741824  # 1GB
```

**Formula:** `buffer_size = max_tokens × num_layers × 2 × hidden_size × 4_bytes`

### 9.3 NIXL backends

```yaml
# Single node with NVLink
nixl_backends: [UCX]

# Multi-node (see R-022)
nixl_backends: [UCX, TCP]
```

## 10. Performance Tips

| Optimization | Configuration | Benefit |
|--------------|---------------|---------|
| NVLink | `UCX_TLS=cuda_ipc` | Fastest GPU-to-GPU transfer |
| Larger decode buffer | `pd_buffer_size: 4GB` | Handle longer contexts |
| CUDA buffers | `pd_buffer_device: cuda` | Avoid CPU copy overhead |
| Eager mode | `--enforce-eager` | PD stability (required) |
| Disable prefix caching | `--no-enable-prefix-caching` | Avoid conflicts with PD |

## 11. Troubleshooting / Common Pitfalls

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| NIXL import error | NIXL not installed | `pip install nixl` |
| UCX connection failed | GPU topology issue | Check `nvidia-smi topo -m` |
| Transfer timeout | Buffer too small | Increase `pd_buffer_size` |
| Proxy connection refused | Proxy not running | Start proxy before vLLM |
| CUDA OOM on decode | Buffer in GPU memory | Use `pd_buffer_device: cpu` |
| Slow transfer | Using TCP instead of cuda_ipc | Verify `UCX_TLS` includes `cuda_ipc` |
| Request hangs | Roles mismatched | Check `pd_role` (sender vs receiver) |

### Debug NIXL connectivity

```bash
# Check GPU topology
nvidia-smi topo -m

# Verify NVLink (if available)
nvidia-smi nvlink --status

# Test UCX directly
ucx_info -d
cuda-memcheck  # If CUDA errors
```

### Common UCX_TLS settings

```bash
# Single node with NVLink
export UCX_TLS=cuda_ipc,cuda_copy,tcp

# Single node without NVLink
export UCX_TLS=cuda_copy,tcp

# Multi-node (see R-022)
export UCX_TLS=tcp,sockcm
```

## 12. PD vs Storage Mode: Decision Guide

Choose **PD Mode** when:
- ✅ Long context prompts (4K+ tokens)
- ✅ High concurrent request volume
- ✅ Separate GPU resources available
- ✅ Optimize for throughput over latency
- ✅ No need for cache persistence

Choose **Storage Mode** (R-001, R-010) when:
- ✅ Cache reuse across requests is important
- ✅ Need cache persistence
- ✅ Simpler deployment preferred
- ✅ Short to medium prompts
- ✅ Single GPU or shared GPU

## 13. Limitations and Constraints

1. **No persistence:** KV is not stored, only transferred
2. **Request-to-request only:** Cannot reuse KV from previous requests
3. **Resource requirements:** Requires 2+ GPUs
4. **Eager mode required:** `--enforce-eager` is mandatory
5. **No prefix caching:** Must disable with `--no-enable-prefix-caching`
6. **Cannot combine with Storage Mode:** PD and Storage are mutually exclusive

## 14. Additional Resources
- Multi-node PD recipe: `recipes/vllm_pd_multi_node.md` (R-022)
- PD tuning guide: `recipes/vllm_pd_tuning.md` (R-023)
- CPU hot cache: `recipes/dense_instruct_cpu_hot_cache.md` (R-001)
- Example scripts: `examples/disagg_prefill/1p1d/`
