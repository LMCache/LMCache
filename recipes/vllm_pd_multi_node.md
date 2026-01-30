# LMCache + vLLM: Disaggregated Prefill-Decode (PD) Across Multiple Nodes

## 1. Introduction

**Target workload**
- Production-scale deployments with dedicated prefill and decode clusters
- Multi-node GPU clusters for large-scale LLM serving
- Separation of concerns: prefill nodes for compute, decode nodes for generation
- **Scalable prefill pools and decode pools**

**LMCache mode**
- **Transport Mode (PD)**
- Multi-node deployment
- NIXL channel with TCP transport for cross-node KV transfer

This recipe extends **R-021 (Single-node PD)** to **multi-node deployments** where:

1. **Prefill pool** - Multiple nodes dedicated to KV computation
2. **Decode pool** - Multiple nodes dedicated to token generation
3. **Cross-node NIXL transfer** - KV moves over RDMA or high-speed network
4. **Proxy server** - Routes requests and coordinates between pools

> **Prerequisites:** Complete R-021 (Single-node PD) first. Multi-node PD requires understanding of single-node concepts.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Node 1: Prefill Pool                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  Prefill Inst 1 │  │  Prefill Inst 2 │  │  Prefill Inst N │             │
│  │   (GPU 0,1)     │  │   (GPU 2,3)     │  │    (GPU...)     │             │
│  │   Port 7100     │  │   Port 7101     │  │    Port...      │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
└───────────┼────────────────────┼────────────────────┼──────────────────────┘
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 │ NIXL over Network
            ┌────────────────────┼────────────────────┐
            │                    │                    │
┌───────────┼────────────────────┼────────────────────┼──────────────────────┐
│           │        Node 2: Decode Pool               │                     │
│  ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐            │
│  │   Decode Inst 1 │  │   Decode Inst 2 │  │   Decode Inst N │            │
│  │    (GPU 0,1)    │  │    (GPU 2,3)    │  │    (GPU...)     │            │
│  │    Port 7200    │  │    Port 7201    │  │    Port...      │            │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘            │
└───────────┼────────────────────┼────────────────────┼──────────────────────┘
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 │
                      ┌──────────▼──────────┐
                      │    Proxy Server     │
                      │     (any node)      │
                      │     Port 9100       │
                      └─────────────────────┘
```

**Expected outcome**
- Horizontal scaling of prefill and decode independently
- Cross-node KV transfer via NIXL
- Load balancing across pools

## 2. When to Use Multi-Node PD

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Single node, 2-4 GPUs | **Single-node PD** (R-021) | Simpler, no network overhead |
| Multiple nodes, 8+ GPUs | **Multi-node PD** (this recipe) | Scale prefill/decode separately |
| Different GPU types | **Multi-node PD** | Prefill on H100, decode on A100 |
| High availability required | **Multi-node PD** | Pool redundancy |
| Single node sufficient | **Single-node PD** (R-021) | Lower latency, simpler ops |

## 3. Installing vLLM + LMCache + NIXL on All Nodes

### 3.1 Hardware Prerequisites
- **2+ nodes** (each with 2+ GPUs)
- **High-speed interconnect** (RDMA/InfiniBand recommended, 10Gbps+ Ethernet minimum)
- **Same model** on all nodes

### 3.2 Network Requirements
```bash
# Verify network connectivity between nodes
ping <node2-ip>

# Check bandwidth (iperf3)
# On node 2: iperf3 -s
# On node 1: iperf3 -c <node2-ip>
# Expected: 10Gbps+ for good performance
```

### 3.3 Software Installation

Install on **all nodes** (prefill and decode):

```bash
# Install LMCache and NIXL
pip install lmcache nixl

# Install vLLM
pip install vllm

# Verify NIXL
python -c "import nixl; print(nixl.__version__)"
```

### 3.4 Firewall Configuration
Ensure these ports are open between nodes:
- **7100-710N**: Prefill instance ports
- **7200-720N**: Decode instance ports
- **7300-730N**: NIXL init ports (for peer discovery)
- **7400-740N**: NIXL alloc ports (for buffer allocation)
- **9100**: Proxy server port

## 4. LMCache Configuration

### 4.1 Prefill Node Configuration

Create `recipes/vllm_pd_multinode_prefiller.yaml`:

```yaml
local_cpu: False
local_disk: False

enable_pd: True
transfer_channel: "nixl"
pd_role: "sender"

# Use node IP for multi-node
pd_proxy_host: "<proxy-node-ip>"  # IP of proxy server node
pd_proxy_port: 7500

pd_buffer_size: 2147483648  # 2GB for multi-node
pd_buffer_device: "cuda"

# Multi-node NIXL configuration
nixl_backends: [UCX, TCP]  # TCP for cross-node
```

### 4.2 Decode Node Configuration

Create `recipes/vllm_pd_multinode_decoder.yaml`:

```yaml
local_cpu: False
local_disk: False

enable_pd: True
transfer_channel: "nixl"
pd_role: "receiver"

# Peer discovery - will connect to prefillers
pd_peer_host: "0.0.0.0"  # Listen on all interfaces
pd_peer_init_port: 7300
pd_peer_alloc_port: 7400

# Larger buffer for decode
pd_buffer_size: 4294967296  # 4GB
pd_buffer_device: "cuda"

nixl_backends: [UCX, TCP]
```

## 5. Launching the PD Cluster

### 5.1 Node 1: Start Prefill Instances

```bash
export PYTHONHASHSEED=0

# Prefill Instance 1 (Port 7100)
UCX_TLS=cuda_ipc,cuda_copy,tcp,rdmacm \
LMCACHE_CONFIG_FILE=recipes/vllm_pd_multinode_prefiller.yaml \
VLLM_ENABLE_V1_MULTIPROCESSING=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 7100 \
  --tensor-parallel-size 2 \
  --disable-log-requests \
  --enforce-eager \
  --no-enable-prefix-caching \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer1"}}' &

# Prefill Instance 2 (Port 7101)
UCX_TLS=cuda_ipc,cuda_copy,tcp,rdmacm \
LMCACHE_CONFIG_FILE=recipes/vllm_pd_multinode_prefiller.yaml \
VLLM_ENABLE_V1_MULTIPROCESSING=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
CUDA_VISIBLE_DEVICES=2,3 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 7101 \
  --tensor-parallel-size 2 \
  --disable-log-requests \
  --enforce-eager \
  --no-enable-prefix-caching \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer2"}}' &
```

### 5.2 Node 2: Start Decode Instances

```bash
export PYTHONHASHSEED=0

# Decode Instance 1 (Port 7200)
UCX_TLS=cuda_ipc,cuda_copy,tcp,rdmacm \
LMCACHE_CONFIG_FILE=recipes/vllm_pd_multinode_decoder.yaml \
VLLM_ENABLE_V1_MULTIPROCESSING=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 7200 \
  --tensor-parallel-size 2 \
  --disable-log-requests \
  --enforce-eager \
  --no-enable-prefix-caching \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1", "skip_last_n_tokens": 1}}' &

# Decode Instance 2 (Port 7201)
UCX_TLS=cuda_ipc,cuda_copy,tcp,rdmacm \
LMCACHE_CONFIG_FILE=recipes/vllm_pd_multinode_decoder.yaml \
VLLM_ENABLE_V1_MULTIPROCESSING=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
CUDA_VISIBLE_DEVICES=2,3 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 7201 \
  --tensor-parallel-size 2 \
  --disable-log-requests \
  --enforce-eager \
  --no-enable-prefix-caching \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer2", "skip_last_n_tokens": 1}}' &
```

### 5.3 Start Proxy Server (can be on any node)

```bash
python examples/disagg_prefill/disagg_proxy_server.py \
  --prefiller-host "<node1-ip>,<node1-ip>" \
  --prefiller-port "7100,7101" \
  --decoder-host "<node2-ip>,<node2-ip>" \
  --decoder-port "7200,7201" \
  --decoder-init-port "7300,7301" \
  --decoder-alloc-port "7400,7401" \
  --port 9100
```

## 6. Startup Validation

### Verify prefill instances (Node 1)
```bash
ss -tlnp | grep -E "7100|7101"
# Shows listening on 7100 and 7101
```

### Verify decode instances (Node 2)
```bash
ss -tlnp | grep -E "7200|7201|7300|7301|7400|7401"
# Shows listening on decode and NIXL ports
```

### Verify proxy (Proxy node)
```bash
curl http://localhost:9100/health
# Should return 200 OK
```

### Test cross-node connectivity
```bash
# From Node 1, test decode port on Node 2
nc -zv <node2-ip> 7200

# From Node 2, test prefill port on Node 1
nc -zv <node1-ip> 7100
```

## 7. Inference and Multi-Node Validation

### Send request through proxy
```bash
curl http://<proxy-node-ip>:9100/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Write a detailed analysis of distributed systems architecture.",
    "max_tokens": 200
  }'
```

### Verify cross-node transfer in logs

**Node 1 (Prefill) logs:**
```
LMCache INFO: Computing KV for prompt (150 tokens)
LMCache INFO: Initiating NIXL transfer to <node2-ip>
LMCache INFO: KV transfer complete: 0.25 GB in 45.2 ms
```

**Node 2 (Decode) logs:**
```
LMCache INFO: Received KV from <node1-ip>:7100
LMCache INFO: Starting token generation
```

## 8. Benchmarking

### Load test with multiple concurrent requests
```bash
vllm bench serve --port 9100 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 7500 \
  --random-output-len 200 \
  --num-prompts 100 \
  --burstiness 50 \
  --request-rate 10
```

### Scaling test
```bash
# 1 prefill + 1 decode
# vs
# 2 prefill + 2 decode
# vs
# 4 prefill + 4 decode
```

### Expected throughput scaling
| Configuration | Throughput | TTFT |
|--------------|------------|------|
| 1P+1D | ~3000 tok/s | ~400ms |
| 2P+2D | ~6000 tok/s | ~350ms |
| 4P+4D | ~12000 tok/s | ~300ms |

## 9. Production Considerations

### 9.1 Session Affinity
For multi-turn conversations, use session affinity:
```bash
export CLIENT_BOUND="true"
export CLIENT_BOUND_KEY="session-id"

python disagg_proxy_server.py \
  --prefiller-host "..." \
  --decoder-host "..." \
  --port 9100
```

Client adds header:
```python
extra_headers = {"session-id": "user-123-session"}
```

### 9.2 Health Checks and Auto-scaling
```bash
# Kubernetes readiness probe
readinessProbe:
  httpGet:
    path: /health
    port: 9100
  periodSeconds: 10
```

### 9.3 Network Optimization
```bash
# For InfiniBand
export UCX_TLS=rc,cuda_ipc,cuda_copy,tcp

# For RoCE
export UCX_TLS=dc,cuda_ipc,cuda_copy,tcp

# For Ethernet only
export UCX_TLS=tcp,sockcm
```

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Connection timeout | Firewall blocking ports | Open ports 7100-7400 |
| Slow transfer | TCP instead of RDMA | Check UCX_TLS, verify RDMA |
| Proxy routing fails | Wrong IP in config | Use correct node IPs |
| NIXL init fails | Port conflict | Use different ports per instance |
| GPU OOM | Buffer too large | Reduce pd_buffer_size |
| Uneven load | No load balancing | Proxy uses round-robin by default |

### Debug network path
```bash
# Check RDMA connectivity (if available)
ib_write_bw  # Server
ib_write_bw <server-ip>  # Client

# Check UCX info
ucx_info -d | grep -E "Transport|Device"
```

## 11. Scaling Guidance

### Prefill pool sizing
- Scale based on input token throughput
- Rule of thumb: 1 prefill instance per 5000 input tokens/second

### Decode pool sizing
- Scale based on output token throughput
- Rule of thumb: 1 decode instance per 1000 output tokens/second

### Ratio guidance
| Workload Type | Prefill:Decode Ratio |
|---------------|---------------------|
| Long input, short output | 2:1 or 3:1 |
| Balanced | 1:1 |
| Short input, long output | 1:2 or 1:3 |

## 12. Additional Resources
- Single-node PD: `recipes/vllm_pd_single_node.md` (R-021)
- PD tuning: `recipes/vllm_pd_tuning.md` (R-023)
- Example scripts: `examples/disagg_prefill/xpyd/`
