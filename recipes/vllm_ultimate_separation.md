# LMCache + vLLM: Ultimate Separation Path (PD at Scale)

## 1. Introduction

**Target workload**
- Maximum throughput at scale
- Complete separation of prefill and decode
- Production-grade disaggregated serving
- **The pinnacle of PD deployment**

**LMCache mode**
- **Transport Mode (PD)**
- Multi-node with pools
- NIXL with RDMA
- Controller orchestration

This recipe represents the **ultimate separation architecture**, combining all PD features:

1. **Prefill Pool** - Dedicated nodes for KV computation
2. **Decode Pool** - Dedicated nodes for token generation
3. **NIXL with RDMA** - Ultra-fast GPU-to-GPU transfer
4. **Controller** - Centralized orchestration
5. **Proxy Layer** - Intelligent routing

> **Prerequisites:** Complete R-021 (Single-node PD) and R-022 (Multi-node PD) first.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Global Load Balancer                           │
│                         (Geo-DNS / Anycast)                              │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
    ┌─────────────────────────────┼─────────────────────────────┐
    │                             │                             │
┌───▼──────────┐    ┌─────────────▼──────────────┐    ┌────────▼──────┐
│  Zone: East  │    │       Zone: Central        │    │  Zone: West   │
│              │    │                            │    │               │
│ ┌──────────┐ │    │  ┌──────────────────────┐  │    │ ┌──────────┐  │
│ │Prefill   │ │    │  │     Controller       │  │    │ │Prefill   │  │
│ │Pool (4)  │ │    │  │   (Orchestration)    │  │    │ │Pool (4)  │  │
│ │          │◀┼────┼──│  - Health checks     │──┼────┼▶│          │  │
│ │ GPU x16  │ │    │  │  - Load balancing    │  │    │ │ GPU x16  │  │
│ └────┬─────┘ │    │  │  - Failover          │  │    │ └────┬─────┘  │
│      │       │    │  └──────────────────────┘  │    │      │        │
│      │ RDMA  │    │                            │    │      │ RDMA   │
│      ▼       │    │  ┌──────────────────────┐  │    │      ▼        │
│ ┌──────────┐ │    │  │   Decode Pool (8)    │  │    │ ┌──────────┐  │
│ │ Decode   │ │    │  │                      │  │    │ │ Decode   │  │
│ │ Pool (8) │ │    │  │  GPU x32             │  │    │ │ Pool (8) │  │
│ │          │ │    │  │  Auto-scaling        │  │    │ │          │  │
│ │ GPU x32  │ │    │  └──────────────────────┘  │    │ │ GPU x32  │  │
│ └──────────┘ │    │                            │    │ └──────────┘  │
└──────────────┘    └────────────────────────────┘    └───────────────┘
```

**Expected outcome**
- Maximum throughput through complete separation
- Sub-100ms TTFT at scale
- Automatic failover and scaling

## 2. When to Use Ultimate Separation

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Maximum throughput | **Ultimate separation** (this recipe) | Complete specialization |
| Global deployment | **Ultimate separation** | Multi-zone with controller |
| RDMA available | **Ultimate separation** | Ultra-fast transfer |
| Simple deployment | R-021 (Single-node PD) | Lower complexity |
| Cost sensitive | Standard caching | Lower infrastructure cost |

## 3. Installing Prerequisites

### 3.1 Hardware
- 3+ zones/regions
- InfiniBand or RoCE (RDMA)
- 100+ GPUs total

### 3.2 Software
```bash
# Install LMCache with all PD features
pip install lmcache[nixl,controller]

# Install vLLM
pip install vllm

# Verify RDMA
ibstat
ib_write_bw  # Test RDMA bandwidth
```

## 4. Configuration

## 5. Launching the Ultimate Separation Platform

### 5.1 Controller Setup

### 4.1 Prefill nodes

```yaml
# vllm_ultimate_prefill.yaml
chunk_size: 256
local_cpu: false
local_disk: false

enable_pd: true
transfer_channel: "nixl"
pd_role: "sender"

# Controller registration
enable_controller: true
controller_url: "http://controller.central.zone:9000"
instance_type: "prefill"
zone: "east"

# NIXL with RDMA
nixl_backends: [UCX]
ucx_tls: "rc,cuda_ipc,cuda_copy"

pd_buffer_size: 2147483648  # 2GB
pd_buffer_device: "cuda"
```

### 4.2 Decode nodes

```yaml
# vllm_ultimate_decode.yaml
chunk_size: 256
local_cpu: false
local_disk: false

enable_pd: true
transfer_channel: "nixl"
pd_role: "receiver"

# Controller registration
enable_controller: true
controller_url: "http://controller.central.zone:9000"
instance_type: "decode"
zone: "central"

# NIXL with RDMA
nixl_backends: [UCX]
ucx_tls: "rc,cuda_ipc,cuda_copy"

pd_buffer_size: 4294967296  # 4GB
pd_buffer_device: "cuda"
```

## 5. Controller Setup

### 5.1 Controller configuration

```yaml
# controller.yaml
port: 9000

# Health check interval
health_check_interval: 10

# Load balancing strategy
# options: round_robin, least_loaded, latency_aware
load_balancer: "latency_aware"

# Auto-scaling
autoscaling:
  enabled: true
  min_prefill_replicas: 4
  max_prefill_replicas: 20
  min_decode_replicas: 8
  max_decode_replicas: 50
  
  # Scale up when queue depth > 10
  scale_up_threshold: 10
  # Scale down when queue depth < 3
  scale_down_threshold: 3

# Failover
failover:
  enabled: true
  health_check_timeout: 30
  max_retries: 3
```

### 5.2 Start controller

```bash
python -m lmcache.controller \
  --config controller.yaml \
  --port 9000
```

### 5.2 Deployment

### 6.1 Start prefill pools

```bash
# Zone: East
for i in {1..4}; do
  UCX_TLS=rc,cuda_ipc,cuda_copy \
  LMCACHE_CONFIG_FILE=vllm_ultimate_prefill.yaml \
  CUDA_VISIBLE_DEVICES=$((i-1)) \
  vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port $((7100 + i)) \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer"}' &
done

# Zone: West (same configuration)
```

### 6.2 Start decode pools

```bash
# Zone: Central
for i in {1..8}; do
  UCX_TLS=rc,cuda_ipc,cuda_copy \
  LMCACHE_CONFIG_FILE=vllm_ultimate_decode.yaml \
  CUDA_VISIBLE_DEVICES=$((i-1)) \
  vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port $((7200 + i)) \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer"}' &
done
```

### 6.3 Start proxy with controller

```bash
python examples/disagg_prefill/disagg_proxy_server.py \
  --controller-url http://controller.central.zone:9000 \
  --port 9100 \
  --enable-zone-aware-routing
```

## 6. Validation

### 7.1 Check controller

```bash
# List registered instances
curl http://controller.central.zone:9000/api/v1/instances

# Check health
curl http://controller.central.zone:9000/health
```

### 7.2 Check RDMA connectivity

```bash
# From prefill to decode
ib_write_bw -d mlx5_0  # On decode
ib_write_bw -d mlx5_0 <decode-ip>  # On prefill

# Expected: 50+ Gbps
```

## 7. Benchmarking at Scale

### 8.1 Load test

```bash
vllm bench serve \
  --url http://llm-api.global:9100 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 500 \
  --num-prompts 10000 \
  --max-concurrency 500 \
  --request-rate 100
```

### 8.2 Expected results

| Metric | Target |
|--------|--------|
| Throughput | 20,000+ tok/s |
| p50 TTFT | < 80ms |
| p99 TTFT | < 150ms |
| KV transfer | < 20ms (RDMA) |
| Availability | 99.99% |

## 8. Operations

### 9.1 Scaling

```bash
# Manual scale up
kubectl scale deployment prefill-pool --replicas=8

# Auto-scaling triggers based on queue depth
```

### 9.2 Failover

```bash
# Simulate failure
kubectl delete pod prefill-pool-xyz

# Controller detects and routes to healthy instances
```

### 9.3 Monitoring

```bash
# Controller metrics
curl http://controller.central.zone:9000/metrics

# Key metrics:
# - active_prefill_instances
# - active_decode_instances
# - avg_queue_depth
# - cross_zone_latency
```

## 9. Performance Tuning

### 10.1 RDMA optimization

```bash
# Enable GPU Direct RDMA
export UCX_MEMTYPE_CACHE=y
export UCX_RNDV_SCHEME=get_zcopy

# Tune for large transfers
export UCX_RC_MAX_RD_ATOMIC=16
```

### 10.2 Buffer sizing

```yaml
# Larger buffers for high-throughput
pd_buffer_size: 8589934592  # 8GB
```

### 10.3 Connection pooling

```yaml
# Keep connections alive
keepalive_interval: 30
connection_pool_size: 100
```

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| High TTFT | RDMA not working | Check `ibstat`, use TCP fallback |
| Controller errors | Network partition | Check zone connectivity |
| OOM on decode | Buffer too large | Reduce `pd_buffer_size` |
| Low throughput | Imbalanced pools | Check controller metrics |

## 11. Additional Resources
- Single-node PD: `recipes/vllm_pd_single_node.md` (R-021)
- Multi-node PD: `recipes/vllm_pd_multi_node.md` (R-022)
- PD tuning: `recipes/vllm_pd_tuning.md` (R-023)
- Enterprise platform: `recipes/vllm_enterprise_platform.md` (R-031)
