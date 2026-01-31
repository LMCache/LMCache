# LMCache + vLLM: P2P KV Sharing (Controller + NIXL)

## 1. Introduction

**Target workload**
- Multi-instance vLLM on the same host or high-speed network
- KV reuse without a centralized Redis/Valkey tier
- Low-latency, peer-to-peer KV transfer

**LMCache mode**
- **Storage Mode** with P2P enabled
- Controller-mediated discovery and coordination
- NIXL transfer channel

This recipe shows how to enable **P2P KV sharing** so instance B can fetch KV directly from instance A instead of recomputing or using a centralized cache.

**Expected outcome**
- Instance A warms the KV cache
- Instance B retrieves KV directly from A via P2P
- Lower TTFT for repeated prefixes without Redis

Architecture (single host example):

```
      ┌─────────────────────────────────────┐
      │           LMCache Controller        │
      │        pull:8300  reply:8400        │
      └───────────────┬─────────────────────┘
                      │
      ┌───────────────┼─────────────────────┐
      │                                   │
┌─────▼─────┐                         ┌───▼──────┐
│ vLLM A    │                         │ vLLM B   │
│ port 8010 │                         │ port 8011│
│ p2p init 8200                      │ p2p init 8202
│ p2p lookup 8201                    │ p2p lookup 8203
└───────────┘                         └───────────┘
```

## 2. When to Use LMCache P2P

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Two+ vLLM instances, shared prefixes | **P2P** | Direct KV transfer, no central cache |
| Need durability or persistence | **Redis/S3** | P2P does not persist KV |
| Large multi-node clusters | **Redis/Valkey** | Central cache simpler to operate |
| Ultra-low latency with RDMA | **P2P + NIXL** | Avoids centralized store hop |

## 3. Installing vLLM + LMCache

Prerequisites:
- 2 GPUs on the same host (or fast network across hosts)
- NIXL installed and configured
- LMCache controller available

Install vLLM and LMCache:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install lmcache vllm
```

## 4. LMCache Configuration

Create two config files:

`recipes/vllm_p2p_instance_a.yaml`

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 48
enable_async_loading: true

# P2P configurations
enable_p2p: true
p2p_host: "localhost"
p2p_init_ports: 8200
p2p_lookup_ports: 8201
transfer_channel: "nixl"

# Controller configurations
enable_controller: true
lmcache_instance_id: "lmcache_instance_a"
controller_pull_url: "localhost:8300"
controller_reply_url: "localhost:8400"
lmcache_worker_ports: 8500

extra_config:
  lookup_backoff_time: 0.001
  # P2P backend timeout configurations (optional)
  p2p_socket_recv_timeout_ms: 30000
  p2p_socket_send_timeout_ms: 10000
```

`recipes/vllm_p2p_instance_b.yaml`

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 48
enable_async_loading: true

# P2P configurations
enable_p2p: true
p2p_host: "localhost"
p2p_init_ports: 8202
p2p_lookup_ports: 8203
transfer_channel: "nixl"

# Controller configurations
enable_controller: true
lmcache_instance_id: "lmcache_instance_b"
controller_pull_url: "localhost:8300"
controller_reply_url: "localhost:8400"
lmcache_worker_ports: 8501

extra_config:
  lookup_backoff_time: 0.001
```

**Critical guidance**
- `chunk_size` must match across instances.
- P2P requires unique `p2p_init_ports` and `p2p_lookup_ports` per instance.
- Use `transfer_channel: "nixl"` and set `UCX_TLS` accordingly.

## 5. Launching the Server (with LMCache)

Start the controller:

```bash
PYTHONHASHSEED=0 lmcache_controller --host localhost --port 9000 \
  --monitor-ports '{"pull": 8300, "reply": 8400}'
```

Start vLLM instance A:

```bash
PYTHONHASHSEED=0 UCX_TLS=rc CUDA_VISIBLE_DEVICES=0 \
LMCACHE_CONFIG_FILE=recipes/vllm_p2p_instance_a.yaml \
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --gpu-memory-utilization 0.8 \
  --port 8010 \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

Start vLLM instance B:

```bash
PYTHONHASHSEED=0 UCX_TLS=rc CUDA_VISIBLE_DEVICES=1 \
LMCACHE_CONFIG_FILE=recipes/vllm_p2p_instance_b.yaml \
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --gpu-memory-utilization 0.8 \
  --port 8011 \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

## 6. Startup Validation

Expected logs on instance B (P2P setup):

```
LMCache INFO: Established connection to peer_init_url localhost:8200.
LMCache INFO: The peer_lookup_url: localhost:8201
```

## 7. Inference and Cache Validation

### 7.1 Warm cache on instance A

```bash
python - <<'PY' | curl http://localhost:8010/v1/completions \
  -H "Content-Type: application/json" \
  -d @-
import json
prompt = "Explain the significance of KV cache in language models. " * 100
payload = {
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": prompt,
    "max_tokens": 16,
}
print(json.dumps(payload))
PY
```

### 7.2 Hit cache on instance B

```bash
python - <<'PY' | curl http://localhost:8011/v1/completions \
  -H "Content-Type: application/json" \
  -d @-
import json
prompt = "Explain the significance of KV cache in language models. " * 100
payload = {
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": prompt,
    "max_tokens": 16,
}
print(json.dumps(payload))
PY
```

Expected logs on instance B:

```
LMCache INFO: Retrieved 1002 out of total 1002 tokens. size: 0.1223 gb
```

## 8. Benchmarking

Use the same prompt on A then B and compare TTFT:

```bash
vllm bench serve --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 6144 \
  --prefix-repetition-suffix-len 128 \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len 32 \
  --num-prompts 20 --request-rate 0.5 --max-concurrency 1
```

### Comparison table

| Scenario | Instance A TTFT | Instance B TTFT | Notes |
|----------|------------------|------------------|-------|
| No sharing (local only) | ~600ms | ~600ms | Both cold |
| P2P sharing | ~600ms | ~150ms | B pulls from A |
| Redis centralized | ~600ms | ~200ms | Extra network hop |

## 9. Optimizing Performance

- Use RDMA-capable fabric for NIXL (set `UCX_TLS=rc`).
- Keep `lookup_backoff_time` low to reduce lookup latency.
- Align `chunk_size` across instances for cache hits.
- Consider Redis if you need persistence or cross-cluster reuse.

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| No P2P hits | Ports mismatch | Ensure init/lookup ports are unique and correct |
| Connection refused | Firewall or network | Open ports 8200-8203 and controller ports |
| Controller errors | Misconfigured URLs | Verify `controller_pull_url` and `controller_reply_url` |
| Low hit rate | Prompt mismatch | Ensure identical prompts/tokenization |
| Slow transfers | UCX config | Set `UCX_TLS=rc` for RDMA |

## 11. Additional Resources

- P2P example: `examples/kv_cache_reuse/share_across_instances/p2p_sharing/README.md`
- LMCache controller: `lmcache/tools/controller_benchmark/README.md`
- Redis sharing recipe: `recipes/vllm_redis_remote.md`
