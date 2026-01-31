# LMCache + vLLM: Controller-Based Routing and Cache Operations

## 1. Introduction

**Target workload**
- Multi-instance vLLM deployments needing external orchestration
- KV cache movement across instances (move/copy)
- Operational control for compress/decompress/health checks

**LMCache mode**
- **Storage Mode** with controller enabled
- Controller-mediated operations
- NIXL transfer channel for peer movement

This recipe shows how to run the **LMCache controller** to coordinate cache operations across multiple vLLM instances. You will move KV from instance A to instance B and validate health via the controller API.

**Expected outcome**
- Controller receives worker heartbeats
- Cache move succeeds between instances
- Health checks return worker status

Architecture (single host example):

```
      ┌─────────────────────────────────────┐
      │           LMCache Controller        │
      │   api:9000  pull:8300  reply:8400  │
      └───────────────┬─────────────────────┘
                      │
      ┌───────────────┼─────────────────────┐
      │                                   │
┌─────▼─────┐                         ┌───▼──────┐
│ vLLM A    │                         │ vLLM B   │
│ port 8000 │                         │ port 8001│
│ p2p init 8200                      │ p2p init 8202
│ p2p lookup 8201                    │ p2p lookup 8203
└───────────┘                         └───────────┘
```

## 2. When to Use the Controller

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Need external control of KV | **Controller** | Move/copy/compress via API |
| P2P KV transfer across instances | **Controller + NIXL** | Coordinates peer transfers |
| Simple shared cache | **Redis/Valkey** | Fewer moving parts |
| Single node dev | **Local CPU cache** | No controller required |

## 3. Installing vLLM + LMCache

Prerequisites:
- 2 GPUs on the same host (or fast network)
- NIXL installed and configured
- Controller API port reachable (default 9000)

Install vLLM and LMCache:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install lmcache vllm
```

## 4. LMCache Configuration

Create two config files:

`recipes/vllm_controller_instance_a.yaml`

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 48

# P2P configurations (for KV movement between peers)
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
```

`recipes/vllm_controller_instance_b.yaml`

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 48

# P2P configurations (for KV movement between peers)
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
```

**Critical guidance**
- `chunk_size` must match across instances.
- P2P init/lookup ports must be unique per instance.
- Controller pull/reply URLs must be reachable by workers.

## 5. Launching the Server (with LMCache)

Start the controller:

```bash
PYTHONHASHSEED=0 lmcache_controller --host localhost --port 9000 \
  --monitor-ports '{"pull": 8300, "reply": 8400}'
```

Start vLLM instance A:

```bash
PYTHONHASHSEED=0 UCX_TLS=rc CUDA_VISIBLE_DEVICES=0 \
LMCACHE_CONFIG_FILE=recipes/vllm_controller_instance_a.yaml \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --gpu-memory-utilization 0.8 \
  --port 8000 \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

Start vLLM instance B:

```bash
PYTHONHASHSEED=0 UCX_TLS=rc CUDA_VISIBLE_DEVICES=1 \
LMCACHE_CONFIG_FILE=recipes/vllm_controller_instance_b.yaml \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --gpu-memory-utilization 0.8 \
  --port 8001 \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

## 6. Startup Validation

Expected controller logs:

```
LMCache INFO: Controller listening on 0.0.0.0:9000
LMCache INFO: Monitor pull:8300 reply:8400
```

Expected worker logs:

```
LMCache INFO: Registered with controller pull:localhost:8300 reply:localhost:8400
```

## 7. Inference and Cache Validation

### 7.1 Warm cache on instance A

```bash
python - <<'PY' | curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d @-
import json
prompt = "Explain the significance of KV cache in language models. " * 100
payload = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": prompt,
    "max_tokens": 16,
}
print(json.dumps(payload))
PY
```

### 7.2 Tokenize the prompt

```bash
python - <<'PY' | curl http://localhost:8000/tokenize \
  -H "Content-Type: application/json" \
  -d @-
import json
payload = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Explain the significance of KV cache in language models. " * 100,
}
print(json.dumps(payload))
PY
```

Copy the returned `tokens` array for the controller operations below.

### 7.3 Move KV cache via controller

```bash
curl -X POST http://localhost:9000/move \
  -H "Content-Type: application/json" \
  -d '{
    "old_position": ["lmcache_instance_a", "LocalCPUBackend"],
    "new_position": ["lmcache_instance_b", "LocalCPUBackend"],
    "tokens": [128000, 849, 21435, 279]
  }'
```

Expected response:

```
{"num_tokens": 4, "event_id": "Move..."}
```

### 7.4 Health check via controller

```bash
curl -X POST http://localhost:9000/health \
  -H "Content-Type: application/json" \
  -d '{"instance_id": "lmcache_instance_b"}'
```

Expected response:

```
{"event_id":"health...","error_codes":{"0":0}}
```

## 8. Benchmarking

Run the same prompt on A then on B after a move and compare TTFT:

```bash
vllm bench serve --model meta-llama/Llama-3.1-8B-Instruct \
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
| No controller | ~600ms | ~600ms | Both cold |
| Controller move | ~600ms | ~150ms | B reuses KV |

## 9. Optimizing Performance

- Use RDMA-capable fabric for NIXL (`UCX_TLS=rc`).
- Keep controller on the same network as workers to reduce orchestration latency.
- Avoid large token lists in a single move; batch if needed.

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Controller API 404 | Wrong port | Ensure controller is on port 9000 |
| Worker not registered | Controller URLs wrong | Check `controller_pull_url`/`controller_reply_url` |
| Move fails | Token list invalid | Use `/tokenize` from the source instance |
| Slow move | UCX config | Set `UCX_TLS=rc` and verify RDMA |
| Health errors | Worker offline | Restart vLLM instance |

## 11. Additional Resources

- Controller move example: `examples/cache_controller/move/README.md`
- Controller API server: `lmcache/v1/api_server/__main__.py`
- P2P sharing: `recipes/vllm_p2p_sharing.md`
