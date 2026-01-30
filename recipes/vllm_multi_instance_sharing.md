# LMCache + vLLM: Multi-Instance Cache Sharing with Centralized Store

## 1. Introduction

**Target workload**
- Horizontal scaling with multiple vLLM instances
- Kubernetes deployments with multiple pods
- Load-balanced serving architectures
- **Shared cache across a pool of workers**

**LMCache mode**
- **Storage Mode**
- Multi-instance
- Centralized remote backend (Redis or LMCache Server)

This recipe demonstrates a **complete multi-instance architecture** where multiple vLLM instances share a single centralized cache. Unlike single-instance recipes (R-001, R-007), this pattern enables:

1. **Cache efficiency** - No duplicate caching across instances
2. **Warm start for new instances** - New pods immediately access existing cache
3. **Load balancer friendly** - Any instance can serve any request with equal cache efficiency
4. **Horizontal scalability** - Add/remove instances without cache fragmentation

**Architecture:**
```
                    ┌─────────────────┐
                    │   Load Balancer │
                    │    (Nginx/      │
                    │   Kubernetes)   │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
    ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
    │  vLLM       │   │  vLLM       │   │  vLLM       │
    │  Instance A │   │  Instance B │   │  Instance N │
    │  Port 8000  │   │  Port 8001  │   │  Port 80xx  │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Remote Cache   │
                    │  (Redis or      │
                    │  LMCache Server)│
                    └─────────────────┘
```

> **Prerequisites:** This recipe builds on R-010 (Redis) or R-014 (LMCache Server). Complete one of those recipes first to have a running remote cache backend.

**Expected outcome**
- Instance A processes a request and stores KV to remote cache
- Instance B serves the same prompt with a cache hit from remote
- Both instances benefit from shared cache namespace

## 2. When to Use Multi-Instance Sharing

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Multiple vLLM instances behind load balancer | **Multi-instance + Remote** | Consistent cache across all instances |
| Kubernetes HPA (auto-scaling) | **Multi-instance + Remote** | New pods warm up quickly |
| Rolling deployments | **Multi-instance + Remote** | Zero-downtime with cache preservation |
| Single instance | **Local cache only** (R-001) | Simpler, lower overhead |
| Cache warming pipeline | **Multi-instance + Remote** | Batch jobs warm, serving instances consume |

## 3. Prerequisites

Before starting, you need:

1. **Remote cache backend running:**
   - Redis server (R-010), OR
   - LMCache Server (R-014)

2. **Verify backend is accessible:**
   ```bash
   # For Redis
   redis-cli -h <host> -p 6379 ping
   
   # For LMCache Server
   nc -zv <host> 65432
   ```

3. **Sufficient GPU resources:**
   - 2+ GPUs (one per instance), OR
   - Single GPU with careful memory management (demo only)

## 3. Installing vLLM + LMCache

```bash
# Install LMCache
pip install lmcache

# Install vLLM
pip install vllm
```

## 4. Configuration

Create `recipes/vllm_multi_instance.yaml`:

```yaml
chunk_size: 256
local_cpu: true           # Keep local CPU cache for hot data
max_local_cpu_size: 24    # Local hot tier
local_disk: false
# Remote backend - CHOOSE ONE:
# Option A: Redis
remote_url: "redis://localhost:6379"
# Option B: LMCache Server
# remote_url: "lm://localhost:65432"
use_layerwise: false
save_unfull_chunk: true
```

> **Tiering note:** This configuration uses **local CPU + remote** tiering. The local CPU cache provides fast hits for recently accessed data, while the remote cache enables cross-instance sharing.

## 5. Launching Multiple vLLM Instances

### 5.1 Start the Remote Backend

**Option A: Redis**
```bash
# If not already running
redis-server
# or
docker run -d -p 6379:6379 redis:7-alpine
```

**Option B: LMCache Server**
```bash
python -m lmcache.server --host 0.0.0.0 --port 65432 --max-cache-size 50
```

### 5.2 Start Instance A (Port 8000)

```bash
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=recipes/vllm_multi_instance.yaml \
CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
--max-model-len 8192 \
--gpu-memory-utilization 0.85 \
--port 8000 \
--no-enable-prefix-caching \
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

### 5.3 Start Instance B (Port 8001)

In a separate terminal:

```bash
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=recipes/vllm_multi_instance.yaml \
CUDA_VISIBLE_DEVICES=1 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
--max-model-len 8192 \
--gpu-memory-utilization 0.85 \
--port 8001 \
--no-enable-prefix-caching \
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

### 5.4 Verify both instances are healthy

```bash
# Check Instance A
curl http://localhost:8000/v1/models

# Check Instance B
curl http://localhost:8001/v1/models
```

## 6. Startup Validation

### Instance A logs
```
LMCache INFO: Loading LMCache config file recipes/vllm_multi_instance.yaml
LMCache INFO: Creating LMCacheEngine with config:
  {'chunk_size': 256, 'local_cpu': True, 'remote_url': 'redis://localhost:6379', ...}
LMCache INFO: Initializing LocalCPUBackend
LMCache INFO: Initializing RedisBackend at localhost:6379
```

### Instance B logs
```
LMCache INFO: Loading LMCache config file recipes/vllm_multi_instance.yaml
LMCache INFO: Creating LMCacheEngine with config:
  {'chunk_size': 256, 'local_cpu': True, 'remote_url': 'redis://localhost:6379', ...}
LMCache INFO: Initializing LocalCPUBackend
LMCache INFO: Initializing RedisBackend at localhost:6379
```

### Verify backend connections

```bash
# For Redis - check connected clients
redis-cli CLIENT LIST | grep -c "vllm"
# Should show 2+ connections

# For LMCache Server - check logs
# Should show "Client connected" messages
```

## 7. Inference and Cross-Instance Cache Validation

### 7.1 Warm cache on Instance A

```bash
# Send request to Instance A
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": "You are a helpful AI assistant.\n\nUser: Write a comprehensive analysis of machine learning techniques. This prompt should be long enough to generate multiple KV cache chunks for meaningful sharing.",
    "max_tokens": 100
  }'
```

Expected Instance A logs:
```
LMCache INFO: Reqid: ..., Total tokens 512, LMCache hit tokens: 0, need to load: 0
LMCache INFO: Stored 512 out of total 512 tokens. size: 0.0703 GB
# Data is stored to BOTH local CPU cache AND remote backend
```

### 7.2 Get cache hit on Instance B (FIRST TIME)

```bash
# Send SAME request to Instance B
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": "You are a helpful AI assistant.\n\nUser: Write a comprehensive analysis of machine learning techniques. This prompt should be long enough to generate multiple KV cache chunks for meaningful sharing.",
    "max_tokens": 100
  }'
```

Expected Instance B logs:
```
LMCache INFO: Reqid: ..., Total tokens 512, LMCache hit tokens: 512, need to load: 512
LMCache INFO: Retrieved 512 out of 512 required tokens. size: 0.0703 gb
# Data is retrieved from remote backend
```

**Key observation:** Instance B got a **cache hit from the remote backend** despite never having seen this prompt before.

### 7.3 Get cache hit on Instance B (SECOND TIME)

Send the **same request again** to Instance B:

```bash
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": "You are a helpful AI assistant.\n\nUser: Write a comprehensive analysis of machine learning techniques. This prompt should be long enough to generate multiple KV cache chunks for meaningful sharing.",
    "max_tokens": 100
  }'
```

Expected Instance B logs:
```
LMCache INFO: Reqid: ..., Total tokens 512, LMCache hit tokens: 512, need to load: 512
LMCache INFO: Retrieved 512 out of 512 required tokens. size: 0.0703 gb
# Data is retrieved from local CPU cache (faster than remote!)
```

**Key observation:** The second hit on Instance B is served from **local CPU cache** (faster), demonstrating the tiering behavior.

## 8. Benchmarking

### 8.1 Test script for multi-instance scenario

```bash
#!/bin/bash
# save as: test_multi_instance.sh

PROMPT="You are a helpful AI assistant. User: Explain transformer architecture in detail."

echo "=== Test 1: Cold start on Instance A ==="
time curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"Qwen/Qwen3-4B-Instruct-2507\", \"prompt\": \"$PROMPT\", \"max_tokens\": 50}" | jq -r '.choices[0].text'

echo ""
echo "=== Test 2: Cross-instance hit on Instance B ==="
time curl -s http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"Qwen/Qwen3-4B-Instruct-2507\", \"prompt\": \"$PROMPT\", \"max_tokens\": 50}" | jq -r '.choices[0].text'

echo ""
echo "=== Test 3: Local hit on Instance B ==="
time curl -s http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"Qwen/Qwen3-4B-Instruct-2507\", \"prompt\": \"$PROMPT\", \"max_tokens\": 50}" | jq -r '.choices[0].text'
```

### 8.2 Expected results

| Request | Instance | Cache Source | Expected TTFT |
|---------|----------|--------------|---------------|
| 1st | A | None (cold) | ~600-800ms |
| 2nd | B | Remote (Redis/LMCache Server) | ~200-300ms |
| 3rd | B | Local CPU | ~100-150ms |

### 8.3 Load balancer simulation

Simulate load balancer distributing requests:

```python
import random
import requests
import time

instances = ["http://localhost:8000", "http://localhost:8001"]
prompt = "Explain the benefits of KV cache sharing. " * 50

for i in range(10):
    instance = random.choice(instances)
    start = time.time()
    resp = requests.post(
        f"{instance}/v1/completions",
        json={
            "model": "Qwen/Qwen3-4B-Instruct-2507",
            "prompt": prompt,
            "max_tokens": 50
        }
    )
    elapsed = time.time() - start
    print(f"Request {i+1} -> {instance.split(':')[-1]}: TTFT={elapsed:.3f}s")
```

## 9. Kubernetes Deployment Pattern

For Kubernetes deployments, use a shared remote cache:

```yaml
# vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-serving
spec:
  replicas: 3  # Multiple instances
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        env:
        - name: LMCACHE_CONFIG_FILE
          value: "/config/lmcache.yaml"
        - name: PYTHONHASHSEED
          value: "0"
        command:
        - vllm
        - serve
        - Qwen/Qwen3-4B-Instruct-2507
        - --port=8000
        - --kv-transfer-config
        - '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
        volumeMounts:
        - name: lmcache-config
          mountPath: /config
      volumes:
      - name: lmcache-config
        configMap:
          name: lmcache-config
---
# lmcache-config.yaml (ConfigMap)
apiVersion: v1
kind: ConfigMap
metadata:
  name: lmcache-config
data:
  lmcache.yaml: |
    chunk_size: 256
    local_cpu: true
    max_local_cpu_size: 24
    remote_url: "redis://redis-service:6379"  # Shared Redis service
```

## 10. Performance Tips

| Optimization | Configuration | Benefit |
|--------------|---------------|---------|
| Tiered caching | `local_cpu: true` + `remote_url` | Fast local + shared remote |
| Local cache sizing | `max_local_cpu_size: 24` | 1.5x GPU memory |
| Chunk size | `chunk_size: 256` | Balance granularity vs overhead |
| Network | Co-locate remote cache | Minimize latency |
| Load balancer | Sticky sessions (optional) | Improve local hit rate |

## 11. Troubleshooting / Common Pitfalls

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| No cross-instance hits | Different `chunk_size` | Use identical config on all instances |
| No cross-instance hits | Different models | Same model required for cache sharing |
| Slow remote hits | Network latency | Co-locate cache backend with vLLM |
| Instance B not starting | Port conflict | Use different ports per instance |
| GPU OOM | Multiple instances | Use separate GPUs or reduce memory |
| Redis connection errors | Redis not running | Start Redis/LMCache Server first |
| Cache inconsistency | Clock skew | Ensure NTP sync across nodes |

### Debug cross-instance sharing

```bash
# 1. Verify both instances connect to same backend
redis-cli CLIENT LIST

# 2. Check for cache keys after Instance A request
redis-cli KEYS "*" | head -10

# 3. Monitor Instance B logs for "Retrieved" messages
tail -f /var/log/vllm_instance_b.log | grep "LMCache INFO"

# 4. Compare configs
md5sum /path/to/instance_a/lmcache.yaml
md5sum /path/to/instance_b/lmcache.yaml
# Should be identical!
```

## 12. Advanced: Cache Warming Pattern

Use a dedicated "cache warmer" instance to pre-populate the cache:

```bash
# Cache warmer (batch job)
python -c "
import requests
prompts = [
    'Common system prompt 1...',
    'Common system prompt 2...',
    'Frequently asked question 1...',
]
for p in prompts:
    requests.post('http://localhost:8000/v1/completions', json={
        'model': 'Qwen/Qwen3-4B-Instruct-2507',
        'prompt': p,
        'max_tokens': 1  # Minimal generation, just cache the prompt
    })
print('Cache warmed')
"

# Serving instances now have warm cache
# All production instances benefit from pre-populated cache
```

## 13. Migration from Single to Multi-Instance

If you're currently running single-instance with local cache:

1. **Start remote backend** (Redis or LMCache Server)
2. **Update config** to include `remote_url`
3. **Restart first instance** with new config (it will populate remote cache)
4. **Start additional instances** with same config
5. **Verify** cross-instance sharing works
6. **Update load balancer** to distribute traffic

## 14. Additional Resources
- Redis backend recipe: `recipes/vllm_redis_remote.md` (R-010)
- LMCache Server recipe: `recipes/vllm_lmcache_server.md` (R-014)
- CPU hot cache recipe: `recipes/dense_instruct_cpu_hot_cache.md` (R-001)
- Tiered storage recipe: `recipes/vllm_tiered_storage.md` (R-029)
