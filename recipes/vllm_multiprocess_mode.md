# LMCache + vLLM: Multiprocess Mode for Isolation and Throughput

## 1. Introduction

**Target workload**
- Production deployments requiring process isolation
- High-throughput scenarios with multiple vLLM instances
- Scenarios where LMCache needs dedicated resources
- **Separate process for cache operations**

**LMCache mode**
- **Storage Mode with Multiprocess**
- Single or multi-node
- Dedicated LMCache process(es) for cache operations

This recipe demonstrates **Multiprocess Mode**, where LMCache runs in a separate process from vLLM:

1. **Process isolation** - Cache operations don't affect inference process
2. **Resource separation** - Dedicated CPU/memory for cache I/O
3. **Crash isolation** - LMCache crash doesn't bring down vLLM
4. **Scalability** - Multiple LMCache processes for high throughput

> **Trade-off:** Multiprocess mode adds inter-process communication (IPC) overhead. It's most valuable when:
> - Process isolation is required (security/stability)
> - Dedicated resources for cache operations
> - High throughput with multiple workers

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                     Main Process                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   vLLM      │    │   vLLM      │    │   vLLM      │     │
│  │  Instance 1 │    │  Instance 2 │    │  Instance N │     │
│  │             │    │             │    │             │     │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘     │
│         │                  │                  │            │
│         └──────────────────┼──────────────────┘            │
│                            │ IPC (Unix sockets/shared mem) │
└────────────────────────────┼───────────────────────────────┘
                             │
┌────────────────────────────┼───────────────────────────────┐
│                     LMCache Process(es)                    │
│  ┌─────────────────────────▼─────────────────────────────┐ │
│  │              LMCache Manager                          │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │ │
│  │  │ Worker 1 │  │ Worker 2 │  │ Worker N │           │ │
│  │  │ (CPU)    │  │ (Disk)   │  │ (Remote) │           │ │
│  │  └──────────┘  └──────────┘  └──────────┘           │ │
│  └───────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

**Expected outcome**
- Process isolation between vLLM and LMCache
- Dedicated resources for cache operations
- Better crash isolation

## 2. When to Use Multiprocess Mode

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Process isolation required | **Multiprocess** (this recipe) | Security/stability |
| High throughput | **Multiprocess** | Parallel cache workers |
| Resource contention | **Multiprocess** | Dedicated cache resources |
| Simple deployments | Standard mode (R-001) | Lower overhead |
| Memory constrained | Standard mode | Lower memory usage |

## 3. Installing vLLM + LMCache

```bash
# Install LMCache (multiprocess mode included)
pip install lmcache

# Install vLLM
pip install vllm
```

## 4. LMCache Configuration

Create `recipes/vllm_multiprocess.yaml`:

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 48
local_disk: true
local_disk_path: "/var/lib/lmcache/kv_cache"
max_local_disk_size: 200

# Enable multiprocess mode
enable_multiprocess: true

# Number of LMCache worker processes
# Each worker handles a subset of cache operations
multiprocess_workers: 4

# IPC method: "shared_memory" or "unix_socket"
# shared_memory: Faster, more memory
# unix_socket: Slower, less memory
ipc_method: "shared_memory"

# Shared memory size for IPC
# Must be large enough for KV chunks
shared_memory_size: 1073741824  # 1GB

save_unfull_chunk: true
```

### Configuration Tuning

```yaml
# Light (few workers, socket IPC)
multiprocess_workers: 2
ipc_method: "unix_socket"

# Balanced (recommended)
multiprocess_workers: 4
ipc_method: "shared_memory"
shared_memory_size: 1073741824

# Heavy (many workers, large shared memory)
multiprocess_workers: 8
ipc_method: "shared_memory"
shared_memory_size: 4294967296  # 4GB
```

## 5. Launching vLLM with Multiprocess Mode

### 5.1 Start LMCache manager process

```bash
# Start LMCache manager (optional - vLLM can auto-start)
python -m lmcache.multiprocess.manager \
  --config recipes/vllm_multiprocess.yaml \
  --port 65000 &
```

### 5.2 Start vLLM

```bash
export PYTHONHASHSEED=0
export LMCACHE_CONFIG_FILE=recipes/vllm_multiprocess.yaml

CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
--max-model-len 8192 \
--gpu-memory-utilization 0.85 \
--port 8000 \
--no-enable-prefix-caching \
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

## 6. Startup Validation

Expected LMCache logs:
```
LMCache INFO: Loading LMCache config file recipes/vllm_multiprocess.yaml
LMCache INFO: Multiprocess mode enabled with 4 workers
LMCache INFO: IPC method: shared_memory (1GB)
LMCache INFO: Starting LMCache manager process
LMCache INFO: Creating LMCacheEngine with config:
  {
    'chunk_size': 256,
    'enable_multiprocess': True,
    'multiprocess_workers': 4,
    'ipc_method': 'shared_memory',
    'shared_memory_size': 1073741824,
    ...
  }
```

Verify processes:
```bash
# Check for LMCache manager process
ps aux | grep "lmcache.multiprocess"

# Check for worker processes
ps aux | grep "lmcache-worker"

# Should see 1 manager + N workers
```

## 7. Inference and Multiprocess Validation

### 7.1 Basic request

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": "Explain the benefits of process isolation in distributed systems.",
    "max_tokens": 100
  }'
```

Expected log (multiprocess):
```
LMCache INFO: [Main] Requesting cache operation
LMCache INFO: [Worker-1] Processing store request
LMCache INFO: [Worker-1] Store completed: 256 tokens
LMCache INFO: [Main] Continuing inference (non-blocking)
```

### 7.2 Verify process isolation

```bash
# Find vLLM and LMCache PIDs
pgrep -a -f "vllm serve"
pgrep -a -f "lmcache"

# Check they are separate processes
ps -o pid,ppid,cmd -p $(pgrep -f "vllm|lmcache") | head -20

# Should show different PIDs and parent processes
```

## 8. Benchmarking

### 8.1 Single process baseline

```yaml
# Standard config (single process)
chunk_size: 256
local_cpu: true
max_local_cpu_size: 48
# enable_multiprocess not set (defaults to false)
```

### 8.2 With multiprocess mode

```bash
# With multiprocess config
vllm serve Qwen/Qwen3-4B-Instruct \
  --max-model-len 8192 \
  --port 8000 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

# Benchmark
vllm bench serve --port 8000 \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 100 \
  --num-prompts 100 \
  --max-concurrency 20
```

### 8.3 Expected results

| Metric | Single Process | Multiprocess | Note |
|--------|----------------|--------------|------|
| p50 TTFT | 115ms | 120ms | Slight overhead |
| p99 TTFT | 280ms | 250ms | Better isolation |
| Throughput | 55 req/s | 60 req/s | Parallel workers |
| Memory | 10GB | 12GB | IPC overhead |
| Stability | Good | Better | Crash isolation |

## 9. Multiprocess Tuning

### 9.1 Number of workers

```yaml
# Few workers (lower overhead)
multiprocess_workers: 2

# Balanced
multiprocess_workers: 4

# Many workers (higher throughput)
multiprocess_workers: 8
```

**Guideline:** Match to number of storage backends or 2-4x CPU cores.

### 9.2 IPC method

```yaml
# Unix socket (lower memory, higher latency)
ipc_method: "unix_socket"

# Shared memory (higher memory, lower latency)
ipc_method: "shared_memory"
shared_memory_size: 1073741824
```

### 9.3 Shared memory sizing

```yaml
# Small (suitable for small models)
shared_memory_size: 536870912  # 512MB

# Medium
shared_memory_size: 1073741824  # 1GB

# Large (suitable for large models)
shared_memory_size: 4294967296  # 4GB
```

**Formula:** `shared_memory_size = max_concurrent_requests × avg_chunk_size × 2`

## 10. Performance Tips

| Tip | Configuration | Impact |
|-----|---------------|--------|
| Use shared memory | `ipc_method: shared_memory` | Lower IPC latency |
| Right-size workers | `multiprocess_workers: 4` | Match to CPUs/backends |
| Monitor IPC | Watch logs | Tune `shared_memory_size` |
| Combine with async | `enable_async: true` | Maximum throughput |

## 11. Troubleshooting / Common Pitfalls

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| No separate processes | `enable_multiprocess: false` | Check config |
| Shared memory error | Size too small | Increase `shared_memory_size` |
| High IPC latency | Unix socket + large chunks | Use shared memory |
| Worker crash | Worker bug | Check worker logs |
| Memory leak | Shared memory not freed | Restart LMCache manager |

### Debug multiprocess

```bash
# Check IPC statistics
tail -f vllm.log | grep -i "IPC\|multiprocess"

# Monitor worker processes
watch -n 1 'ps -o pid,cpu,mem,cmd -p $(pgrep -f lmcache-worker)'

# Check shared memory usage
ipcs -m | grep $(whoami)
```

## 12. Production Deployment

### 12.1 Systemd service for LMCache manager

```ini
# /etc/systemd/system/lmcache-manager.service
[Unit]
Description=LMCache Manager
After=network.target

[Service]
Type=simple
User=lmcache
ExecStart=/usr/bin/python -m lmcache.multiprocess.manager \
  --config /etc/lmcache/multiprocess.yaml \
  --port 65000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 12.2 Kubernetes deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-with-lmcache
spec:
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        env:
        - name: LMCACHE_CONFIG_FILE
          value: "/config/multiprocess.yaml"
        volumeMounts:
        - name: lmcache-config
          mountPath: /config
        - name: shm
          mountPath: /dev/shm
      - name: lmcache-manager
        image: lmcache/lmcache:latest
        command:
        - python
        - -m
        - lmcache.multiprocess.manager
        - --config
        - /config/multiprocess.yaml
        volumeMounts:
        - name: lmcache-config
          mountPath: /config
        - name: shm
          mountPath: /dev/shm
      volumes:
      - name: lmcache-config
        configMap:
          name: lmcache-config
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 2Gi
```

## 13. Additional Resources
- CPU hot cache: `recipes/dense_instruct_cpu_hot_cache.md` (R-001)
- Async loading: `recipes/vllm_async_loading.md` (R-027)
- Tiered storage: `recipes/vllm_tiered_storage.md` (R-029)
