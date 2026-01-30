# LMCache + vLLM: LMCache Server (lm://) Lightweight Remote Backend

## 1. Introduction

**Target workload**
- Multi-instance deployments without Redis infrastructure
- Local development with multiple vLLM processes
- Lightweight sharing without external dependencies
- **Zero-dependency** cache sharing

**LMCache mode**
- **Storage Mode**
- Multi-instance (single node or multi-node)
- LMCache native server backend (`lm://` protocol)

This recipe demonstrates how to use the **LMCache Server** as a lightweight shared cache backend. Unlike Redis (which requires separate installation), the LMCache Server is built into the LMCache package and provides:

1. **Zero external dependencies** - No Redis, no database, just Python
2. **Native protocol** - Optimized for LMCache's specific needs
3. **Lightweight** - Minimal memory and CPU overhead
4. **Easy setup** - Single command to start the server

> **Trade-off:** LMCache Server is simpler than Redis but lacks enterprise features like clustering, persistence, and advanced monitoring. For production at scale, consider Redis (R-010) or tiered storage (R-029).

**Expected outcome**
- LMCache Server runs as a standalone process
- Multiple vLLM instances connect via `remote_url: lm://hostname:port`
- Cache is shared across all connected instances

## 2. When to Use LMCache Server

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Quick prototyping / development | **LMCache Server** | No Redis setup needed |
| No Redis expertise available | **LMCache Server** | Built-in, simple to run |
| Single-node multi-GPU | **LMCache Server** | Low overhead, easy to start |
| Production with high availability | **Redis** (R-010) | Better monitoring, clustering |
| Multi-node production | **Redis** or **Tiered** (R-029) | More robust, better performance |
| Need persistence | **Disk** (R-007) or **Redis** | LMCache Server is in-memory only |

## 3. Installing vLLM + LMCache

```bash
# Install LMCache (includes lmcache_server)
pip install lmcache

# Install vLLM
pip install vllm
```

Verify installation:
```bash
# Check that lmcache_server command is available
python -m lmcache.server --help
```

## 4. LMCache Server Configuration

### 4.1 Start the LMCache Server

```bash
# Basic start (default port 65432)
python -m lmcache.server \
  --host 0.0.0.0 \
  --port 65432 \
  --max-cache-size 50
```

**Parameters:**
- `--host`: Bind address (use `0.0.0.0` to accept remote connections, `127.0.0.1` for local only)
- `--port`: Server port (default: 65432)
- `--max-cache-size`: Maximum cache size in GB (default: 50)

### 4.2 Run in background (production)

```bash
# Using nohup
nohup python -m lmcache.server \
  --host 0.0.0.0 \
  --port 65432 \
  --max-cache-size 100 \
  > /var/log/lmcache_server.log 2>&1 &

# Or using systemd (create /etc/systemd/system/lmcache-server.service)
```

**Systemd service example:**
```ini
[Unit]
Description=LMCache Server
After=network.target

[Service]
Type=simple
User=lmcache
ExecStart=/usr/bin/python -m lmcache.server --host 0.0.0.0 --port 65432 --max-cache-size 100
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 4.3 Verify server is running

```bash
# Check process
ps aux | grep lmcache.server

# Check port is listening
netstat -tlnp | grep 65432
# or
ss -tlnp | grep 65432

# Test connectivity (if netcat available)
nc -zv localhost 65432
```

## 5. LMCache Client Configuration

Create `recipes/vllm_lmcache_server.yaml`:

```yaml
chunk_size: 256
# Disable local storage to demonstrate pure remote caching
local_cpu: false
local_disk: false
# Enable LMCache Server remote backend
remote_url: "lm://localhost:65432"
use_layerwise: false
save_unfull_chunk: true
```

> **URL Format:** `lm://hostname:port`
> 
> Examples:
> - `lm://localhost:65432` - Local server
> - `lm://192.168.1.100:65432` - Remote server on LAN
> - `lm://lmcache-server.internal:65432` - Kubernetes service

## 6. Launching Multiple vLLM Instances (with LMCache Server)

### 6.1 Start Instance A (Port 8000)

```bash
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=recipes/vllm_lmcache_server.yaml \
CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
--max-model-len 8192 \
--gpu-memory-utilization 0.85 \
--port 8000 \
--no-enable-prefix-caching \
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

### 6.2 Start Instance B (Port 8001)

In a separate terminal:

```bash
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=recipes/vllm_lmcache_server.yaml \
CUDA_VISIBLE_DEVICES=1 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
--max-model-len 8192 \
--gpu-memory-utilization 0.85 \
--port 8001 \
--no-enable-prefix-caching \
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

> **Note:** Both instances connect to the same LMCache Server at `lm://localhost:65432`

## 7. Startup Validation

### LMCache Server logs
```
LMCache Server starting on 0.0.0.0:65432
Max cache size: 50 GB
Server ready
```

### Instance A logs
```
LMCache INFO: Loading LMCache config file recipes/vllm_lmcache_server.yaml
LMCache INFO: Creating LMCacheEngine with config:
  {'chunk_size': 256, 'remote_url': 'lm://localhost:65432', ...}
LMCache INFO: Initializing RemoteBackend at localhost:65432
```

### Instance B logs
```
LMCache INFO: Loading LMCache config file recipes/vllm_lmcache_server.yaml
LMCache INFO: Creating LMCacheEngine with config:
  {'chunk_size': 256, 'remote_url': 'lm://localhost:65432', ...}
LMCache INFO: Initializing RemoteBackend at localhost:65432
```

## 8. Inference and Cross-Instance Cache Validation

### 8.1 Warm cache on Instance A

```bash
# Send request to Instance A
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": "You are a helpful AI assistant.\n\nUser: Explain the architecture of transformer models in detail. This is a comprehensive prompt that will generate KV cache data.",
    "max_tokens": 100
  }'
```

Expected Instance A logs (cold, stores to LMCache Server):
```
LMCache INFO: Reqid: ..., Total tokens 512, LMCache hit tokens: 0, need to load: 0
LMCache INFO: Stored 512 out of total 512 tokens. size: 0.0703 GB
```

### 8.2 Get cache hit on Instance B

```bash
# Send SAME request to Instance B
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": "You are a helpful AI assistant.\n\nUser: Explain the architecture of transformer models in detail. This is a comprehensive prompt that will generate KV cache data.",
    "max_tokens": 100
  }'
```

Expected Instance B logs (warm, retrieves from LMCache Server):
```
LMCache INFO: Reqid: ..., Total tokens 512, LMCache hit tokens: 512, need to load: 512
LMCache INFO: Retrieved 512 out of 512 required tokens. size: 0.0703 gb
```

**Key observation:** Instance B got a cache hit from the LMCache Server, which stored the KV from Instance A.

## 9. Benchmarking

### 9.1 Baseline (no sharing)

Run two instances WITHOUT remote backend:

```bash
# Both instances with local_cpu only
# Instance A - send request
# Instance B - send same request
# Result: Both compute from scratch
```

### 9.2 With LMCache Server

```bash
# Instance A - warm cache
# Instance B - cache hit from LMCache Server
```

### 9.3 Benchmark results

| Scenario | Instance A TTFT | Instance B TTFT | Improvement |
|----------|-----------------|-----------------|-------------|
| No sharing | ~600ms | ~600ms | None |
| LMCache Server | ~600ms | ~200ms | **~67% faster** |

### 9.4 Compare: LMCache Server vs Redis

| Feature | LMCache Server | Redis (R-010) |
|---------|---------------|---------------|
| Setup complexity | Minimal | Requires Redis install |
| External dependencies | None | Redis server |
| Persistence | No (in-memory only) | Yes (RDB/AOF) |
| Clustering | No | Yes (Redis Cluster) |
| Monitoring | Basic | Advanced |
| Performance | Similar | Similar |
| Best for | Dev, prototyping, small scale | Production, large scale |

## 10. LMCache Server Tuning

### 10.1 Memory sizing

```bash
# For ~100GB cache
python -m lmcache.server --max-cache-size 100

# Monitor memory usage
watch -n 5 'ps aux | grep lmcache.server'
```

### 10.2 Connection limits

The LMCache Server automatically handles concurrent connections. Default limits are suitable for most deployments:
- Max concurrent connections: ~100
- If you need more, consider Redis (R-010)

### 10.3 Network optimization

```bash
# For multi-node setup, ensure low latency
# Use dedicated network if possible
# Place LMCache Server close to vLLM instances

# Example: Server on dedicated node
python -m lmcache.server --host 0.0.0.0 --port 65432 --max-cache-size 200

# Client config
remote_url: "lm://10.0.0.10:65432"  # Server IP
```

## 11. Performance Tips

| Parameter | Recommendation | Impact |
|-----------|---------------|--------|
| Server location | Same host or same rack | Minimize network latency |
| `max-cache-size` | 1.5-2x expected working set | Avoid eviction churn |
| `chunk_size` | 256-512 | Balance granularity vs server ops |
| Connections | Let LMCache handle pooling | Automatic optimization |
| For production | Consider Redis | Better monitoring, HA |

## 12. Troubleshooting / Common Pitfalls

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Connection refused | Server not running | Start `python -m lmcache.server` |
| Connection refused | Wrong port/host | Check `--port` and URL match |
| Connection refused | Firewall | Open port 65432 (or chosen port) |
| No cache hit on Instance B | Different `chunk_size` | Ensure identical YAML on all instances |
| Server OOM | `--max-cache-size` too large | Reduce to fit available RAM |
| Slow performance | Network latency | Co-locate server with vLLM instances |
| Server crash | Out of memory | Monitor with `free -h`, reduce cache size |
| "Address already in use" | Another server running | Kill old process or use different port |

### Restart LMCache Server

```bash
# Kill existing server
pkill -f "lmcache.server"

# Clear cache and restart
python -m lmcache.server --max-cache-size 100

# Or use different port
python -m lmcache.server --port 65433
```

### Debug connections

```bash
# Check server is listening
ss -tlnp | grep 65432

# Check established connections
ss -tn | grep 65432

# View server logs
journalctl -u lmcache-server -f
# or
tail -f /var/log/lmcache_server.log
```

## 13. Multi-Node Deployment

For deployments across multiple machines:

### On the LMCache Server node:
```bash
# Bind to all interfaces
python -m lmcache.server \
  --host 0.0.0.0 \
  --port 65432 \
  --max-cache-size 200
```

### On vLLM client nodes:
```yaml
# vllm_lmcache_server.yaml
chunk_size: 256
remote_url: "lm://<SERVER_IP>:65432"
# Replace <SERVER_IP> with actual server address
```

**Network considerations:**
- Ensure firewall allows port 65432 between nodes
- Use private network/VPC for security
- Consider network bandwidth (KV data can be large)

## 14. LMCache Server vs Redis: Decision Guide

Choose **LMCache Server** when:
- ✅ Quick prototyping or development
- ✅ No Redis expertise on team
- ✅ Single-node or small multi-node setup
- ✅ Want minimal external dependencies
- ✅ Don't need persistence or HA

Choose **Redis** (R-010) when:
- ✅ Production deployment at scale
- ✅ Need persistence across server restarts
- ✅ Need high availability (Redis Sentinel)
- ✅ Need clustering for very large caches
- ✅ Need advanced monitoring and tooling
- ✅ Team has Redis expertise

## 15. Additional Resources
- Redis backend recipe: `recipes/vllm_redis_remote.md` (R-010)
- CPU hot cache recipe: `recipes/dense_instruct_cpu_hot_cache.md` (R-001)
- Multi-instance sharing: `recipes/vllm_multi_instance_sharing.md` (R-018)
- Tiered storage recipe: `recipes/vllm_tiered_storage.md` (R-029)
