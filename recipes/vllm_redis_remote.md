# LMCache + vLLM: Redis Remote Backend for Cross-Instance Cache Sharing

## 1. Introduction

**Target workload**
- Multiple vLLM instances sharing a common cache
- Microservices architecture with multiple serving pods
- Need for cache consistency across scaled-out deployments
- **Horizontal scaling** scenarios

**LMCache mode**
- **Storage Mode**
- Multi-instance (single node or multi-node)
- Remote Redis backend

This recipe demonstrates how to run **multiple vLLM instances with a shared Redis cache**. Unlike local CPU/disk storage (which is instance-private), Redis enables:

1. **Cross-instance cache hits** - Instance B can use KV cached by Instance A
2. **Horizontal scalability** - Add more vLLM instances without cache duplication
3. **Centralized cache management** - Single source of truth for cached KV
4. **Cache warming** - Pre-populate cache from batch jobs, serve from real-time instances

> **Trade-off:** Network latency to Redis is higher than local memory/disk. This recipe focuses on **sharing** and **scalability** over raw speed. For best performance, combine with local CPU hot cache (tiering).

**Expected outcome**
- Instance A warms the cache (stores KV to Redis)
- Instance B gets cache hits (retrieves KV from Redis)
- Both instances share the same cache namespace

## 2. When to Use Redis Remote Backend

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Multiple vLLM instances | **LMCache + Redis** | Share cache across all instances |
| Microservices / Kubernetes | **LMCache + Redis** | Pods come and go, cache persists in Redis |
| Cache warming pattern | **LMCache + Redis** | Batch jobs warm, serving instances consume |
| Single instance only | **LMCache + local** | No need for network overhead |
| Latency-sensitive + multi-instance | **LMCache + tiered** | Local CPU + Redis remote (see R-029) |

## 3. Prerequisites

### 3.1 Install Redis Server

**Option A: Local Redis (for testing)**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install redis-server
sudo systemctl enable redis
sudo systemctl start redis

# Verify
redis-cli ping
# Expected: PONG
```

**Option B: Docker Redis**
```bash
docker run -d --name redis-lmcache \
  -p 6379:6379 \
  --restart unless-stopped \
  redis:7-alpine

# Verify
docker exec redis-lmcache redis-cli ping
```

**Option C: Cloud Redis (AWS ElastiCache, etc.)**
- Obtain Redis endpoint URL
- Ensure security groups allow connections from vLLM instances

### 3.2 Verify Redis Connectivity

```bash
# Test connection from vLLM host
redis-cli -h localhost -p 6379 ping

# Check memory policy (should be allkeys-lru or allkeys-lfu)
redis-cli CONFIG GET maxmemory-policy

# Set appropriate eviction policy if needed
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

## 4. Installing vLLM + LMCache

```bash
# Install LMCache
pip install lmcache

# Install vLLM
pip install vllm
```

## 5. LMCache Configuration

Create `recipes/vllm_redis_remote.yaml`:

```yaml
chunk_size: 256
# Disable local storage to demonstrate pure remote caching
local_cpu: false
local_disk: false
# Enable Redis remote backend
remote_url: "redis://localhost:6379"  # Adjust host/port as needed
# For Redis with authentication:
# remote_url: "redis://:password@localhost:6379"
# For Redis with specific DB:
# remote_url: "redis://localhost:6379/0"
use_layerwise: false
save_unfull_chunk: true
```

> **⚠️ Redis URL Format**
> 
> Standard Redis URL format: `redis://[password@]host:port[/database]`
> 
> Examples:
> - `redis://localhost:6379` - Basic local Redis
> - `redis://:mypassword@redis.example.com:6379` - With password
> - `redis://redis.internal:6379/1` - Using database index 1
> - `rediss://...` - TLS/SSL connection (note: double 's')

## 6. Launching Multiple vLLM Instances (with Shared Redis)

### 6.1 Start Instance A (Port 8000)

```bash
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=recipes/vllm_redis_remote.yaml \
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
LMCACHE_CONFIG_FILE=recipes/vllm_redis_remote.yaml \
CUDA_VISIBLE_DEVICES=1 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
--max-model-len 8192 \
--gpu-memory-utilization 0.85 \
--port 8001 \
--no-enable-prefix-caching \
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

> **Note:** Both instances use the same `LMCACHE_CONFIG_FILE` pointing to the same Redis. They share the cache namespace.

## 7. Startup Validation

### Instance A logs
```
LMCache INFO: Loading LMCache config file recipes/vllm_redis_remote.yaml
LMCache INFO: Creating LMCacheEngine with config:
  {'chunk_size': 256, 'remote_url': 'redis://localhost:6379', ...}
LMCache INFO: Initializing RedisBackend at localhost:6379
```

### Instance B logs
```
LMCache INFO: Loading LMCache config file recipes/vllm_redis_remote.yaml
LMCache INFO: Creating LMCacheEngine with config:
  {'chunk_size': 256, 'remote_url': 'redis://localhost:6379', ...}
LMCache INFO: Initializing RedisBackend at localhost:6379
```

### Verify Redis connection
```bash
# Check connected clients
redis-cli CLIENT LIST | grep -c "cmd="

# Should show 2+ connections (from both instances)
```

## 8. Inference and Cross-Instance Cache Validation

### 8.1 Warm cache on Instance A

```bash
# Send request to Instance A
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": "You are a helpful AI assistant.\n\nUser: Explain quantum computing in detail. This is a long prompt that will generate significant KV cache data that we want to share across instances.",
    "max_tokens": 100
  }'
```

Expected Instance A logs (cold, stores to Redis):
```
LMCache INFO: Reqid: ..., Total tokens 512, LMCache hit tokens: 0, need to load: 0
LMCache INFO: Stored 512 out of total 512 tokens. size: 0.0703 GB
```

Verify data in Redis:
```bash
# Check Redis keys (LMCache uses hash-based keys)
redis-cli KEYS "*" | wc -l
# Shows number of cached chunks

# Check memory usage
redis-cli INFO memory | grep used_memory_human
```

### 8.2 Get cache hit on Instance B

```bash
# Send SAME request to Instance B
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": "You are a helpful AI assistant.\n\nUser: Explain quantum computing in detail. This is a long prompt that will generate significant KV cache data that we want to share across instances.",
    "max_tokens": 100
  }'
```

Expected Instance B logs (warm, retrieves from Redis):
```
LMCache INFO: Reqid: ..., Total tokens 512, LMCache hit tokens: 512, need to load: 512
LMCache INFO: Retrieved 512 out of 512 required tokens. size: 0.0703 gb
```

**Key observation:** Instance B got a cache hit despite never having seen this prompt before, because the KV was stored to Redis by Instance A.

## 9. Benchmarking

### 9.1 Baseline (no sharing)

Run two instances WITHOUT Redis (local cache only):

```bash
# Instance A - send request
# Instance B - send same request
# Result: Both compute from scratch (no sharing)
```

### 9.2 With Redis sharing

```bash
# Instance A - warm cache
# Instance B - cache hit from Redis
```

### 9.3 Benchmark results

| Scenario | Instance A TTFT | Instance B TTFT | Improvement |
|----------|-----------------|-----------------|-------------|
| No sharing | ~600ms | ~600ms | None |
| Redis sharing | ~600ms | ~200ms | **~67% faster** |

### 9.4 Redis performance metrics

Monitor Redis during the test:

```bash
# In a separate terminal, watch Redis stats
watch -n 1 'redis-cli INFO stats | grep -E "keyspace_hits|keyspace_misses"'

# Expected: keyspace_hits increases when Instance B queries
```

## 10. Redis Configuration Tuning

### 10.1 Memory Management

```bash
# Set maxmemory (adjust based on your Redis server capacity)
redis-cli CONFIG SET maxmemory 32gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

### 10.2 Persistence (optional)

```bash
# Enable RDB snapshots for cache recovery
redis-cli CONFIG SET save "900 1 300 10 60 10000"

# Or use AOF for better durability
redis-cli CONFIG SET appendonly yes
```

### 10.3 Connection Pooling

LMCache automatically manages Redis connection pooling. For high-throughput scenarios, ensure:

```bash
# Check max clients
redis-cli CONFIG GET maxclients

# Increase if needed
redis-cli CONFIG SET maxclients 10000
```

## 11. Performance Tips

| Parameter | Recommendation | Impact |
|-----------|---------------|--------|
| Redis network | Same datacenter/VPC | Minimize latency |
| `chunk_size` | 256-512 | Balance granularity vs Redis ops |
| Redis memory | 2-3x expected cache size | Headroom for growth |
| Local + Remote | Enable both (see R-029) | Local hits are faster |
| Connection reuse | LMCache handles pooling | Don't create new connections per request |

## 12. Troubleshooting / Common Pitfalls

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| No cache hit on Instance B | Different `chunk_size` in configs | Ensure identical YAML on all instances |
| Redis connection refused | Wrong host/port or firewall | Check `remote_url` and security groups |
| Authentication failed | Password not in URL | Use `redis://:password@host:port` |
| Slow cache hits | Network latency to Redis | Co-locate Redis with vLLM instances |
| Redis OOM | `maxmemory` too small | Increase or set eviction policy |
| Cache key conflicts | Different models same Redis | Use separate Redis DBs or instances |
| Stale cache | Old model version cached | Clear Redis or use versioning |

### Clear Redis cache if needed

```bash
# Flush all LMCache data (WARNING: deletes everything)
redis-cli FLUSHDB

# Or use pattern delete (safer)
redis-cli --scan --pattern "*" | xargs -L 100 redis-cli DEL
```

### Debug Redis keys

```bash
# List all keys
redis-cli KEYS "*"

# Get key info
redis-cli DEBUG OBJECT <key>

# Check TTL
redis-cli TTL <key>
```

## 13. Advanced: Redis with Authentication

For production Redis with password:

```yaml
# vllm_redis_remote.yaml
chunk_size: 256
remote_url: "redis://:your_password@redis.example.com:6379/0"
use_layerwise: false
save_unfull_chunk: true
```

With TLS/SSL:

```yaml
remote_url: "rediss://:your_password@redis.example.com:6380/0"
```

## 14. Additional Resources
- CPU hot cache recipe: `recipes/dense_instruct_cpu_hot_cache.md`
- Disk persistence recipe: `recipes/vllm_disk_persistence.md`
- Tiered storage recipe: `recipes/vllm_tiered_storage.md` (R-029)
- Multi-instance sharing: `recipes/vllm_multi_instance_sharing.md` (R-018)
- Redis documentation: https://redis.io/documentation
