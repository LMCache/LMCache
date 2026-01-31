# LMCache Recipe Library - Master Index

**Complete Recipe Collection** | 32 Production-Ready Recipes | v1.0

---

## Quick Navigation

| Category | Recipes | Focus Area |
|----------|---------|------------|
| [A - Core Integrations](#category-a-core-integrations) | 5 recipes | vLLM, SGLang, Production Stack, KServe, llm-d |
| [B - Storage Backends](#category-b-storage-backends) | 12 recipes | CPU, Disk, GDS, NIXL, Redis, S3, Mooncake, InfiniStore |
| [C - Multi-Instance](#category-c-multi-instance-caching) | 3 recipes | Multiple instances, P2P, Controller |
| [D - PD Transport](#category-d-prefill-decode) | 3 recipes | Single-node, Multi-node, Tuning |
| [E - Optimizations](#category-e-optimizations) | 5 recipes | Layerwise, CacheGen, CacheBlend, Async, Multiprocess |
| [F - End-to-End](#category-f-end-to-end) | 4 recipes | Tiered storage, Multi-node shared, Enterprise, Ultimate PD |

---

## Category A: Core Integrations

### R-001: vLLM with CPU Hot Cache
[📖 Read Recipe](./R-001-vllm-cpu-hot-cache.md)

**Priority**: P1 | **Status**: ✅ Complete

Quick-start recipe for adding CPU-backed KV cache to vLLM. Ideal for first-time users.

```bash
# Key config
efficient_cpu_cache:
  local_cpu: true
  max_local_cpu_size: 50
  lookup_url: /tmp/lmcache_lookup.sock
```

---

### R-002: SGLang with CPU Hot Cache
[📖 Read Recipe](./R-002-sglang-cpu-hot-cache.md)

**Priority**: P1 | **Status**: ✅ Complete

Quick-start recipe for adding CPU-backed KV cache to SGLang.

```bash
# Key config
export SGLANG_ENABLE_LMCACHE=true
sglang_server --enable-lmcache \
  --lmcache-config /etc/sglang/lmcache.yaml
```

---

### R-003: Production Stack Multi-Node
[📖 Read Recipe](./R-003-production-stack-multi-node.md)

**Priority**: P1 | **Status**: ✅ Complete

Deploy LMCache across Kubernetes clusters using Production Stack Helm charts.

---

### R-004: KServe Integration
[📖 Read Recipe](./R-004-kserve-integration.md)

**Priority**: P1 | **Status**: ✅ Complete

Integrate LMCache with KServe InferenceService for enterprise Kubernetes deployments.

---

### R-005: llm-d Daemon Integration
[📖 Read Recipe](./R-005-llm-d-daemon-integration.md)

**Priority**: P2 | **Status**: ✅ Complete

Deploy LMCache with llm-d for distributed LLM serving across heterogeneous clusters.

---

## Category B: Storage Backends

### R-006: CPU Backend Deep Dive
[📖 Read Recipe](./R-006-cpu-backend.md)

**Priority**: P1 | **Status**: ✅ Complete

Comprehensive guide to CPU KV cache storage with sizing and performance tuning.

---

### R-007: Disk Backend
[📖 Read Recipe](./R-007-disk-backend.md)

**Priority**: P1 | **Status**: ✅ Complete

Use local SSD/NVMe for larger-than-RAM KV cache storage.

```yaml
efficient_cpu_cache:
  local_disk: /mnt/fastssd/lmcache
  max_local_disk_size: 500
```

---

### R-008: GPUDirect Storage (GDS)
[📖 Read Recipe](./R-008-gds-backend.md)

**Priority**: P2 | **Status**: ✅ Complete

Enable direct GPU-to-NVMe transfers for ultra-low latency disk caching.

---

### R-009: NIXL Backend
[📖 Read Recipe](./R-009-nixl-backend.md)

**Priority**: P1 | **Status**: ✅ Complete

High-performance network backend for multi-GPU and multi-node deployments.

```yaml
remote_cache:
  url: nixl://10.0.0.1:9000
  nixl_backends: [UCX, TCP]
```

---

### R-010: Redis Backend
[📖 Read Recipe](./R-010-redis-backend.md)

**Priority**: P1 | **Status**: ✅ Complete

Share KV caches across multiple LMCache instances with Redis.

```yaml
remote_cache:
  url: redis://redis-cluster:6379
```

---

### R-011: Redis Cluster Backend
[📖 Read Recipe](./R-011-redis-cluster-backend.md)

**Priority**: P2 | **Status**: ✅ Complete

Production Redis Cluster deployment for high availability.

```yaml
remote_cache:
  url: redis://node1:6379,node2:6379,node3:6379
```

---

### R-012: Redis Sentinel Backend
[📖 Read Recipe](./R-012-redis-sentinel-backend.md)

**Priority**: P2 | **Status**: ✅ Complete

Automatic failover with Redis Sentinel for enterprise deployments.

```yaml
remote_cache:
  url: redis://sentinel1:26379,sentinel2:26379/0
  redis_sentinel_master: mymaster
```

---

### R-013: Redis with TLS
[📖 Read Recipe](./R-013-redis-tls-backend.md)

**Priority**: P2 | **Status**: ✅ Complete

Encrypted Redis connections for security-sensitive environments.

---

### R-014: LMCache Server Backend
[📖 Read Recipe](./R-014-lmcache-server-backend.md)

**Priority**: P2 | **Status**: ✅ Complete

Centralized LMCache server for shared caching across distributed workers.

```yaml
remote_cache:
  url: lm://lmcache-server.internal:65432
```

---

### R-015: S3 Backend
[📖 Read Recipe](./R-015-s3-backend.md)

**Priority**: P2 | **Status**: ✅ Complete

Persistent KV cache storage in object storage (AWS S3, MinIO, GCS).

```yaml
remote_cache:
  url: s3://mybucket/lmcache/
  s3_region: us-east-1
```

---

### R-016: Mooncake Backend
[📖 Read Recipe](./R-016-mooncake-backend.md)

**Priority**: P2 | **Status**: ✅ Complete

Alibaba Mooncake storage for high-throughput distributed caching.

```yaml
remote_cache:
  url: mooncake://master:11000
```

---

### R-017: InfiniStore Backend
[📖 Read Recipe](./R-017-infinistore-backend.md)

**Priority**: P2 | **Status**: ✅ Complete

ByteDance InfiniStore for large-scale memory pool sharing.

```yaml
remote_cache:
  url: infinistore://10.0.0.1:12345
  infinistore_connection_type: RDMA
```

---

## Category C: Multi-Instance Caching

### R-018: Two Instance Sharing
[📖 Read Recipe](./R-018-two-instance-sharing.md)

**Priority**: P1 | **Status**: ✅ Complete

Basic pattern for sharing KV caches between two LMCache instances.

---

### R-019: P2P KV Sharing
[📖 Read Recipe](./R-019-p2p-kv-sharing.md)

**Priority**: P1 | **Status**: ✅ Complete

Peer-to-peer KV cache sharing without central server for ultra-low latency.

```yaml
p2p:
  enable: true
  host: 10.0.0.1
  init_ports: [12000, 12001]
  lookup_ports: [13000, 13001]
```

---

### R-020: Controller Orchestration
[📖 Read Recipe](./R-020-controller-orchestration.md)

**Priority**: P1 | **Status**: ✅ Complete

Centralized controller for managing multiple LMCache instances.

```yaml
controller:
  enable: true
  pull_url: http://10.0.0.1:8000
  lookup_url: /tmp/lmcache_controller.sock
```

---

## Category D: Prefill-Decode (PD) Transport

### R-021: Single-Node PD
[📖 Read Recipe](./R-021-single-node-pd.md)

**Priority**: P1 | **Status**: ✅ Complete

Basic prefill-decode disaggregation on a single node.

```yaml
prefill_decode:
  enable_pd: true
  transfer_channel: nixl
```

---

### R-022: Multi-Node PD
[📖 Read Recipe](./R-022-multi-node-pd.md)

**Priority**: P1 | **Status**: ✅ Complete

Scale prefill-decode across multiple nodes for large deployments.

```yaml
prefill_decode:
  enable_pd: true
  nixl_backends: [UCX, TCP]
```

---

### R-023: PD Performance Tuning
[📖 Read Recipe](./R-023-pd-performance-tuning.md)

**Priority**: P2 | **Status**: ✅ Complete

Optimize transfer latency, throughput, and resource utilization for PD.

---

## Category E: Optimizations

### R-024: Layerwise Storage
[📖 Read Recipe](./R-024-layerwise-storage.md)

**Priority**: P1 | **Status**: ✅ Complete

Enable layerwise KV cache storage for 40% memory efficiency improvement.

```yaml
optimizations:
  use_layerwise: true
  chunk_size: 256
```

---

### R-025: CacheGen Compression
[📖 Read Recipe](./R-025-cachegen-compression.md)

**Priority**: P1 | **Status**: ✅ Complete

4-8x KV cache compression with minimal quality loss.

```yaml
optimizations:
  enable_compression: true
  compression_algorithm: cachegen
```

---

### R-026: CacheBlend
[📖 Read Recipe](./R-026-cacheblend.md)

**Priority**: P1 | **Status**: ✅ Complete

Reuse KV caches from multiple source documents for RAG optimization.

```yaml
optimizations:
  enable_blending: true
```

---

### R-027: Async Operations
[📖 Read Recipe](./R-027-async-operations.md)

**Priority**: P2 | **Status**: ✅ Complete

Non-blocking store/retrieve operations for maximum throughput.

```yaml
optimizations:
  enable_async: true
  async_queue_size: 100
```

---

### R-028: Multiprocess Storage
[📖 Read Recipe](./R-028-multiprocess-storage.md)

**Priority**: P2 | **Status**: ✅ Complete

Dedicated storage process to reduce main process overhead.

```yaml
optimizations:
  enable_multiprocess: true
  max_local_cpu_size: 100
```

---

## Category F: End-to-End

### R-029: Tiered Storage Stack
[📖 Read Recipe](./R-029-tiered-storage-stack.md)

**Priority**: P2 | **Status**: ✅ Complete

Complete tiered storage: GPU → CPU → Disk with intelligent promotion.

```yaml
# All tiers enabled
efficient_cpu_cache:
  local_cpu: true
  max_local_cpu_size: 50
  local_disk: /mnt/fastssd/lmcache
  max_local_disk_size: 500
```

---

### R-030: Multi-Node Shared Cache
[📖 Read Recipe](./R-030-multi-node-shared-cache.md)

**Priority**: P2 | **Status**: ✅ Complete

Share KV caches across a compute cluster with distributed coordination.

---

### R-031: Enterprise Production Stack
[📖 Read Recipe](./R-031-enterprise-production-stack.md)

**Priority**: P3 | **Status**: ✅ Complete

Production-ready stack with monitoring, HA, and security hardening.

---

### R-032: Ultimate PD Pool
[📖 Read Recipe](./R-032-ultimate-pd-pool.md)

**Priority**: P3 | **Status**: ✅ Complete

Ultimate prefill-decode deployment combining all optimizations.

---

## Decision Matrix

### Which Recipe Should I Start With?

| Your Situation | Start Here | Next Steps |
|---------------|------------|------------|
| New to LMCache | [R-001: vLLM CPU Hot Cache](#r-001-vllm-with-cpu-hot-cache) | [R-006: CPU Deep Dive](#r-006-cpu-backend-deep-dive) |
| Using SGLang | [R-002: SGLang CPU Hot Cache](#r-002-sglang-with-cpu-hot-cache) | [R-010: Redis](#r-010-redis-backend) |
| Kubernetes deployment | [R-003: Production Stack](#r-003-production-stack-multi-node) | [R-031: Enterprise Stack](#r-031-enterprise-production-stack) |
| Need larger cache | [R-007: Disk Backend](#r-007-disk-backend) | [R-029: Tiered Storage](#r-029-tiered-storage-stack) |
| Multiple instances | [R-018: Two Instance Sharing](#r-018-two-instance-sharing) | [R-019: P2P Sharing](#r-019-p2p-kv-sharing) |
| Optimize for RAG | [R-026: CacheBlend](#r-026-cacheblend) | [R-025: CacheGen](#r-025-cachegen-compression) |
| Reduce prefill latency | [R-021: Single-Node PD](#r-021-single-node-pd) | [R-022: Multi-Node PD](#r-022-multi-node-pd) |
| Maximum performance | [R-032: Ultimate PD Pool](#r-032-ultimate-pd-pool) | [R-031: Enterprise Stack](#r-031-enterprise-production-stack) |

---

## Configuration Quick Reference

### Storage Mode (Persistent Cache)

```yaml
efficient_cpu_cache:
  local_cpu: true
  max_local_cpu_size: 50              # GB
  local_disk: /mnt/fastssd/lmcache    # Optional
  max_local_disk_size: 500            # GB
  remote_url: redis://host:6379       # Optional
```

### PD Mode (No Persistence)

```yaml
prefill_decode:
  enable_pd: true
  transfer_channel: nixl
  nixl_backends: [UCX, TCP]
```

**Note**: Storage mode and PD mode are mutually exclusive.

---

## Validation Checklist

Before deploying any recipe to production:

- [ ] YAML config validated with `python -c "import yaml; yaml.safe_load(open('config.yaml'))"`
- [ ] All required sections present in recipe (11 sections)
- [ ] Tested on non-production cluster first
- [ ] Resource limits configured (CPU, memory, disk)
- [ ] Monitoring and alerting configured
- [ ] Backup/restore procedures documented

---

## Resources

- **Documentation**: https://docs.lmcache.ai/
- **GitHub**: https://github.com/LMCache/LMCache
- **Slack**: https://join.slack.com/t/lmcacheworkspace/shared_invite
- **Blogs**: https://blog.lmcache.ai/

---

## Recipe Statistics

| Metric | Value |
|--------|-------|
| Total Recipes | 32 |
| P1 Priority | 12 (100% complete) |
| P2 Priority | 16 (100% complete) |
| P3 Priority | 4 (100% complete) |
| YAML Configs | 36 |
| Categories | 6 (A-F) |

---

*Last Updated: 2026-01-31*  
*Version: 1.0*  
*Branch: dense-instruct-cpu-hot-cache*
