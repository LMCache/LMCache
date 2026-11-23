# LMCache Examples

This directory contains runnable multiprocess examples, mode-agnostic tools,
storage backends, and SDK demos.

> **Single-node vs. multi-node:** Examples marked with NIXL or UCX require a
> high-bandwidth interconnect (NVLink or PCIe Gen4/5). Running them on a
> single machine with two GPUs is simpler than a true multi-node setup, but
> cross-node deployments add network configuration complexity — don't start
> there if you're still learning the basics.

---

## Before You Start

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`kv_cache_calculator/`](kv_cache_calculator/) | Web UI for calculating KV cache size (GB) given model architecture, dtype, and token count. Start here to size GPU memory and cache tiers before running anything. | None (browser) |

---

## Multiprocess Mode (Top-Level Directories)

LMCache runs as a standalone server (`lmcache server`). Engines connect to it
over gRPC / HTTP.

### Tier 2 — Disaggregated Prefill and Multi-Instance Sharing

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`disagg_prefill_mp/`](disagg_prefill_mp/) | PD disaggregation via the LMCache multiprocess server. P and D exchange KV through the LMCache MP service — no direct NIXL connection required. **Recommended starting point for PD disaggregation.** | 2 GPUs |
| [`kv_cache_reuse/share_across_engines/`](kv_cache_reuse/share_across_engines/) | An SGLang instance computes a prompt KV cache and a vLLM instance reuses it through one LMCache multiprocess server. Includes a pinned `uv` setup and an automated correctness check. | 2 GPUs (80 GB each for the validated Qwen 32B model) |
| [`p2p/`](p2p/) | P2P KV cache sharing in multiprocess mode: each node runs an LMCache server, and a server reads a prefix it lacks directly from the peer that holds it over RDMA. A coordinator handles peer discovery. Includes single-node (debug) and multi-node setups and the logs to expect. | 2+ GPUs (single- or multi-node) + RDMA fabric |

### Tier 3 — Production Operations and Observability

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`observability/`](observability/) | Full observability stack: OTel → Prometheus + Tempo → Grafana. Pre-provisioned dashboard shows cache hit rate, read/write throughput, and per-request trace waterfalls. Started with `docker compose up`. | 1 GPU + Docker |
| [`multi_process/`](multi_process/) | Kubernetes DaemonSet YAML for deploying the LMCache server as a per-node sidecar (60 GB L1, 4 workers). Includes resource `requests`/`limits` calibrated to the L1 size. | Kubernetes cluster with GPU nodes |
| [`remote_config_server/`](remote_config_server/) | Flask reference server for centralised LMCache configuration. Workers POST their current config on startup and receive overrides — useful for managing settings across a large fleet. | None |

### Framework Integrations

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`sgl_integration/`](sgl_integration/) | Using LMCache with SGLang instead of vLLM (connecting to the standalone MP server). | 1 GPU |
| [`dynamo_integration/`](dynamo_integration/) | Recipes for running NVIDIA Dynamo vLLM serving with KV cache offloaded to an LMCache MP server. | 2+ GPUs |
| [`llm_d_integration/`](llm_d_integration/) | Kubernetes manifests for vLLM pods behind the llm-d Router with KV cache in an LMCache MP server: prefix KV reuse and P/D disaggregation. | Kubernetes cluster, 1 GPU (2 GPUs on one node for P/D) |

### SDK and Advanced Features

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`kvcache_sdk/`](kvcache_sdk/) | Python SDK examples for storing and retrieving KV cache tensors through the LMCache MP HTTP API. | 1 GPU |
| [`token_dropping/`](token_dropping/) | Google Colab notebooks for token-dropping eviction strategies (SnapKV, RKV) using the LMCache SDK. | 1 GPU |
| [`mp_runtime_plugins/`](mp_runtime_plugins/) | MP-sidecar runtime plugins: heartbeat, metric reporter, alert hooks. | None |

---

## Mode-Agnostic (Top Level)

These examples, backends, and tools are not tied to a specific deployment mode.

### Storage Backends and Serialization

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`kv_cache_reuse/remote_backends/`](kv_cache_reuse/remote_backends/) | Remote storage backends: InfiniStore, Mooncake, S3, Valkey, Redis. Each subdirectory has its own README with backend-specific setup. | 1 GPU + respective backend |
| [`kv_cache_reuse/share_across_instances/centralized_sharing/`](kv_cache_reuse/share_across_instances/centralized_sharing/) | Two vLLM instances share KV cache through Redis. A prefix computed by instance A is reused by instance B. ⚠️ Requires `PYTHONHASHSEED=0` in all processes — without this, hashes differ across processes and cache lookups silently miss. | 2 GPUs (same node) + Redis |
| [`serde/fp8/`](serde/fp8/) | Quantize KV cache to fp8 before writing to disk (L2 adapter), then dequantize on prefetch. Halves disk storage requirements. | Hopper / Ada GPU (H100, RTX 40-series) |
| [`redis_lookup/`](redis_lookup/) | Shows the Redis key schema used by LMCache (`model@world_size@worker_id@chunk_hash`) and `redis-cli` commands for inspecting live cache entries. | Redis + 1 GPU |

### Developer Extensibility

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`lmc_external_l2_adapter/`](lmc_external_l2_adapter/) and [`lmc_external_native_connector/`](lmc_external_native_connector/) | Templates for writing a custom L2 storage adapter (Python) or a native C++ GPU connector plugin. | Depends on implementation |
| [`agents/`](agents/) | Script for analyzing prefix-hash distribution of a prompt dataset. Useful for estimating cache efficiency before deployment. | None |
