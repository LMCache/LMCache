# LMCache Examples

This directory contains runnable examples organized by deployment mode:

- [`inprocess/`](inprocess/) — LMCache runs inside the vLLM / SGLang worker process
  (via the `LMCacheConnectorV1` connector or the in-process Python API).
- Existing multiprocess examples remain in their top-level directories because
  they already identify the standalone MP server mode in their names or docs.
- Other top-level entries are mode-agnostic tools, storage backends, and SDK demos.

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

## In-Process Mode ([`inprocess/`](inprocess/))

LMCache is embedded directly in the vLLM / SGLang process. Start vLLM with
`--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'`.

### Tier 1 — Core Concepts (Single-Node)

Focus: local caching within a single machine. No cross-node networking required.

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`inprocess/kv_cache_reuse/local_backends/`](inprocess/kv_cache_reuse/local_backends/) | Offload KV cache from GPU to CPU memory or local disk. Repeated requests with identical prefixes skip prefill entirely. Includes a Rust-based NVMe backend with `io_uring` support (Linux kernel 5.10+ required for `io_uring`). | 1 GPU; optional NVMe for disk path |
| [`online_session/`](online_session/) | Measure TTFT (time-to-first-token) for cold vs. cache-hit requests. Outputs JSONL for plotting. Includes a sweep script across context lengths. | 1 GPU |

### Tier 2 — Disaggregated Prefill and Multi-Instance Sharing

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`inprocess/disagg_prefill/1p1d/`](inprocess/disagg_prefill/1p1d/) | PD disaggregation with direct NIXL transfer: 1 prefill server + 1 decode server + a FastAPI proxy. Includes a benchmark script with expected latency numbers. High-bandwidth interconnect (NVLink or PCIe Gen4/5) is strongly recommended — without it, KV transfer overhead may negate the gains. | 2 GPUs + [NIXL](https://github.com/ai-dynamo/nixl) |
| [`inprocess/disagg_prefill/xpyd/`](inprocess/disagg_prefill/xpyd/) | xP + yD topology: multiple prefill servers feeding multiple decode servers. | 3+ GPUs + NIXL |
| [`inprocess/kv_cache_reuse/share_across_instances/p2p_sharing/`](inprocess/kv_cache_reuse/share_across_instances/p2p_sharing/) | Two vLLM instances transfer KV directly peer-to-peer using NIXL and UCX RDMA (`UCX_TLS=rc`). | 2 GPUs + NIXL + UCX |

### Tier 3 — Production Operations and Observability

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`inprocess/chunk_statistics/`](inprocess/chunk_statistics/) | Track chunk reuse rate using a memory Bloom filter or an on-disk hash log. Query via REST (`/chunk_statistics/status`). Use this to answer "does my workload benefit from caching?" before committing to a full deployment. | 1+ GPUs |
| [`inprocess/cache_controller/`](inprocess/cache_controller/) | REST APIs for cache orchestration: `lookup` (which instance holds a key), `move` (migrate a hot context between instances), `pin` (prevent eviction), `clear`, and `compress`. | 1 GPU |
| [`inprocess/cache_with_configs/`](inprocess/cache_with_configs/) and [`inprocess/cache_interface/`](inprocess/cache_interface/) | Per-request control: attach tags (`lmcache.tag.*`), set TTL (`lmcache.ttl`), or skip caching entirely (`lmcache.skip_save: true`). Essential for multi-tenant deployments. | 1 GPU |

### Framework Integrations

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`inprocess/frontend/`](inprocess/frontend/) | Streamlit chat UI demo backed by vLLM + LMCache. Uses the ffmpeg man page as a long shared context to demonstrate TTFT reduction across turns. | 1 GPU |

### Non-Prefix KV Reuse (CacheBlend)

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`inprocess/blend/`](inprocess/blend/) | CacheBlend in LMCache's in-process mode: reuse KV cache even when the new prompt is not a prefix of the cached one (e.g., RAG with swapped documents). Requires a small patch to vLLM source. | 1 GPU (experimental) |

### Developer Extensibility

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`inprocess/runtime_plugins/`](inprocess/runtime_plugins/) | Sidecar scripts (Python or shell) that run alongside LMCache workers: heartbeats, metric reporters, alert hooks. Filename prefix controls which role (`scheduler`, `worker_0`, `all`) runs the script. | None |
| [`inprocess/basic_check/`](inprocess/basic_check/) | CLI tool for verifying storage backend health and generating test keys. Useful in CI and for on-call diagnostics. | Optional GPU |

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

### SDK and Advanced Features

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`kvcache_sdk/`](kvcache_sdk/) | Python SDK examples for storing and retrieving KV cache tensors through the LMCache MP HTTP API. | 1 GPU |
| [`token_dropping/`](token_dropping/) | Google Colab notebooks for token-dropping eviction strategies (SnapKV, RKV) using the LMCache SDK. | 1 GPU |
| [`mp_runtime_plugins/`](mp_runtime_plugins/) | MP-sidecar runtime plugins: heartbeat, metric reporter, alert hooks. | None |

---

## Mode-Agnostic (Top Level)

These examples, backends, and tools work with either in-process or multiprocess deployment.

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
