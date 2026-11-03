# LMCache In-Process Examples

LMCache is embedded directly in the vLLM / SGLang process. Start vLLM with
`--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'`.

## Tier 1 — Core Concepts (Single-Node)

Focus: local caching within a single machine. No cross-node networking required.

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`kv_cache_reuse/local_backends/`](kv_cache_reuse/local_backends/) | Offload KV cache from GPU to CPU memory or local disk. Repeated requests with identical prefixes skip prefill entirely. Includes a Rust-based NVMe backend with `io_uring` support (Linux kernel 5.10+ required for `io_uring`). | 1 GPU; optional NVMe for disk path |
| [`../online_session/`](../online_session/) | Measure TTFT (time-to-first-token) for cold vs. cache-hit requests. Outputs JSONL for plotting. Includes a sweep script across context lengths. | 1 GPU |

## Tier 2 — Disaggregated Prefill and Multi-Instance Sharing

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`disagg_prefill/1p1d/`](disagg_prefill/1p1d/) | PD disaggregation with direct NIXL transfer: 1 prefill server + 1 decode server + a FastAPI proxy. Includes a benchmark script with expected latency numbers. High-bandwidth interconnect (NVLink or PCIe Gen4/5) is strongly recommended — without it, KV transfer overhead may negate the gains. | 2 GPUs + [NIXL](https://github.com/ai-dynamo/nixl) |
| [`disagg_prefill/xpyd/`](disagg_prefill/xpyd/) | xP + yD topology: multiple prefill servers feeding multiple decode servers. | 3+ GPUs + NIXL |
| [`kv_cache_reuse/share_across_instances/p2p_sharing/`](kv_cache_reuse/share_across_instances/p2p_sharing/) | Two vLLM instances transfer KV directly peer-to-peer using NIXL and UCX RDMA (`UCX_TLS=rc`). | 2 GPUs + NIXL + UCX |

## Tier 3 — Production Operations and Observability

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`chunk_statistics/`](chunk_statistics/) | Track chunk reuse rate using a memory Bloom filter or an on-disk hash log. Query via REST (`/chunk_statistics/status`). Use this to answer "does my workload benefit from caching?" before committing to a full deployment. | 1+ GPUs |
| [`cache_controller/`](cache_controller/) | REST APIs for cache orchestration: `lookup` (which instance holds a key), `move` (migrate a hot context between instances), `pin` (prevent eviction), `clear`, and `compress`. | 1 GPU |
| [`cache_with_configs/`](cache_with_configs/) and [`cache_interface/`](cache_interface/) | Per-request control: attach tags (`lmcache.tag.*`), set TTL (`lmcache.ttl`), or skip caching entirely (`lmcache.skip_save: true`). Essential for multi-tenant deployments. | 1 GPU |

## Framework Integrations

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`frontend/`](frontend/) | Streamlit chat UI demo backed by vLLM + LMCache. Uses the ffmpeg man page as a long shared context to demonstrate TTFT reduction across turns. | 1 GPU |

## Non-Prefix KV Reuse (CacheBlend)

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`blend/`](blend/) | CacheBlend in LMCache's in-process mode: reuse KV cache even when the new prompt is not a prefix of the cached one (e.g., RAG with swapped documents). Requires a small patch to vLLM source. | 1 GPU (experimental) |

## Developer Extensibility

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`runtime_plugins/`](runtime_plugins/) | Sidecar scripts (Python or shell) that run alongside LMCache workers: heartbeats, metric reporters, alert hooks. Filename prefix controls which role (`scheduler`, `worker_0`, `all`) runs the script. | None |
| [`basic_check/`](basic_check/) | CLI tool for verifying storage backend health and generating test keys. Useful in CI and for on-call diagnostics. | Optional GPU |
