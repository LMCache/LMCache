# L0 E2E Validation Plan After LMCache 0.4.6

## Purpose

This note defines the next L0 end-to-end validation target for MP
observability after LMCache 0.4.6.

The goal is to prove the complete reuse path:

```text
Mooncake or another L2 adapter
  -> L1 CPU memory prefetch
  -> L0 GPU KV block retrieve
  -> vLLM-visible cached-token stats
  -> Prometheus/OTel evidence
```

This is intentionally stronger than PR #3255. PR #3255 proves that vLLM block
allocation reports cross into the LMCache MP metrics pipeline. The E2E target
must prove that cache data actually moved from L2/L1 into GPU KV blocks and was
credited in the request result.

## 0.4.6 Release Implications

LMCache 0.4.6 changed the shape of the right solution:

- L2 adapters must now report real bytes written on the store path through
  `L2StoreResult`.
- MP observability metric names and units were refactored, so new dashboards
  must use the `lmcache_mp.*` metric surface.
- MP mode now has an L2 adapter benchmark CLI.
- StorageManager exports L2 usage via `lmcache_mp.l2_usage_bytes`.
- The MP connector reports cached token stats in `KVTransferParams`.
- MP KV transfer supports HND formats.

The L0 E2E plan should build on these surfaces. It should not add a separate
ad-hoc JSONL evidence path unless maintainers explicitly ask for file-based
boundary evidence.

## Existing Evidence Surfaces

### L2 to L1

Current events and metrics:

- `L2_PREFETCH_LOOKUP_SUBMITTED`
- `L2_PREFETCH_LOOKUP_COMPLETED`
- `L2_LOAD_TASK_SUBMITTED`
- `L2_LOAD_TASK_COMPLETED`
- `L2_PREFETCH_LOAD_SUBMITTED`
- `L2_PREFETCH_LOAD_COMPLETED`
- `lmcache_mp.l2_prefetch_hit`
- `lmcache_mp.l2_load_completed`
- `lmcache_mp.l2_load_throughput`
- `lmcache_mp.num_inflight_l2_loads`
- `lmcache_mp.inflight_load_memory_usage_bytes`
- `lmcache_mp.l2_usage_bytes`

Important limitation: store throughput uses real transferred bytes from
`L2StoreResult`, but load throughput currently uses submitted `total_bytes`.
That is enough for a first E2E validation, but not enough to prove real loaded
bytes if an adapter can partially satisfy or fast-path loads.

### L1 CPU to L0 GPU

Current events and metrics:

- `MP_RETRIEVE_SUBMITTED`
- `MP_RETRIEVE_START`
- `MP_RETRIEVE_END`
- `lmcache_mp.l0_l1_load_throughput`
- `lmcache_mp.l0_l1_load_requests`
- `lmcache_mp.l0_l1_load_bytes`
- `lmcache_mp.num_chunks_loaded`

`MP_RETRIEVE_END` carries:

- `retrieved_count`
- `engine_id`
- `model_name`
- `cache_salt`
- `total_bytes`
- `device`

This is the core CPU-to-GPU proof surface.

### L0 Ownership

Current events and metrics:

- `MP_VLLM_BLOCK_ALLOCATION`
- `MP_VLLM_END_SESSION`
- `lmcache_mp.l0_block_allocation_records`
- `lmcache_mp.l0_block_allocated_blocks`
- sampled L0 lifecycle histograms

These prove vLLM-reported GPU block ownership crosses the MP boundary. They do
not prove L2 or CPU-to-GPU data movement by themselves.

### vLLM Request Result

The MP connector can return `cached_token_stats` through `KVTransferParams`:

- `num_vllm_cached_tokens`
- `num_lmcache_cached_tokens`
- `num_lmcache_extra_cached_tokens`

For L0 E2E, `num_lmcache_extra_cached_tokens > 0` is the request-level proof
that LMCache supplied tokens beyond vLLM APC.

## Required E2E Scenario

Use an MP vLLM deployment with an L2 adapter. Prefer Mooncake for the final
target, but keep a cheaper mock/fs adapter lane for CI and iteration.

### Pass A: Populate L2

1. Start LMCache MP server with metrics enabled and one L2 adapter.
2. Start vLLM with `LMCacheMPConnector`.
3. Send a repeated-prefix request that stores chunks.
4. Wait until L2 store work drains.
5. Scrape metrics.

Required proof:

- `lmcache_mp_l2_store_completed_requests_total{l2_name="..."}` increases.
- `lmcache_mp_l2_store_throughput_GB_per_second{l2_name="..."}` has a nonzero sample.
- `lmcache_mp_l2_usage_bytes{l2_name="..."}` increases or remains positive.
- `lmcache_mp_l0_l1_store_throughput_GB_per_second` has a nonzero sample.

### Pass B: Force L2 to L1 to L0 Reuse

1. Clear or bypass local L1 hot-cache state while preserving L2, or run with an
   L1 capacity small enough that the second request must prefetch from L2.
2. Send the same repeated-prefix request.
3. Request `cached_token_stats`.
4. Scrape metrics after the request completes.

Required proof:

- `lmcache_mp_lookup_hit_tokens_total` increases for the request model/salt.
- `lmcache_mp_l2_prefetch_hit_chunks_total` increases.
- `lmcache_mp_l2_load_completed_requests_total{l2_name="..."}` increases.
- `lmcache_mp_l2_load_throughput_GB_per_second{l2_name="..."}` has a nonzero sample.
- `lmcache_mp_l0_l1_load_throughput_GB_per_second` has a nonzero sample.
- `lmcache_mp_l0_l1_load_requests_total` increases.
- `lmcache_mp_l0_l1_load_bytes_total` increases.
- `lmcache_mp_num_chunks_loaded_total` increases.
- `lmcache_mp_l0_block_allocation_records_total` and
  `lmcache_mp_l0_block_allocated_blocks_total` increase for the same worker/model.
- The request returns `num_lmcache_extra_cached_tokens > 0`.

## What We Need To Add

### P0: E2E Harness

Add a runnable E2E harness that launches:

- LMCache MP server with metrics.
- vLLM using `LMCacheMPConnector`.
- one L2 adapter.
- a repeated-prefix workload.
- a metrics scrape and assertion step.

The harness should support two modes:

- `--adapter fs` or `--adapter mock` for CI/CPU-safe structure validation.
- `--adapter mooncake` for the final Mooncake lane.

The Mooncake lane should use TCP first for portability, then RDMA as a separate
performance lane. RDMA should not block the correctness lane.

### P0: Metrics Assertion Script

Add a small parser that reads a Prometheus scrape and asserts the required
series are present and nonzero. Keep it independent from the launch harness so
Modal, Buildkite, or a manual GPU host can reuse it.

Current script:

```bash
python tools/mp_observability/assert_l0_e2e_metrics.py scrape.prom
python tools/mp_observability/assert_l0_e2e_metrics.py scrape.prom --scope full-e2e
```

Required assertion groups:

- L2 store proof.
- L2 load proof.
- L1 CPU to L0 GPU retrieve proof.
- L0 block allocation proof.
- request result proof from `cached_token_stats`.

### P0: Direct L0 CPU-to-GPU Counters

Add direct counters on `MP_RETRIEVE_END` so a Prometheus scrape can prove that
CPU-to-GPU retrieval happened even when histogram buckets are hard to interpret:

- `lmcache_mp.l0_l1_load_requests`: one completed retrieve that loaded at least
  one chunk.
- `lmcache_mp.l0_l1_load_bytes`: `total_bytes` copied from L1 CPU memory into
  L0 GPU KV blocks.

These counters complement `lmcache_mp.l0_l1_load_throughput` and
`lmcache_mp.num_chunks_loaded`; they do not add a new event or evidence side
channel.

### P1: Real Loaded Bytes For L2 Loads

Extend the L2 load path to report real loaded bytes, matching the 0.4.6 store
path improvement.

Candidate shape:

- Add an immutable `L2LoadResult` or extend load result metadata without
  breaking the bitmap contract.
- Publish `bytes_transferred` on `L2_LOAD_TASK_COMPLETED`.
- Teach `L2ThroughputSubscriber` to prefer completed real bytes over submitted
  bytes, just as store does.

This matters for Mooncake because partial loads, missing keys, or adapter
fast-paths should not inflate L2 load throughput.

### P1: Better Correlation Labels

The existing surfaces are close, but the E2E is easier to prove if every path
can be grouped by request/model/worker:

- Add `model_name` and `cache_salt` to L2 prefetch/load task events when the
  keys are available.
- Consider adding `request_id` to `MP_RETRIEVE_END` attributes through metrics
  only if it is safe for cardinality. For production dashboards, avoid high
  cardinality; for test scrapes, the event recorder or trace export is safer.

### P1: Mooncake MP L2 Documentation

Current Mooncake docs describe the legacy `LMCacheConnectorV1` remote backend
path. Add a small MP-specific section once the harness is stable:

- MP connector config.
- L2 adapter config for Mooncake.
- TCP correctness setup.
- RDMA performance setup.
- expected LMCache and Mooncake metrics.

### P2: Dashboard Panel

Add a Grafana row that tracks the E2E chain:

- L2 usage.
- L2 store/load throughput by `l2_name`.
- L2 prefetch hit chunks.
- L0/L1 load throughput by worker/device/model.
- chunks loaded into engine.
- L0 block allocation counters.
- LMCache extra cached tokens, if exposed by request logs or a future metric.

## Non-Goals

- Do not merge CacheBlend GPU data-path proof into the L0 allocation-boundary
  PR. CacheBlend has separate events and counters.
- Do not treat vLLM CPU offload metrics as LMCache L2 or L0 proof.
- Do not require RDMA for the correctness lane.
- Do not claim complete Mooncake production readiness from fs/mock adapter
  validation.

## Suggested PR Split

1. Keep PR #3255 focused on L0 allocation-boundary counters and unit tests.
2. Add the E2E harness and Prometheus assertion script.
3. Add real loaded-byte accounting for L2 loads.
4. Add MP Mooncake documentation and dashboard panels.
5. Run the final Mooncake E2E lane on a GPU host.

## Completion Criteria

The L0 E2E goal is complete only when a saved run artifact shows:

- exact LMCache and vLLM commits;
- adapter type and config;
- request workload;
- Prometheus scrape;
- request result with `num_lmcache_extra_cached_tokens > 0`;
- nonzero L2 store, L2 load, L0/L1 retrieve, engine loaded chunks, and L0 block
  allocation metrics;
- no EventBus dropped events or subscriber exceptions;
- Mooncake master metrics/logs for the Mooncake lane.
