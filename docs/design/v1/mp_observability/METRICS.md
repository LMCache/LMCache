# LMCache MP Mode Observability Metrics

## Overview

The observability system uses an **EventBus with pub/sub dispatch** and
**OpenTelemetry** for metrics instrumentation.

- **Producers** (`L1Manager`, `StorageManager`, `MPCacheEngine`) publish `Event` objects
  to the EventBus.
- **Metrics subscribers** (`L1MetricsSubscriber`, `SMMetricsSubscriber`) subscribe to
  specific event types and update OTel counters.
- **Logging subscribers** (`MPServerLoggingSubscriber`) log events at debug level.
- **Tracing subscribers** (`MPServerTracingSubscriber`) create OTel spans from START/END pairs.
- **Export** is via OTLP push to an OTel collector (production) or an in-process
  Prometheus `/metrics` endpoint (dev/debug fallback).

All metrics use the `lmcache_mp.` prefix (mp = multiprocess), distinct from the main
engine's `lmcache.` namespace. On Prometheus, `.` is converted to `_` and counters get
a `_total` suffix (e.g., `lmcache_mp_l1_read_keys_total`).

For implementation guidance on adding new events and subscribers, see [README.md](README.md).

## Global Resource Attributes

Every metric (and span) exported by an MP server carries Resource-level
attributes built at startup:

| Attribute | CLI flag | Source | Applies to |
|---|---|---|---|
| `service.instance.id` | `--service-instance-id` | `ObservabilityConfig.service_instance_id` (`None` defaults to a random UUID v4; explicit `""` preserved) | All metrics + spans |

Resource attributes are attached to the `MeterProvider` / `TracerProvider`
in `otel_init.py` and therefore appear on every datapoint exported via
OTLP.  On Prometheus, SDK resource attributes are typically surfaced via
the `target_info` series rather than on each time-series.

Per-metric attributes (e.g. `cache_salt`) remain on the individual
datapoints and are orthogonal to these Resource attributes.

---

## StorageManager Read Metrics

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_mp.sm_read_requests` | `lmcache_mp_sm_read_requests_total` | Counter | `SM_READ_PREFETCHED` | +1 per event |
| `lmcache_mp.sm_read_succeed_keys` | `lmcache_mp_sm_read_succeed_keys_total` | Counter | `SM_READ_PREFETCHED` | `+len(succeeded_keys)` |
| `lmcache_mp.sm_read_failed_keys` | `lmcache_mp_sm_read_failed_keys_total` | Counter | `SM_READ_PREFETCHED` | `+len(failed_keys)` |

**What it answers:** How often does the StorageManager receive read requests? What is the L1 hit rate?

> **Note:** `SM_READ_PREFETCHED_FINISHED` is published but has no metrics subscriber —
> it is available for logging subscribers only.

---

## StorageManager Write Metrics

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_mp.sm_write_requests` | `lmcache_mp_sm_write_requests_total` | Counter | `SM_WRITE_RESERVED` | +1 per event |
| `lmcache_mp.sm_write_succeed_keys` | `lmcache_mp_sm_write_succeed_keys_total` | Counter | `SM_WRITE_RESERVED` | `+len(succeeded_keys)` |
| `lmcache_mp.sm_write_failed_keys` | `lmcache_mp_sm_write_failed_keys_total` | Counter | `SM_WRITE_RESERVED` | `+len(failed_keys)` |

**What it answers:** How often are writes attempted? What fraction fail due to OOM or write conflicts?

> **Note:** `SM_WRITE_FINISHED` is published but has no metrics subscriber.

---

## L1 Read Metrics

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_mp.l1_read_keys` | `lmcache_mp_l1_read_keys_total` | Counter | `L1_READ_FINISHED` | `+len(keys)` |

**What it answers:** How many keys are being read from L1?

> **Note:** `L1_READ_RESERVED` is published but has no metrics subscriber — key counts
> are recorded only when the read actually completes.

---

## L1 Write Metrics

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_mp.l1_write_keys` | `lmcache_mp_l1_write_keys_total` | Counter | `L1_WRITE_FINISHED` | `+len(keys)` |
| *(same counter)* | *(same)* | Counter | `L1_WRITE_FINISHED_AND_READ_RESERVED` | `+len(keys)` |

**What it answers:** How many keys are being written to L1?

> **Note:** `L1_WRITE_RESERVED` is published but has no metrics subscriber.
> `L1_WRITE_FINISHED_AND_READ_RESERVED` (atomic write-then-read used by prefetch)
> increments the same write counter.

---

## L1 Eviction Metrics

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_mp.l1_evicted_keys` | `lmcache_mp_l1_evicted_keys_total` | Counter | `L1_KEYS_EVICTED` | `+len(keys)` |

**What it answers:** How aggressively is the eviction controller clearing L1? A high eviction rate relative to writes signals memory pressure.

---

## L1 Chunk Lifecycle Histograms

Sampled (default 1%) chunk-level lifecycle tracking.  Only sampled chunks
contribute to histograms; counters above always count all events.

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_mp.l1_chunk_lifetime_seconds` | `lmcache_mp_l1_chunk_lifetime_seconds` | Histogram | `L1_KEYS_EVICTED` | `eviction_time - alloc_time` per sampled chunk |
| `lmcache_mp.l1_chunk_idle_before_evict_seconds` | `lmcache_mp_l1_chunk_idle_before_evict_seconds` | Histogram | `L1_KEYS_EVICTED` | `eviction_time - last_access_time` per sampled chunk |
| `lmcache_mp.l1_chunk_reuse_gap_seconds` | `lmcache_mp_l1_chunk_reuse_gap_seconds` | Histogram | `L1_READ_FINISHED`, `L1_WRITE_FINISHED`, `L1_WRITE_FINISHED_AND_READ_RESERVED` | Time gap between consecutive touches of the same chunk |
| `lmcache_mp.l1_chunk_evict_reuse_gap_seconds` | `lmcache_mp_l1_chunk_evict_reuse_gap_seconds` | Histogram | `L1_KEYS_EVICTED` → `L1_WRITE_FINISHED` | Time from eviction to next reuse (capped at 300 s) |

**What it answers:** How long do L1 chunks live? How idle are they before eviction? How quickly are evicted chunks reused?

---


## L2 Store Metrics

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_mp.l2_store_tasks` | `lmcache_mp_l2_store_tasks_total` | Counter | `L2_STORE_SUBMITTED` | +1 per event |
| `lmcache_mp.l2_store_keys` | `lmcache_mp_l2_store_keys_total` | Counter | `L2_STORE_SUBMITTED` | `+key_count` |
| `lmcache_mp.l2_store_completed` | `lmcache_mp_l2_store_completed_total` | Counter | `L2_STORE_COMPLETED` | +1 per event |
| `lmcache_mp.l2_store_succeeded_keys` | `lmcache_mp_l2_store_succeeded_keys_total` | Counter | `L2_STORE_COMPLETED` | `+succeeded_count` |
| `lmcache_mp.l2_store_failed_keys` | `lmcache_mp_l2_store_failed_keys_total` | Counter | `L2_STORE_COMPLETED` | `+failed_count` |

**What it answers:** How many keys are being pushed to L2? What fraction fail?

---

## L2 Prefetch Metrics

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_mp.l2_prefetch_lookups` | `lmcache_mp_l2_prefetch_lookups_total` | Counter | `L2_PREFETCH_LOOKUP_SUBMITTED` | +1 per event |
| `lmcache_mp.l2_prefetch_lookup_keys` | `lmcache_mp_l2_prefetch_lookup_keys_total` | Counter | `L2_PREFETCH_LOOKUP_SUBMITTED` | `+key_count` |
| `lmcache_mp.l2_prefetch_hit_keys` | `lmcache_mp_l2_prefetch_hit_keys_total` | Counter | `L2_PREFETCH_LOOKUP_COMPLETED` | `+prefix_hit_count` |
| `lmcache_mp.l2_prefetch_load_tasks` | `lmcache_mp_l2_prefetch_load_tasks_total` | Counter | `L2_PREFETCH_LOAD_SUBMITTED` | `+adapter_count` |
| `lmcache_mp.l2_prefetch_load_keys` | `lmcache_mp_l2_prefetch_load_keys_total` | Counter | `L2_PREFETCH_LOAD_SUBMITTED` | `+key_count` |
| `lmcache_mp.l2_prefetch_loaded_keys` | `lmcache_mp_l2_prefetch_loaded_keys_total` | Counter | `L2_PREFETCH_LOAD_COMPLETED` | `+loaded_count` |
| `lmcache_mp.l2_prefetch_failed_keys` | `lmcache_mp_l2_prefetch_failed_keys_total` | Counter | `L2_PREFETCH_LOAD_COMPLETED` | `+failed_count` |

**What it answers:** How effective is L2 prefetching? What is the L2 hit rate? How many keys fail to load?

---

## L0 (GPU) Block Lifecycle Histograms

Sampled (default 1%) GPU KV cache block lifecycle tracking via shadow monitoring
of `MP_VLLM_BLOCK_ALLOCATION` and `MP_VLLM_END_SESSION` events.  Eviction is
detected at reallocation time (when a block is assigned different tokens).

All L0 histograms carry `instance_id` and `model_name` OTel attributes, enabling
per-instance and per-model Prometheus metric slicing (e.g.
`lmcache_mp_l0_block_lifetime_seconds{instance_id="12345",model_name="llama-7b"}`).

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_mp.l0_block_lifetime_seconds` | `lmcache_mp_l0_block_lifetime_seconds` | Histogram | `MP_VLLM_BLOCK_ALLOCATION` (eviction detected) | `eviction_time - alloc_time` per sampled block |
| `lmcache_mp.l0_block_idle_before_evict_seconds` | `lmcache_mp_l0_block_idle_before_evict_seconds` | Histogram | `MP_VLLM_BLOCK_ALLOCATION` (eviction detected) | `eviction_time - last_access_time` per sampled block |
| `lmcache_mp.l0_block_reuse_gap_seconds` | `lmcache_mp_l0_block_reuse_gap_seconds` | Histogram | `MP_VLLM_BLOCK_ALLOCATION` (cache hit) | Time gaps between consecutive accesses from access history |

**What it answers:** How long do GPU blocks live before eviction? How idle are they? How frequently are cached blocks reused? Which instance/model is experiencing the most churn?

---

## L0 ↔ L1 Throughput Histograms

Sampled (default 1%) per-request throughput of GPU↔CPU copies via
`L0L1ThroughputSubscriber`. Correlates `MP_{STORE,RETRIEVE}_START` → `MP_{STORE,RETRIEVE}_END`
pairs by `session_id`, computes `total_bytes / (end_ts - start_ts)` in GB/s.
START/END events fire on the GPU cupy stream (`publish_on_stream`), so
timestamps reflect true GPU-stream copy time — not Python/lock overhead.

All throughput histograms carry `engine_id` (vLLM worker instance id),
`device` (e.g. `"cuda:3"`), and `model_name` OTel attributes, enabling
per-worker, per-device, and per-model slicing in Prometheus (e.g.
`lmcache_mp_l0_l1_store_throughput_gbs{engine_id="0",device="cuda:3",model_name="meta-llama/Llama-3.1-8B"}`).

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_mp.l0_l1_store_throughput_gbs` | `lmcache_mp_l0_l1_store_throughput_gbs` | Histogram | `MP_STORE_START` → `MP_STORE_END` | `total_bytes / (end_ts - start_ts) / 1e9` per sampled request |
| `lmcache_mp.l0_l1_load_throughput_gbs` | `lmcache_mp_l0_l1_load_throughput_gbs` | Histogram | `MP_RETRIEVE_START` → `MP_RETRIEVE_END` | `total_bytes / (end_ts - start_ts) / 1e9` per sampled request |

**What it answers:** What GPU↔CPU throughput is each vLLM worker actually
achieving for KV store/load? Does it match the theoretical PCIe bandwidth?
Are some workers or GPUs underperforming?

---

## MPCacheEngine Observable Gauges

These metrics are registered directly via `register_gauge` (pull-based OTel
observable gauges) rather than through the EventBus, because they represent
point-in-time state snapshots that do not correspond to discrete events.

| OTel metric name | Prometheus name | Type | Source | Calculation |
|---|---|---|---|---|
| `lmcache_mp.active_prefetch_jobs` | `lmcache_mp_active_prefetch_jobs` | ObservableGauge | `MPCacheEngine._prefetch_jobs` | `len(_prefetch_jobs)` at scrape time |

**What it answers:** How many prefetch jobs are currently in-flight? A sustained high value may indicate slow L2 backends or client-side polling delays.

---

## Cache Blending (CB) Metrics

Metrics for Cache Blending operations use the `lmcache_blend.` prefix (distinct from the
MP mode `lmcache_mp.` namespace).  On Prometheus, `.` becomes `_` and counters get
`_total` suffix (e.g., `lmcache_blend_lookup_requests_total`).

### CB Lookup Metrics

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_blend.lookup_requests` | `lmcache_blend_lookup_requests_total` | Counter | `CB_LOOKUP_START` | +1 per event |
| `lmcache_blend.lookup_fingerprint_hits` | `lmcache_blend_lookup_fingerprint_hits_total` | Counter | `CB_LOOKUP_END` | `+fingerprint_hits` |
| `lmcache_blend.lookup_storage_hits` | `lmcache_blend_lookup_storage_hits_total` | Counter | `CB_LOOKUP_END` | `+storage_hits` |
| `lmcache_blend.lookup_stale_chunks` | `lmcache_blend_lookup_stale_chunks_total` | Counter | `CB_LOOKUP_END` | `+stale_chunks` |
| `lmcache_blend.lookup_no_gpu_context_errors` | `lmcache_blend_lookup_no_gpu_context_errors_total` | Counter | `CB_LOOKUP_END` | +1 when `no_gpu_context=True` |

**What it answers:** How often does the CB server receive lookup requests? What fraction hit the fingerprint table? What fraction are confirmed in storage? How many stale evictions occur?

### CB Retrieve Metrics

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_blend.retrieve_requests` | `lmcache_blend_retrieve_requests_total` | Counter | `CB_RETRIEVE_START` | +1 per event |
| `lmcache_blend.retrieve_chunks` | `lmcache_blend_retrieve_chunks_total` | Counter | `CB_RETRIEVE_START` | `+num_chunks` |
| `lmcache_blend.retrieve_failures` | `lmcache_blend_retrieve_failures_total` | Counter | `CB_RETRIEVE_END` | +1 when `success=False` |

**What it answers:** How often is CB retrieval invoked? How many chunks are retrieved per call? What is the failure rate?

### CB Store Pre-computed Metrics

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_blend.store_pre_computed_requests` | `lmcache_blend_store_pre_computed_requests_total` | Counter | `CB_STORE_PRE_COMPUTED_START` | +1 per event |
| `lmcache_blend.store_pre_computed_chunks` | `lmcache_blend_store_pre_computed_chunks_total` | Counter | `CB_STORE_PRE_COMPUTED_END` | `+stored_chunks` |
| `lmcache_blend.store_pre_computed_failures` | `lmcache_blend_store_pre_computed_failures_total` | Counter | `CB_STORE_PRE_COMPUTED_END` | +1 when `success=False` |

**What it answers:** How often is pre-computed CB storage invoked? How many chunks are written? What is the failure rate?

### CB Store Final Metrics

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_blend.store_final_requests` | `lmcache_blend_store_final_requests_total` | Counter | `CB_STORE_FINAL_START` | +1 per event |
| `lmcache_blend.store_final_chunks` | `lmcache_blend_store_final_chunks_total` | Counter | `CB_STORE_FINAL_END` | `+stored_chunks` |
| `lmcache_blend.store_final_failures` | `lmcache_blend_store_final_failures_total` | Counter | `CB_STORE_FINAL_END` | +1 when `success=False` |

**What it answers:** How often is final CB storage invoked? How many chunks are committed? What is the failure rate?

### CB Fingerprint Table Metrics

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_blend.fingerprints_registered` | `lmcache_blend_fingerprints_registered_total` | Counter | `CB_FINGERPRINTS_REGISTERED` | `+num_chunks` |
| `lmcache_blend.chunks_evicted` | `lmcache_blend_chunks_evicted_total` | Counter | `CB_CHUNKS_EVICTED` | `+num_chunks` |

**What it answers:** How many chunks are indexed into the fingerprint table? How many stale entries are evicted?

---

For event metadata contracts (what keys each `EventType` carries), see [EVENTS.md](EVENTS.md).
