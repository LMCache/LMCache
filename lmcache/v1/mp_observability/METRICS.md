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

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_mp.l0_block_lifetime_seconds` | `lmcache_mp_l0_block_lifetime_seconds` | Histogram | `MP_VLLM_BLOCK_ALLOCATION` (eviction detected) | `eviction_time - alloc_time` per sampled block |
| `lmcache_mp.l0_block_idle_before_evict_seconds` | `lmcache_mp_l0_block_idle_before_evict_seconds` | Histogram | `MP_VLLM_BLOCK_ALLOCATION` (eviction detected) | `eviction_time - last_access_time` per sampled block |
| `lmcache_mp.l0_block_reuse_gap_seconds` | `lmcache_mp_l0_block_reuse_gap_seconds` | Histogram | `MP_VLLM_BLOCK_ALLOCATION` (cache hit) | Time gaps between consecutive accesses from access history |

**What it answers:** How long do GPU blocks live before eviction? How idle are they? How frequently are cached blocks reused?

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

For event metadata contracts (what keys each `EventType` carries), see [EVENTS.md](EVENTS.md).
