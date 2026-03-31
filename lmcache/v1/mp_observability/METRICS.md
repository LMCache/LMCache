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

For event metadata contracts (what keys each `EventType` carries), see [EVENTS.md](EVENTS.md).
