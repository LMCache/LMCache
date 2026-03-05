# Distributed Storage Manager Observability Metrics

## Overview

The observability model is listener-based with **separate per-tier loggers**:

- **`StorageManagerStatsLogger`** (`StorageManagerListener`) — events from the top-level `StorageManager` API (`submit_prefetch_task`, `reserve_write`, etc.)
- **`L1ManagerStatsLogger`** (`L1ManagerListener`) — events from the L1 in-memory cache tier (key reads, writes, evictions)

Each logger owns its own stats dataclass (`StorageManagerStats`, `L1Stats`) and independently
implements the `PrometheusLogger` interface.

`PrometheusController` runs a background daemon thread that calls `log_prometheus()` on every
registered logger at a configurable interval. `log_prometheus()` atomically snapshots `self.stats`,
resets it to a fresh stats instance, and pushes all accumulated values to Prometheus.

All metrics use the `lmcache_mp:` prefix (mp = multiprocess), distinct from the main engine's
`lmcache:` namespace.

For implementation guidance on adding new loggers, see [LOGGER_GUIDE.md](LOGGER_GUIDE.md).

---

## StorageManager Read Metrics

| Python field | Prometheus name | Type | Source callback | Calculation |
|---|---|---|---|---|
| `interval_sm_read_requests` | `lmcache_mp:sm_read_requests` | Counter | `on_sm_read_prefetched` | +1 per call |
| `interval_sm_read_succeed_keys` | `lmcache_mp:sm_read_succeed_keys` | Counter | `on_sm_read_prefetched` | `+len(succeeded_keys)` per call |
| `interval_sm_read_failed_keys` | `lmcache_mp:sm_read_failed_keys` | Counter | `on_sm_read_prefetched` | `+len(failed_keys)` per call |

**What it answers:** How often does the StorageManager receive read requests? What is the L1 hit rate?

> **Note:** StorageManager-level read latency is not tracked. The `on_sm_read_prefetched` callback fires
> during the `lookup()` RPC, while `on_sm_read_prefetched_finished` fires during the separate
> `retrieve()` RPC. With concurrent workers these can complete out of order, making a simple
> FIFO latency tracker unreliable.

---

## StorageManager Write Metrics

| Python field | Prometheus name | Type | Source callback | Calculation |
|---|---|---|---|---|
| `interval_sm_write_requests` | `lmcache_mp:sm_write_requests` | Counter | `on_sm_reserved_write` | +1 per call |
| `interval_sm_write_succeed_keys` | `lmcache_mp:sm_write_succeed_keys` | Counter | `on_sm_reserved_write` | `+len(succeeded_keys)` per call |
| `interval_sm_write_failed_keys` | `lmcache_mp:sm_write_failed_keys` | Counter | `on_sm_reserved_write` | `+len(failed_keys)` per call |

**What it answers:** How often are writes attempted? What fraction fail due to OOM or write conflicts?

---

## L1 Read Metrics

| Python field | Prometheus name | Type | Source callback | Calculation |
|---|---|---|---|---|
| `interval_l1_read_keys` | `lmcache_mp:l1_read_keys` | Counter | `on_l1_keys_read_finished` | `+len(keys)` per call |

**What it answers:** How many keys are being read from L1?

> **Note:** `on_l1_keys_reserved_read` is a no-op — key counts are recorded only when the read
> actually completes via `on_l1_keys_read_finished`, giving an accurate count of successfully
> served reads.

---

## L1 Write Metrics

| Python field | Prometheus name | Type | Source callback | Calculation |
|---|---|---|---|---|
| `interval_l1_write_keys` | `lmcache_mp:l1_write_keys` | Counter | `on_l1_keys_write_finished` | `+len(keys)` per call |

**What it answers:** How many keys are being written to L1?

> **Note:** `on_l1_keys_reserved_write` is a no-op — key counts are recorded only when the write
> actually completes via `on_l1_keys_write_finished`.

---

## L1 Eviction Metrics

| Python field | Prometheus name | Type | Source callback | Calculation |
|---|---|---|---|---|
| `interval_l1_evicted_keys` | `lmcache_mp:l1_evicted_keys` | Counter | `on_l1_keys_deleted_by_manager` | `+len(keys)` per call |

**What it answers:** How aggressively is the eviction controller clearing L1? A high eviction rate relative to writes signals memory pressure.

---

## L2 Placeholder

`L2ManagerListener.on_l2_lookup_and_lock()` is currently a no-op. L2 metrics will be
added to a new stats dataclass and logger once the L2 manager interface is finalized.
