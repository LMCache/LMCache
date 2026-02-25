# Distributed Storage Manager Observability Metrics

## Overview

The observability model is listener-based. `StorageStatsListener` implements three listener interfaces simultaneously:

- **`StorageManagerListener`** — events from the top-level `StorageManager` API (`submit_prefetch_task`, `reserve_write`, etc.)
- **`L1ManagerListener`** — events from the L1 in-memory cache tier (key reservations, completions, evictions)
- **`L2ManagerListener`** — placeholder for L2 storage tier events (not yet populated)

`PrometheusController` runs a background daemon thread that calls `log_prometheus()` on every registered logger at a configurable interval. `log_prometheus()` atomically snapshots `self.stats`, resets it to a fresh `StorageManagerStats()`, and pushes all accumulated values to Prometheus.

---

## Metric Groups

### SM Read Metrics

| Python field | Prometheus name | Type | Source callback | Calculation |
|---|---|---|---|---|
| `interval_sm_read_requests` | `lmcache_mp:sm_read_requests` | Counter | `on_sm_read_prefetched` | +1 per call |
| `interval_sm_read_hit_keys` | `lmcache_mp:sm_read_hit_keys` | Counter | `on_sm_read_prefetched` | `+len(succeeded_keys)` per call |
| `interval_sm_read_miss_keys` | `lmcache_mp:sm_read_miss_keys` | Counter | `on_sm_read_prefetched` | `+len(failed_keys)` per call |

**What it answers:** How often does the SM receive read requests? What is the L1 hit rate?

> **Note:** SM-level read latency is not tracked. The `on_sm_read_prefetched` callback fires
> during the `lookup()` RPC, while `on_sm_read_prefetched_finished` fires during the separate
> `retrieve()` RPC. With concurrent workers these can complete out of order, making a simple
> FIFO latency tracker unreliable. L1-level latency (below) is safe because the L1Manager lock
> serializes all its callbacks.

---

### SM Write Metrics

| Python field | Prometheus name | Type | Source callback | Calculation |
|---|---|---|---|---|
| `interval_sm_write_requests` | `lmcache_mp:sm_write_requests` | Counter | `on_sm_reserved_write` | +1 per call |
| `interval_sm_write_success_keys` | `lmcache_mp:sm_write_success_keys` | Counter | `on_sm_reserved_write` | `+len(succeeded_keys)` per call |
| `interval_sm_write_failed_keys` | `lmcache_mp:sm_write_failed_keys` | Counter | `on_sm_reserved_write` | `+len(failed_keys)` per call |

**What it answers:** How often are writes attempted? What fraction fail due to OOM or write conflicts?

---

### L1 Read Metrics

| Python field | Prometheus name | Type | Source callback | Calculation |
|---|---|---|---|---|
| `interval_l1_read_keys` | `lmcache_mp:l1_read_keys` | Counter | `on_l1_keys_reserved_read` | `+len(keys)` per call |
| `l1_read_latency` | `lmcache_mp:l1_read_latency` | Histogram (seconds) | `on_l1_keys_reserved_read` + `on_l1_keys_read_finished` | `finish_time - start_time` per batch (FIFO) |

**What it answers:** How many keys are being read from L1? How long does an L1 read batch take?

---

### L1 Write Metrics

| Python field | Prometheus name | Type | Source callback | Calculation |
|---|---|---|---|---|
| `interval_l1_write_keys` | `lmcache_mp:l1_write_keys` | Counter | `on_l1_keys_reserved_write` | `+len(keys)` per call |
| `l1_write_latency` | `lmcache_mp:l1_write_latency` | Histogram (seconds) | `on_l1_keys_reserved_write` + `on_l1_keys_write_finished` | `finish_time - start_time` per batch (FIFO) |

**What it answers:** How many keys are being written to L1? How long does an L1 write batch take?

---

### L1 Eviction Metrics

| Python field | Prometheus name | Type | Source callback | Calculation |
|---|---|---|---|---|
| `interval_l1_evicted_keys` | `lmcache_mp:l1_evicted_keys` | Counter | `on_l1_keys_deleted_by_manager` | `+len(keys)` per call |

**What it answers:** How aggressively is the eviction controller clearing L1? A high eviction rate relative to writes signals memory pressure.

---

## Latency Calculation

L1 latency is tracked **per batch** using FIFO queues in `StorageStatsListener`:

```python
_l1_read_start_times:  deque[float]   # batch start timestamps for L1 reads
_l1_write_start_times: deque[float]   # batch start timestamps for L1 writes
```

This is safe because `L1Manager` holds its internal lock for the duration of each
operation **including** the full listener callback loop. Callbacks for the same
direction (read or write) are therefore strictly serialized, so batches always
complete in the same order they started — FIFO is guaranteed.

**Pattern for each direction:**

1. **Start callback** (e.g. `on_l1_keys_reserved_read`):
   ```python
   self._l1_read_start_times.append(time.perf_counter())
   ```

2. **Finish callback** (e.g. `on_l1_keys_read_finished`):
   ```python
   if self._l1_read_start_times:
       self.stats.l1_read_latency.append(
           time.perf_counter() - self._l1_read_start_times.popleft()
       )
   ```

One latency sample is recorded per `reserved → finished` pair, regardless of how many
keys are in the batch. If `read_finished` is called without a matching `reserved_read`
(e.g. during shutdown), the empty-deque guard makes it a safe no-op.

---

## Histogram Buckets

All latency histograms use the following default bucket set (in seconds), covering
sub-millisecond to 10-second latencies:

```
[0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
```

These can be overridden per-metric via the `PrometheusLogger` `extra_config` mechanism
using a key of the form `histogram_bucket_<short_name>`, where `<short_name>` is the
part of the Prometheus metric name after the first `:`.

For example, to tighten the buckets for `lmcache_mp:l1_read_latency`:
```
histogram_bucket_l1_read_latency = [0.001, 0.005, 0.01, 0.05, 0.1]
```

---

## Namespace

All distributed storage manager metrics use the `lmcache_mp:` prefix (mp = multiprocess),
distinct from the main engine's `lmcache:` namespace.

---

## L2 Placeholder

`L2ManagerListener.on_l2_lookup_and_lock()` is currently a no-op. L2 metrics will be
added to `StorageManagerStats` and a new (or extended) listener once the L2 manager
interface is finalized.

---

## How to Add a New Stats Listener

This section walks through adding a second logger — for example, an integrator-level
logger that tracks requests handled by the MP server's RPC layer.

### Step 1 — Define a stats dataclass

Create `lmcache/v1/distributed/observability/stats/my_stats.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import List

@dataclass
class MyStats:
    interval_rpc_requests: int = 0
    rpc_latency: List[float] = field(default_factory=list)
```

### Step 2 — Implement the listener + PrometheusLogger

Create `lmcache/v1/distributed/observability/logger/my_logger.py`:

```python
# SPDX-License-Identifier: Apache-2.0
import threading
import time
from collections import deque
from typing import Deque

from lmcache.v1.distributed.observability.logger.prometheus_logger import PrometheusLogger
from lmcache.v1.distributed.observability.stats.my_stats import MyStats

# Reuse the same latency bucket definition
_LATENCY_BUCKETS = [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1,
                    0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]

_stats_lock = threading.Lock()

def stats_safe(func):
    def wrapper(self, *args, **kwargs):
        with _stats_lock:
            return func(self, *args, **kwargs)
    return wrapper


class MyListener(PrometheusLogger):
    def __init__(self, labels=None, config=None):
        if labels is None:
            labels = {}
        PrometheusLogger.__init__(self, labels=labels, config=config)

        self.stats = MyStats()
        labelnames = list(labels.keys())

        self._rpc_requests_counter = self._create_counter(
            "lmcache_mp:rpc_requests",
            "Total number of RPC requests handled",
            labelnames,
        )
        self._rpc_latency_histogram = self._create_histogram(
            "lmcache_mp:rpc_latency",
            "RPC handler latency in seconds",
            labelnames,
            buckets=_LATENCY_BUCKETS,
        )
        self._rpc_start_times: Deque[float] = deque()

    # --- Call these from your RPC handler / event source ---

    @stats_safe
    def on_rpc_started(self):
        self._rpc_start_times.append(time.perf_counter())
        self.stats.interval_rpc_requests += 1

    @stats_safe
    def on_rpc_finished(self):
        if self._rpc_start_times:
            self.stats.rpc_latency.append(
                time.perf_counter() - self._rpc_start_times.popleft()
            )

    # --- PrometheusLogger protocol ---

    def log_prometheus(self) -> None:
        with _stats_lock:
            stats = self.stats
            self.stats = MyStats()

        self._log_counter(self._rpc_requests_counter, stats.interval_rpc_requests)
        self._log_histogram(self._rpc_latency_histogram, stats.rpc_latency)
```

### Step 3 — Register with PrometheusController

In `lmcache/v1/distributed/observability/prometheus_controller.py`, add your logger
in `__init__` alongside the existing `StorageStatsListener`:

```python
from lmcache.v1.distributed.observability.logger.my_logger import MyListener

class PrometheusController(StorageControllerInterface):
    def __init__(self, storage_manager, l1_manager, log_interval):
        super().__init__(storage_manager, l1_manager)
        self._log_interval = log_interval
        self.all_loggers: List[PrometheusLogger] = []

        # Existing logger
        self.sm_stats_logger = StorageStatsListener()
        self.get_l1_manager().register_listener(self.sm_stats_logger)
        self.get_storage_manager().register_listener(self.sm_stats_logger)
        self.all_loggers.append(self.sm_stats_logger)

        # New logger — register with whatever event source it needs
        self.my_logger = MyListener()
        # e.g. self.get_storage_manager().register_listener(self.my_logger)
        self.all_loggers.append(self.my_logger)   # <-- this is all that's needed
                                                  #     for periodic flushing

        # ... thread setup unchanged ...
```

`PrometheusController._run()` iterates `self.all_loggers` and calls `log_prometheus()`
on each one at every interval. Adding to `all_loggers` is the only change required —
no modifications to `_run()` are needed. Exceptions from individual loggers are caught
and logged, so a broken logger cannot crash the loop.

### Design rules to follow

| Rule | Reason |
|---|---|
| Use a module-level `threading.Lock` + `@stats_safe` decorator | Callbacks fire from the L1Manager thread; `log_prometheus()` fires from the PrometheusController thread — they can race. |
| Swap `self.stats` atomically inside `log_prometheus()` (hold the lock only for the swap, log outside) | Keeps the critical section minimal so callbacks are not blocked during Prometheus I/O. |
| Use FIFO `deque[float]` for latency, not `dict[key, float]` | O(1) per call; safe when the event source serializes callbacks (e.g. under `@l1_mgr_synchronized`). Use a request-ID scheme if callbacks can arrive out of order. |
| Prefix metrics with `lmcache_mp:` | Keeps the MP namespace separate from `lmcache:` (the main engine namespace). |
| Use interval counters (reset each flush) not running totals | Prometheus Counters are cumulative by design; `_log_counter` calls `.inc(delta)` — the delta is the interval count. |
