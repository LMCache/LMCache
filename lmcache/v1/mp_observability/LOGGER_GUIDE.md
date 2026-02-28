# How to Add a New Stats Logger

This guide walks through adding a new logger — for example, an integrator-level
logger that tracks requests handled by the MP server's RPC layer.

For the full list of existing metrics, see [METRICS.md](METRICS.md).

---

## Step 1 — Define a stats dataclass

Create `lmcache/v1/mp_observability/stats/my_stats.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import List

@dataclass
class MyStats:
    interval_rpc_requests: int = 0
    rpc_latency: List[float] = field(default_factory=list)
```

---

## Step 2 — Implement the listener + PrometheusLogger

Create `lmcache/v1/mp_observability/logger/my_logger.py`:

```python
# SPDX-License-Identifier: Apache-2.0
import threading
import time
from collections import deque
from typing import Deque

from lmcache.v1.mp_observability.logger.prometheus_logger import PrometheusLogger
from lmcache.v1.mp_observability.stats.my_stats import MyStats

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

        self._rpc_requests_counter = self.create_counter(
            "lmcache_mp:rpc_requests",
            "Total number of RPC requests handled",
            labelnames,
        )
        self._rpc_latency_histogram = self.create_histogram(
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

        self.log_counter(self._rpc_requests_counter, stats.interval_rpc_requests)
        self.log_histogram(self._rpc_latency_histogram, stats.rpc_latency)
```

---

## Step 3 — Register with PrometheusController

In `lmcache/v1/mp_observability/prometheus_controller.py`, add your logger
in `__init__` alongside the existing loggers:

```python
from lmcache.v1.mp_observability.logger.my_logger import MyListener

class PrometheusController(StorageControllerInterface):
    def __init__(self, storage_manager, l1_manager, log_interval):
        super().__init__(storage_manager, l1_manager)
        self._log_interval = log_interval
        self.all_loggers: List[PrometheusLogger] = []

        # Existing loggers
        self.sm_stats_logger = StorageManagerStatsLogger()
        self.get_storage_manager().register_listener(self.sm_stats_logger)
        self.all_loggers.append(self.sm_stats_logger)

        self.l1_stats_logger = L1ManagerStatsLogger()
        self.get_l1_manager().register_listener(self.l1_stats_logger)
        self.all_loggers.append(self.l1_stats_logger)

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

---

## Design rules to follow

| Rule | Reason |
|---|---|
| Use a module-level `threading.Lock` + `@stats_safe` decorator | Callbacks fire from the L1Manager thread; `log_prometheus()` fires from the PrometheusController thread — they can race. |
| Swap `self.stats` atomically inside `log_prometheus()` (hold the lock only for the swap, log outside) | Keeps the critical section minimal so callbacks are not blocked during Prometheus I/O. |
| Prefix metrics with `lmcache_mp:` | Keeps the MP namespace separate from `lmcache:` (the main engine namespace). |
| Use interval counters (reset each flush) not running totals | Prometheus Counters are cumulative by design; `log_counter` calls `.inc(delta)` — the delta is the interval count. |
