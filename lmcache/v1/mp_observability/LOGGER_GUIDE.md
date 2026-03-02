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

## Step 3 — Self-register with the global PrometheusController

Each module creates its own logger and registers it with the global singleton.
No changes to `PrometheusController` are needed.

In your module's `__init__` (or wherever setup happens):

```python
from lmcache.v1.mp_observability.logger.my_logger import MyListener
from lmcache.v1.mp_observability.prometheus_controller import get_prometheus_controller

class MyModule:
    def __init__(self):
        # ... existing setup ...

        # Self-register observability logger
        my_logger = MyListener()
        # If your logger also implements a listener interface, register it:
        # self.register_listener(my_logger)
        get_prometheus_controller().register_logger(my_logger)
```

`PrometheusController._run()` iterates `all_loggers` and calls `log_prometheus()`
on each one at every interval. Calling `register_logger()` is the only step required —
no modifications to the controller are needed. The controller takes a thread-safe
snapshot of loggers before each flush cycle, so late registration is safe. Exceptions
from individual loggers are caught and logged, so a broken logger cannot crash the loop.

---

## Design rules to follow

| Rule | Reason |
|---|---|
| Use a module-level `threading.Lock` + `@stats_safe` decorator | Callbacks fire from the L1Manager thread; `log_prometheus()` fires from the PrometheusController thread — they can race. |
| Swap `self.stats` atomically inside `log_prometheus()` (hold the lock only for the swap, log outside) | Keeps the critical section minimal so callbacks are not blocked during Prometheus I/O. |
| Prefix metrics with `lmcache_mp:` | Keeps the MP namespace separate from `lmcache:` (the main engine namespace). |
| Use interval counters (reset each flush) not running totals | Prometheus Counters are cumulative by design; `log_counter` calls `.inc(delta)` — the delta is the interval count. |
