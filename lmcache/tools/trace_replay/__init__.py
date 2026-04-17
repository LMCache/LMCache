# SPDX-License-Identifier: Apache-2.0

"""Trace replay subsystem.

This package implements the **replay** half of the ``lmcache trace``
feature.  A trace file is produced by the recorder in
:mod:`lmcache.v1.mp_observability.trace`; the driver here reads that
file and reissues each recorded call against a fresh
:class:`~lmcache.v1.distributed.storage_manager.StorageManager`
instance.

Public API:

* :class:`~lmcache.tools.trace_replay.driver.StorageReplayDriver` —
  high-level entry point; constructs a StorageManager from a
  :class:`~lmcache.v1.distributed.config.StorageManagerConfig` and
  drives the trace.
* :class:`~lmcache.tools.trace_replay.dispatch.CallDispatcher` —
  registry mapping recorded qualnames to live callables.  Useful
  when a caller needs to extend the replay with custom ops.
* :class:`~lmcache.tools.trace_replay.stats.ReplayStatsCollector` —
  per-qualname latency aggregator used by
  ``lmcache bench trace-replay``.
"""

# First Party
from lmcache.tools.trace_replay.dispatch import (
    CallDispatcher,
    ReplayContext,
    build_default_dispatcher,
)
from lmcache.tools.trace_replay.driver import (
    ReplayPace,
    ReplayResult,
    StorageReplayDriver,
)
from lmcache.tools.trace_replay.stats import (
    OpStats,
    ReplayStatsCollector,
)

__all__ = [
    "CallDispatcher",
    "OpStats",
    "ReplayContext",
    "ReplayPace",
    "ReplayResult",
    "ReplayStatsCollector",
    "StorageReplayDriver",
    "build_default_dispatcher",
]
