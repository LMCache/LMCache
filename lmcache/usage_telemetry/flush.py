# SPDX-License-Identifier: Apache-2.0
"""Flush-thread scaffolding shared by the interval usage reporters.

The flush threads are :mod:`lmcache.v1.periodic_thread` threads,
registered in the global ``PeriodicThreadRegistry``.
"""

# Future
from __future__ import annotations

# Standard
from typing import Callable
import os

# First Party
from lmcache.logging import init_logger
from lmcache.v1.periodic_thread import (
    PeriodicThread,
    ThreadLevel,
    ThreadRunSummary,
    create_periodic_thread,
)

logger = init_logger(__name__)


def usage_flush_interval_seconds() -> float:
    """Seconds between telemetry flushes, from ``LMCACHE_USAGE_TRACK_INTERVAL``.

    Returns:
        The configured interval in seconds (default 600), clamped to
        >= 1; a malformed value falls back to the default.
    """
    try:
        flush_interval = float(os.getenv("LMCACHE_USAGE_TRACK_INTERVAL", "600"))
    except ValueError:
        logger.debug("Invalid LMCACHE_USAGE_TRACK_INTERVAL", exc_info=True)
        flush_interval = 600.0
    return max(flush_interval, 1.0)


def start_usage_flush_thread(name: str, flush: Callable[[], None]) -> PeriodicThread:
    """Start a ``LOW``-level :class:`PeriodicThread` that runs *flush*.

    The thread first runs after one full flush interval; ``wake()``
    requests during that initial wait take effect at the first run.
    The thread is registered in the global ``PeriodicThreadRegistry``,
    so *name* must be unique per process.

    Args:
        name: Thread name, also the registry key.
        flush: Called on the flush thread every
            ``LMCACHE_USAGE_TRACK_INTERVAL`` seconds. Any exception is
            caught by ``PeriodicThread`` and logged; the thread keeps
            running.

    Returns:
        The started thread; stop it via ``stop()`` on shutdown.
    """
    interval = usage_flush_interval_seconds()

    def _execute() -> ThreadRunSummary:
        flush()
        return ThreadRunSummary(success=True)

    thread = create_periodic_thread(
        name=name,
        interval=interval,
        execute_fn=_execute,
        level=ThreadLevel.LOW,
        init_wait=interval,
    )
    thread.start()
    return thread
