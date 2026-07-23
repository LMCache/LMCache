# SPDX-License-Identifier: Apache-2.0
"""Flush-thread scaffolding shared by the interval usage reporters."""

# Future
from __future__ import annotations

# Standard
from typing import Callable
import os
import threading

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class UsageFlushThread:
    """Daemon thread that runs a flush callback once per report interval.

    Calls *flush* every ``LMCACHE_USAGE_TRACK_INTERVAL`` seconds
    (default 600, clamped to >= 1 s; a malformed value falls back to
    the default instead of disabling telemetry). The thread starts
    immediately on construction.
    """

    def __init__(self, flush: Callable[[], None]) -> None:
        """Start the flush thread.

        Args:
            flush: Called on the flush thread once per interval. Must
                not raise — an exception ends the thread (the reporters'
                flush methods are guarded and never raise).
        """
        self._flush = flush
        # Clamp to >= 1 s: Event.wait(0) would turn the flush loop into
        # a busy spin.
        try:
            flush_interval = float(os.getenv("LMCACHE_USAGE_TRACK_INTERVAL", "600"))
        except ValueError:
            logger.debug("Invalid LMCACHE_USAGE_TRACK_INTERVAL", exc_info=True)
            flush_interval = 600.0
        self._flush_interval: float = max(flush_interval, 1.0)
        self._stop_event = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="lmcache-usage-report"
        )
        self._thread.start()

    def wake(self) -> None:
        """Trigger one flush now instead of at the next interval tick."""
        self._wake.set()

    def stop(self) -> None:
        """Stop the thread; it runs no further flushes. Idempotent.

        Does not flush — callers send their own final flush on shutdown.
        """
        self._stop_event.set()
        self._wake.set()

    def _run(self) -> None:
        while True:
            self._wake.wait(timeout=self._flush_interval)
            self._wake.clear()
            if self._stop_event.is_set():
                return
            self._flush()
