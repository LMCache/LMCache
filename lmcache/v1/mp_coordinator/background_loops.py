# SPDX-License-Identifier: Apache-2.0
"""The coordinator's periodic work, and how alive it is.

A background loop that raises would otherwise end its task for good, and
nothing would notice until shutdown. :class:`BackgroundLoops` runs each
loop so that a failed pass is logged and counted and the next pass still
runs, and keeps the per-loop stats the observability module exports.
:class:`EventLoopMonitor` measures how long the event loop every handler
and loop shares was blocked.

See ``docs/design/v1/mp_coordinator/observability.md``.
"""

# Standard
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
import asyncio
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.metric_sources import register_metric_source

logger = init_logger(__name__)


@dataclass
class BackgroundLoopStats:
    """What one background loop has done since this process started.

    Attributes:
        interval_seconds: Seconds between passes.
        last_success_timestamp: Wall-clock time the last pass finished
            without raising; ``None`` before the first.
        passes_ok: Passes that finished without raising.
        passes_failed: Passes that raised; the loop kept running.
    """

    interval_seconds: float
    last_success_timestamp: float | None = None
    passes_ok: int = 0
    passes_failed: int = 0


class BackgroundLoops:
    """Runs the coordinator's periodic work and keeps each loop's stats.

    Each loop sleeps its interval, then runs one pass. A pass that raises
    is logged and counted, and the loop goes on; only cancellation stops
    it. Used by the app for its own loops and handed to controllers through
    ``ControllerRuntime``.
    """

    def __init__(self) -> None:
        self._stats: dict[str, BackgroundLoopStats] = {}
        register_metric_source(self)

    def start(
        self,
        name: str,
        interval_seconds: float,
        run_pass: Callable[[], Awaitable[object]],
    ) -> "asyncio.Task[None]":
        """Start a loop running ``run_pass`` every ``interval_seconds``.

        Call from the running event loop. Cancel the returned task to stop
        the loop.

        Args:
            name: The loop's name, as it appears in metrics
                (``health``, ``checkpoint``, ``eviction``, ...). Unique.
            interval_seconds: Seconds to sleep before each pass.
            run_pass: One pass of the loop's work.

        Returns:
            The task running the loop.

        Raises:
            ValueError: If a loop named ``name`` was already started.
        """
        if name in self._stats:
            raise ValueError(f"background loop {name!r} already started")
        stats = BackgroundLoopStats(interval_seconds=interval_seconds)
        self._stats[name] = stats
        return asyncio.create_task(
            self._run(name, stats, run_pass), name=f"coordinator-loop-{name}"
        )

    def stats(self) -> dict[str, BackgroundLoopStats]:
        """Return a copy of every started loop's stats, keyed by name."""
        return {name: replace(stats) for name, stats in self._stats.items()}

    async def _run(
        self,
        name: str,
        stats: BackgroundLoopStats,
        run_pass: Callable[[], Awaitable[object]],
    ) -> None:
        """Sleep, run one pass, count it; until cancelled."""
        while True:
            await asyncio.sleep(stats.interval_seconds)
            try:
                await run_pass()
            except Exception:
                stats.passes_failed += 1
                logger.exception("Background loop %s: pass failed", name)
                continue
            stats.passes_ok += 1
            stats.last_success_timestamp = time.time()


@dataclass(frozen=True)
class EventLoopStats:
    """How long the event loop was blocked.

    Attributes:
        blocked_seconds: Total seconds the event loop was blocked since
            monitoring began.
    """

    blocked_seconds: float


class EventLoopMonitor:
    """Measures how long the event loop was blocked.

    Sleeps half a second at a time and adds up how late each wake-up was.
    A late wake-up means something held the loop: a request handler doing
    synchronous work, or ingest waiting on a checkpoint. Every handler,
    heartbeat and background loop waited that long too.
    """

    def __init__(self) -> None:
        self._blocked_seconds = 0.0
        register_metric_source(self)

    def stats(self) -> EventLoopStats:
        """Return how long the event loop was blocked so far."""
        return EventLoopStats(blocked_seconds=self._blocked_seconds)

    async def run(self) -> None:
        """Measure until cancelled. Run as a task on the monitored loop."""
        while True:
            started = time.monotonic()
            await asyncio.sleep(0.5)
            late = time.monotonic() - started - 0.5
            if late > 0:
                self._blocked_seconds += late
