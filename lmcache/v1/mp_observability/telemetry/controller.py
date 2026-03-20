# SPDX-License-Identifier: Apache-2.0

"""Telemetry controller: queue, drain thread, singleton."""

# Future
from __future__ import annotations

# Standard
import collections
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.telemetry.config import TelemetryConfig
from lmcache.v1.mp_observability.telemetry.event import EventType, TelemetryEvent
from lmcache.v1.mp_observability.telemetry.processors.base import (
    TelemetryProcessor,
)
from lmcache.v1.mp_observability.telemetry.processors.logging_processor import (
    LoggingProcessor,
    LoggingProcessorConfig,
)

logger = init_logger(__name__)


class TelemetryController:
    """Manages telemetry event ingestion, queueing, and dispatch to processors.

    Events are appended to a lock-free deque on the hot path and drained by a
    background thread that dispatches to registered processors.
    """

    def __init__(self, config: TelemetryConfig):
        self._config = config
        self._queue: collections.deque[TelemetryEvent] = collections.deque()
        self._wake = threading.Event()
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()
        self._processors: list[TelemetryProcessor] = []
        self._thread: threading.Thread | None = None
        self._discard_count: int = 0
        self._last_discard_warning: float = 0.0

    def is_enabled(self) -> bool:
        """Return True if telemetry is enabled (i.e. controller is active)."""
        return self._config.enabled

    def log(self, event: TelemetryEvent) -> None:
        """Enqueue a telemetry event (non-blocking hot path).

        When the queue is full, the event is silently discarded with a
        rate-limited WARNING log (at most once per second).
        """
        if not self._config.enabled:
            return

        if len(self._queue) >= self._config.max_queue_size:
            self._discard_count += 1
            now = time.monotonic()
            if now - self._last_discard_warning >= 1.0:
                logger.warning(
                    "Telemetry queue full (max_queue_size=%d), "
                    "%d event(s) discarded so far",
                    self._config.max_queue_size,
                    self._discard_count,
                )
                self._last_discard_warning = now
            return

        event.timestamp = time.time()
        self._queue.append(event)
        self._wake.set()

    def register_processor(self, processor: TelemetryProcessor) -> None:
        """Register a processor for event dispatch (thread-safe)."""
        with self._lock:
            self._processors.append(processor)

    def start(self) -> None:
        """Start the drain thread. No-op when disabled or already started."""
        if not self._config.enabled:
            return

        if self._thread is not None and self._thread.is_alive():
            return

        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="TelemetryController",
        )
        logger.debug("Starting TelemetryController...")
        self._thread.start()

    def stop(self) -> None:
        """Stop the drain thread. Safe to call when not started."""
        self._stop_flag.set()
        self._wake.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()

        # Final drain
        self._drain_all()

        # Shutdown processors
        with self._lock:
            snapshot = list(self._processors)
        for proc in snapshot:
            try:
                proc.shutdown()
            except Exception:
                logger.exception(
                    "TelemetryController: error shutting down %s",
                    type(proc).__name__,
                )

    def _run(self) -> None:
        """Drain loop: wait for wake signal or timeout, then drain."""
        while not self._stop_flag.is_set():
            self._wake.wait(timeout=0.1)
            self._wake.clear()
            self._drain_all()

    def _drain_all(self) -> None:
        """Pop all queued events and dispatch to processors."""
        with self._lock:
            snapshot = list(self._processors)

        while True:
            try:
                event = self._queue.popleft()
            except IndexError:
                break
            for proc in snapshot:
                try:
                    proc.on_new_event(event)
                except Exception:
                    logger.exception(
                        "TelemetryController: error in processor %s",
                        type(proc).__name__,
                    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_processors(config: TelemetryConfig) -> list[TelemetryProcessor]:
    """Create processor instances from a TelemetryConfig.

    Uses ``isinstance()`` dispatch on processor config types.

    Args:
        config: The telemetry config containing processor configs.

    Returns:
        List of instantiated processors.

    Raises:
        ValueError: If a processor config type is unrecognized.
    """
    processors: list[TelemetryProcessor] = []
    for proc_config in config.processor_configs:
        if isinstance(proc_config, LoggingProcessorConfig):
            processors.append(LoggingProcessor(proc_config))
        else:
            raise ValueError(
                f"Unknown telemetry processor config type: "
                f"{type(proc_config).__name__}. "
                f"Add a branch in create_processors() for this config type."
            )
    return processors


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_global_controller = TelemetryController(TelemetryConfig())


def get_telemetry_controller() -> TelemetryController:
    """Return the current global TelemetryController singleton."""
    return _global_controller


def init_telemetry_controller(config: TelemetryConfig) -> TelemetryController:
    """Replace the global singleton with a new controller built from *config*.

    Creates processors from ``config.processor_configs`` and registers them.
    """
    global _global_controller
    _global_controller = TelemetryController(config)
    for proc in create_processors(config):
        _global_controller.register_processor(proc)
    return _global_controller


def log_telemetry(event: TelemetryEvent) -> None:
    """Convenience: log an event to the global controller."""
    _global_controller.log(event)


def make_start_event(
    name: str, session_id: str, **metadata: str | int | float | bool
) -> TelemetryEvent:
    """Create a START telemetry event."""
    return TelemetryEvent(
        name=name,
        event_type=EventType.START,
        session_id=session_id,
        metadata=metadata,
    )


def make_end_event(
    name: str, session_id: str, **metadata: str | int | float | bool
) -> TelemetryEvent:
    """Create an END telemetry event."""
    return TelemetryEvent(
        name=name,
        event_type=EventType.END,
        session_id=session_id,
        metadata=metadata,
    )
