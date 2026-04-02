# SPDX-License-Identifier: Apache-2.0

"""EventBus: unified pub/sub dispatcher for MP observability events."""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable
import collections
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

EventCallback = Callable[[Event], None]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class EventBusConfig:
    """Configuration for the EventBus.

    Attributes:
        enabled: Whether the event bus is active.  When disabled,
            ``publish()`` is a no-op and the drain thread is not started.
        max_queue_size: Maximum number of events in the queue.  When the
            queue is full, new events are silently dropped with a
            rate-limited warning.
    """

    enabled: bool = True
    max_queue_size: int = 10_000


# ---------------------------------------------------------------------------
# Subscriber ABC
# ---------------------------------------------------------------------------


class EventSubscriber(ABC):
    """Base class for per-component event subscribers.

    Subclasses declare which ``EventType``\\s they care about via
    ``get_subscriptions()``.  The ``register()`` helper wires them up to
    an ``EventBus``.
    """

    @abstractmethod
    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        """Return event_type -> callback mapping.

        Called once during ``register()``.  The EventBus stores these
        callbacks directly.
        """
        ...

    def register(self, bus: EventBus) -> None:
        """Subscribe all declared handlers to *bus*."""
        for event_type, callback in self.get_subscriptions().items():
            bus.subscribe(event_type, callback)

    def shutdown(self) -> None:  # noqa: B027
        """Optional cleanup hook.  Called by ``EventBus.stop()``."""
        pass


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


class EventBus:
    """Manages event ingestion, queueing, and dispatch to subscribers.

    Events are appended to a deque on the hot path (``publish()``) and
    drained by a background thread that dispatches to registered callbacks.
    """

    def __init__(self, config: EventBusConfig | None = None) -> None:
        if config is None:
            config = EventBusConfig()
        self._config = config
        self._subscribers: dict[EventType, list[EventCallback]] = defaultdict(list)
        self._queue: collections.deque[Event] = collections.deque()
        self._wake = threading.Event()
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._registered_subscribers: list[EventSubscriber] = []
        self._discard_count: int = 0
        self._last_discard_warning: float = 0.0

    # -- Public API --------------------------------------------------------

    def subscribe(self, event_type: EventType, callback: EventCallback) -> None:
        """Register a callback for a specific event type (thread-safe)."""
        with self._lock:
            self._subscribers[event_type].append(callback)

    def register_subscriber(self, subscriber: EventSubscriber) -> None:
        """Register an ``EventSubscriber`` and wire up its callbacks."""
        subscriber.register(self)
        with self._lock:
            self._registered_subscribers.append(subscriber)

    def publish_on_stream(self, stream: Any, event: Event) -> None:
        """Schedule :meth:`publish` as a CUDA host function on *stream*.

        No-op when the EventBus is disabled, avoiding the overhead of
        scheduling a host function on the CUDA stream entirely.
        """
        if not self._config.enabled:
            return
        stream.launch_host_func(self.publish, event)

    def publish(self, event: Event) -> None:
        """Submit an event (hot path — non-blocking).

        The event's ``timestamp`` is set to ``time.time()`` at call time.
        When the queue is full the event is silently discarded with a
        rate-limited warning (at most once per second).
        """
        if not self._config.enabled:
            return

        if len(self._queue) >= self._config.max_queue_size:
            self._discard_count += 1
            now = time.monotonic()
            if now - self._last_discard_warning >= 1.0:
                logger.warning(
                    "EventBus queue full (max_queue_size=%d), "
                    "%d event(s) discarded so far",
                    self._config.max_queue_size,
                    self._discard_count,
                )
                self._last_discard_warning = now
            return

        event.timestamp = time.time()
        self._queue.append(event)
        self._wake.set()

    def start(self) -> None:
        """Start the background drain thread.  No-op when disabled or
        already running."""
        if not self._config.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="EventBus",
        )
        logger.debug("Starting EventBus drain thread...")
        self._thread.start()

    def stop(self) -> None:
        """Stop the drain thread, flush remaining events, and shut down
        all registered subscribers.  Safe to call when not started."""
        self._stop_flag.set()
        self._wake.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()

        # Final drain
        self._drain_all()

        # Shutdown subscribers
        with self._lock:
            snapshot = list(self._registered_subscribers)
        for sub in snapshot:
            try:
                sub.shutdown()
            except Exception:
                logger.exception(
                    "EventBus: error shutting down %s",
                    type(sub).__name__,
                )

    # -- Internal ----------------------------------------------------------

    def _run(self) -> None:
        """Drain loop: wait for wake signal or timeout, then drain."""
        while not self._stop_flag.is_set():
            self._wake.wait(timeout=0.1)
            self._wake.clear()
            self._drain_all()

    def _drain_all(self) -> None:
        """Pop all queued events and dispatch to subscribers."""
        with self._lock:
            snapshot = dict(self._subscribers)

        while True:
            try:
                event = self._queue.popleft()
            except IndexError:
                break
            for cb in snapshot.get(event.event_type, []):
                try:
                    cb(event)
                except Exception:
                    logger.exception(
                        "EventBus: error in callback for %s",
                        event.event_type.value,
                    )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_global_bus = EventBus(EventBusConfig(enabled=False))
_observability_enabled: bool = False


def is_observability_enabled() -> bool:
    """Fast check for whether observability is active.

    Use this to guard expensive event-construction or CUDA host-function
    scheduling when observability is disabled.
    """
    return _observability_enabled


def get_event_bus() -> EventBus:
    """Return the current global EventBus singleton."""
    return _global_bus


def init_event_bus(config: EventBusConfig | None = None) -> EventBus:
    """Replace the global singleton with a new EventBus built from *config*.

    Returns the newly created bus.
    """
    global _global_bus, _observability_enabled
    _global_bus = EventBus(config)
    _observability_enabled = config.enabled if config else True
    return _global_bus
