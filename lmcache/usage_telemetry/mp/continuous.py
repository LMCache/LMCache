# SPDX-License-Identifier: Apache-2.0
"""Continuous usage reporting for the multiprocess (MP) cache server.

An EventBus subscriber accumulates interval counters on the bus's drain
thread; a dedicated flush thread sends a ``ContinuousContextMessage``
every ``LMCACHE_USAGE_TRACK_INTERVAL`` seconds (default 600). Empty
intervals are still sent and double as session heartbeats.

Note:
    Counters are sourced from ``MP_RETRIEVE_END`` / ``MP_STORE_END``,
    which only the lmcache-driven transfer path emits; engine-driven
    transfers are not counted.
"""

# Future
from __future__ import annotations

# Standard
import os
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.usage_telemetry.guard import swallow_telemetry_errors
from lmcache.usage_telemetry.identity import (
    get_usage_identity,
    is_usage_tracking_enabled,
)
from lmcache.usage_telemetry.messages import ContinuousContextMessage, DeploymentMode
from lmcache.usage_telemetry.transport import (
    DEFAULT_SENDER,
    UsageMessageSender,
    build_usage_payload,
    usage_server_url,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import (
    EventBus,
    EventCallback,
    EventSubscriber,
)

logger = init_logger(__name__)


class MPContinuousUsageReporter(EventSubscriber):
    """Continuous usage reporter for the multiprocess cache server.

    Sends a ``ContinuousContextMessage`` (retrieved/stored tokens, stored
    bytes) every ``LMCACHE_USAGE_TRACK_INTERVAL`` seconds. Interval data
    is dropped, not retried, when a send fails; gaps in
    ``sequence_number`` mark lost intervals. A final flush is sent when
    the owning EventBus stops.
    """

    def __init__(
        self,
        chunk_size: int,
        sender: UsageMessageSender | None = None,
    ) -> None:
        """Initialize the reporter and start its flush thread.

        Args:
            chunk_size: The server chunk size in tokens; converts the
                chunk counts carried by store/retrieve events to tokens.
            sender: Message transport; ``None`` selects the default HTTP
                sender.
        """
        self._chunk_size = chunk_size
        self._sender = sender if sender is not None else DEFAULT_SENDER
        # Clamp to >= 1 s: Event.wait(0) would turn the flush loop into a
        # busy spin.
        self._flush_interval: float = max(
            float(os.getenv("LMCACHE_USAGE_TRACK_INTERVAL", "600")), 1.0
        )
        self._lock = threading.Lock()
        self._interval_hit_tokens = 0
        self._interval_stored_tokens = 0
        self._interval_stored_bytes = 0
        self._sequence_number = 0
        self._start_monotonic = time.monotonic()
        self._stop_event = threading.Event()
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="lmcache-usage-report"
        )
        self._flush_thread.start()

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.MP_RETRIEVE_END: self._on_retrieve_end,
            EventType.MP_STORE_END: self._on_store_end,
        }

    def shutdown(self) -> None:
        """Stop the flush thread and send a final partial-interval flush.

        Called by ``EventBus.stop()``.
        """
        self._stop_event.set()
        self.flush()

    @swallow_telemetry_errors
    def flush(self) -> None:
        """Send the current interval counters and reset them.

        Called periodically by the internal flush thread; safe to call
        from any thread. When usage tracking is disabled the counters are
        dropped without sending. Never raises.
        """
        with self._lock:
            hit_tokens = self._interval_hit_tokens
            stored_tokens = self._interval_stored_tokens
            stored_bytes = self._interval_stored_bytes
            self._interval_hit_tokens = 0
            self._interval_stored_tokens = 0
            self._interval_stored_bytes = 0
            self._sequence_number += 1
            sequence_number = self._sequence_number
        if not is_usage_tracking_enabled():
            return
        message = ContinuousContextMessage(
            interval_num_stored_tokens=stored_tokens,
            interval_num_hit_tokens=hit_tokens,
            interval_stored_kv_size=stored_bytes,
            sequence_number=sequence_number,
            uptime_seconds=time.monotonic() - self._start_monotonic,
        )
        payload = build_usage_payload(
            message, get_usage_identity(), DeploymentMode.MP_SERVER
        )
        self._sender.send(
            usage_server_url(message.ENDPOINT, DeploymentMode.MP_SERVER), payload
        )

    def _flush_loop(self) -> None:
        while not self._stop_event.wait(self._flush_interval):
            self.flush()

    @swallow_telemetry_errors
    def _on_retrieve_end(self, event: Event) -> None:
        retrieved_tokens = int(event.metadata["retrieved_count"]) * self._chunk_size
        with self._lock:
            self._interval_hit_tokens += retrieved_tokens

    @swallow_telemetry_errors
    def _on_store_end(self, event: Event) -> None:
        stored_tokens = int(event.metadata["stored_count"]) * self._chunk_size
        stored_bytes = int(event.metadata["total_bytes"])
        with self._lock:
            self._interval_stored_tokens += stored_tokens
            self._interval_stored_bytes += stored_bytes


@swallow_telemetry_errors
def InitializeMPContinuousUsage(
    event_bus: EventBus,
    chunk_size: int,
    sender: UsageMessageSender | None = None,
) -> MPContinuousUsageReporter | None:
    """Start continuous usage reporting for a multiprocess cache server.

    Registers an :class:`MPContinuousUsageReporter` on *event_bus*. Never
    blocks or raises.

    Args:
        event_bus: The server's started EventBus.
        chunk_size: The server chunk size in tokens.
        sender: Message transport; ``None`` selects the default HTTP sender.

    Returns:
        The reporter, or ``None`` when usage tracking is disabled or
        initialization failed.
    """
    if not is_usage_tracking_enabled():
        return None
    logger.info("Initializing MP continuous usage reporting.")
    reporter = MPContinuousUsageReporter(chunk_size, sender)
    event_bus.register_subscriber(reporter)
    return reporter
