# SPDX-License-Identifier: Apache-2.0
"""Continuous usage reporting for the multiprocess (MP) cache server.

Metrics are defined map-reduce style; the :class:`MetricSpec` contract
and the default metric registry live in :mod:`.metric_specs`. An
EventBus subscriber buffers samples on the bus's drain thread;
a dedicated flush thread reduces and sends every
``LMCACHE_USAGE_TRACK_INTERVAL`` seconds (default 600). Empty intervals
are still sent and double as session heartbeats.

This module is not re-exported from the package root so that importing
:mod:`lmcache.usage_telemetry` (done by the single-process engine path)
never pulls in :mod:`lmcache.v1.mp_observability`.

Note:
    The default metrics are sourced from ``MP_RETRIEVE_END`` /
    ``MP_STORE_END``, which only the lmcache-driven transfer path emits;
    engine-driven transfers are not counted.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import fields
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.usage_telemetry.flush import start_usage_flush_thread
from lmcache.usage_telemetry.guard import swallow_telemetry_errors
from lmcache.usage_telemetry.identity import (
    get_usage_identity,
    is_usage_tracking_enabled,
)
from lmcache.usage_telemetry.messages import ContinuousContextMessage, DeploymentMode
from lmcache.usage_telemetry.metric_specs import MetricSpec, default_metric_specs
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


_NON_METRIC_FIELDS = frozenset({"sequence_number", "uptime_seconds"})
"""``ContinuousContextMessage`` fields filled by the reporter itself."""


class MPContinuousUsageReporter(EventSubscriber):
    """Continuous usage reporter for the multiprocess cache server.

    Buffers one sample per spec-matched event and, every
    ``LMCACHE_USAGE_TRACK_INTERVAL`` seconds, reduces each buffer into
    its ``ContinuousContextMessage`` field and sends the message.
    Interval data is dropped, not retried, when a send fails; gaps in
    ``sequence_number`` mark lost intervals. A final flush is sent when
    the owning EventBus stops.
    """

    def __init__(
        self,
        chunk_size: int,
        sender: UsageMessageSender | None = None,
        specs: list[MetricSpec] | None = None,
        max_buffered_samples: int = 65536,
    ) -> None:
        """Initialize the reporter and start its flush thread.

        Args:
            chunk_size: The server chunk size in tokens, used by the
                default metric specs.
            sender: Message transport; ``None`` selects the default HTTP
                sender.
            specs: Metric definitions; ``None`` selects the defaults.
                Fields must map one-to-one onto the metric fields of
                ``ContinuousContextMessage``.
            max_buffered_samples: Per-metric buffer size that triggers an
                early flush, bounding memory between flushes.

        Raises:
            ValueError: If the spec fields do not cover exactly the
                metric fields of ``ContinuousContextMessage``, or
                ``max_buffered_samples`` is not positive.
        """
        if max_buffered_samples <= 0:
            raise ValueError(
                f"max_buffered_samples must be positive, got {max_buffered_samples}"
            )
        self._specs = specs if specs is not None else default_metric_specs(chunk_size)
        spec_fields = [spec.field for spec in self._specs]
        message_fields = {
            f.name for f in fields(ContinuousContextMessage)
        } - _NON_METRIC_FIELDS
        if (
            len(set(spec_fields)) != len(spec_fields)
            or set(spec_fields) != message_fields
        ):
            raise ValueError(
                f"Metric specs must cover the ContinuousContextMessage metric "
                f"fields {sorted(message_fields)} exactly once each, got "
                f"{sorted(spec_fields)}"
            )
        self._sender = sender if sender is not None else DEFAULT_SENDER
        self._max_buffered_samples = max_buffered_samples
        self._lock = threading.Lock()
        self._buffers: dict[str, list[int | float]] = {
            spec.field: [] for spec in self._specs
        }
        self._sequence_number = 0
        self._start_monotonic = time.monotonic()
        self._flush_thread = start_usage_flush_thread(
            "lmcache-usage-continuous", self.flush
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        subscriptions: dict[EventType, list[MetricSpec]] = {}
        for spec in self._specs:
            subscriptions.setdefault(spec.event_type, []).append(spec)
        return {
            event_type: self._make_callback(event_specs)
            for event_type, event_specs in subscriptions.items()
        }

    def shutdown(self) -> None:
        """Stop the flush thread and send a final partial-interval flush.

        Called by ``EventBus.stop()``.
        """
        self._flush_thread.stop()
        self.flush()

    @swallow_telemetry_errors
    def flush(self) -> None:
        """Reduce the buffered samples, send them, and reset the buffers.

        Called periodically by the internal flush thread; safe to call
        from any thread. When usage tracking is disabled the samples are
        dropped without sending. Never raises.
        """
        with self._lock:
            buffers = self._buffers
            self._buffers = {spec.field: [] for spec in self._specs}
            self._sequence_number += 1
            sequence_number = self._sequence_number
        if not is_usage_tracking_enabled():
            return
        metric_fields = {
            spec.field: int(spec.reduce(buffers[spec.field])) for spec in self._specs
        }
        message = ContinuousContextMessage(
            **metric_fields,
            sequence_number=sequence_number,
            uptime_seconds=time.monotonic() - self._start_monotonic,
        )
        payload = build_usage_payload(
            message, get_usage_identity(), DeploymentMode.MP_SERVER
        )
        self._sender.send(usage_server_url(message.ENDPOINT), payload)

    def _make_callback(self, event_specs: list[MetricSpec]) -> EventCallback:
        """Build the drain-thread callback for one event type.

        The callback only buffers samples; a full buffer wakes the
        flush thread early.
        """

        @swallow_telemetry_errors
        def _on_event(event: Event) -> None:
            overflow = False
            with self._lock:
                for spec in event_specs:
                    sample = spec.extract(event)
                    if sample is None:
                        continue
                    buffer = self._buffers[spec.field]
                    buffer.append(sample)
                    if len(buffer) >= self._max_buffered_samples:
                        overflow = True
            if overflow:
                self._flush_thread.wake()

        return _on_event


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
