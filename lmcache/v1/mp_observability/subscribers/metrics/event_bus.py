# SPDX-License-Identifier: Apache-2.0

"""EventBus self-metrics — surface bus health to OTel.

Registers four metrics that observe the :class:`EventBus` itself so
operators can detect backpressure or dispatch failures:

- ``lmcache_mp.event_bus.queue_depth`` (gauge): events waiting for dispatch.
- ``lmcache_mp.event_bus.drain_lag_seconds`` (gauge): age of the oldest
  queued event; rises when the drain thread falls behind.
- ``lmcache_mp.event_bus.dropped_events_total`` (observable counter):
  cumulative events dropped because the queue was at ``max_queue_size``.
- ``lmcache_mp.event_bus.subscriber_exceptions`` (observable counter,
  tagged by ``subscriber_name``): cumulative exceptions raised by
  subscriber callbacks during dispatch.

Unlike most subscribers in this package these metrics are not driven by
events — they observe bus state directly via the ``EventBus`` accessors.
The registration is exposed as a free function rather than as an
``EventSubscriber`` because there are no events to subscribe to.
"""

# Future
from __future__ import annotations

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event_bus import EventBus
from lmcache.v1.mp_observability.otel_init import register_gauge

logger = init_logger(__name__)

DEFAULT_METER_NAME = "lmcache.event_bus"


def init_event_bus_self_metrics(
    bus: EventBus,
    meter_name: str = DEFAULT_METER_NAME,
) -> None:
    """Register OTel gauges and counters that observe *bus* itself.

    Idempotency is the caller's responsibility — invoking this twice
    against the same meter will create duplicate instruments.  Call once
    per process during MP observability init.

    Args:
        bus: The :class:`EventBus` to observe.  Held by reference; the
            registered callbacks read its state on every scrape.
        meter_name: OTel meter name for these instruments.  Defaults to
            ``"lmcache.event_bus"``.

    Raises:
        Nothing.  When ``opentelemetry`` is not importable, gauge
        registration degrades to a no-op (via :func:`register_gauge`)
        and counter registration is skipped with a debug log.
    """
    register_gauge(
        meter_name,
        "lmcache_mp.event_bus.queue_depth",
        "Number of events currently queued in the EventBus.",
        bus.queue_depth,
    )
    register_gauge(
        meter_name,
        "lmcache_mp.event_bus.drain_lag_seconds",
        (
            "Seconds since the oldest queued event was published; "
            "0.0 when the queue is empty.  Rising values indicate "
            "the drain thread is falling behind publish rate."
        ),
        bus.oldest_event_lag_seconds,
    )

    try:
        # Third Party
        from opentelemetry import metrics as otel_metrics
    except ImportError:
        logger.debug(
            "opentelemetry package not found, "
            "skipping event_bus dropped/exception counters"
        )
        return

    meter = otel_metrics.get_meter(meter_name)

    def _dropped_callback(_options):
        return [otel_metrics.Observation(bus.dropped_events_count())]

    meter.create_observable_counter(
        "lmcache_mp.event_bus.dropped_events_total",
        callbacks=[_dropped_callback],
        description=(
            "Cumulative count of events dropped because the EventBus "
            "queue was at max_queue_size."
        ),
    )

    def _exceptions_callback(_options):
        snapshot = bus.subscriber_exception_counts()
        return [
            otel_metrics.Observation(count, {"subscriber_name": name})
            for name, count in snapshot.items()
        ]

    meter.create_observable_counter(
        "lmcache_mp.event_bus.subscriber_exceptions",
        callbacks=[_exceptions_callback],
        description=(
            "Cumulative count of exceptions raised by subscriber "
            "callbacks during EventBus dispatch, tagged by "
            "``subscriber_name``."
        ),
    )
