# SPDX-License-Identifier: Apache-2.0

"""Timeout metrics subscriber — an OTel counter for timeout errors.

Increments ``lmcache_mp.timeouts`` once per :class:`~lmcache.v1\
.mp_observability.errors.LMCacheTimeoutError` constructed, tagged by
``exception_type`` so operators can alert on the timeout rate per class on the
Prometheus ``/metrics`` endpoint.  This is an anomaly counter: it should stay
near zero in healthy operation.
"""

# Future
from __future__ import annotations

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


class TimeoutMetricsSubscriber(EventSubscriber):
    """Maintains an OTel counter for timeout errors."""

    def __init__(self) -> None:
        meter = metrics.get_meter("lmcache_mp.health")
        self._timeout_counter = meter.create_counter(
            "lmcache_mp.timeouts",
            description=(
                "Count of LMCacheTimeoutError instances constructed, tagged by "
                "``exception_type`` (the timeout class name). Anomaly counter; "
                "should stay near zero in healthy operation."
            ),
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {EventType.TIMEOUT_RAISED: self._on_timeout}

    def _on_timeout(self, event: Event) -> None:
        exception_type: str = event.metadata["exception_type"]
        self._timeout_counter.add(1, {"exception_type": exception_type})
