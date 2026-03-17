# SPDX-License-Identifier: Apache-2.0

"""L1 metrics subscriber — OTel counters for L1Manager events."""

# Future
from __future__ import annotations

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


class L1MetricsSubscriber(EventSubscriber):
    """Maintains OTel counters for L1Manager operations.

    Metric parity with the old ``L1ManagerStatsLogger``:
    - ``lmcache_mp.l1_read_keys``  — keys read from L1
    - ``lmcache_mp.l1_write_keys`` — keys written to L1
    - ``lmcache_mp.l1_evicted_keys`` — keys evicted from L1
    """

    def __init__(self) -> None:
        meter = metrics.get_meter("lmcache.l1")
        self._read_counter = meter.create_counter(
            "lmcache_mp.l1_read_keys",
            description="Total keys read from L1",
        )
        self._write_counter = meter.create_counter(
            "lmcache_mp.l1_write_keys",
            description="Total keys written to L1",
        )
        self._evicted_counter = meter.create_counter(
            "lmcache_mp.l1_evicted_keys",
            description="Total keys evicted from L1",
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.L1_READ_FINISHED: self._on_read_finished,
            EventType.L1_WRITE_FINISHED: self._on_write_finished,
            EventType.L1_WRITE_FINISHED_AND_READ_RESERVED: self._on_write_finished,
            EventType.L1_KEYS_EVICTED: self._on_evicted,
        }

    def _on_read_finished(self, event: Event) -> None:
        self._read_counter.add(len(event.metadata["keys"]))

    def _on_write_finished(self, event: Event) -> None:
        self._write_counter.add(len(event.metadata["keys"]))

    def _on_evicted(self, event: Event) -> None:
        self._evicted_counter.add(len(event.metadata["keys"]))
