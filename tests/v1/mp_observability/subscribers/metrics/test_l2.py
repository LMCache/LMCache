# SPDX-License-Identifier: Apache-2.0

"""Tests for L2MetricsSubscriber.

Uses ``InMemoryMetricReader`` to read back actual OTel counter values
and assert exact counts after publishing known events through the EventBus.

OTel only allows one MeterProvider per process, so we use a module-scoped
provider and assert on counter **deltas** between before/after snapshots.
"""

# Standard
import time

# Third Party
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.l2 import (
    L2MetricsSubscriber,
)

# Time for the drain thread to process queued events.
_DRAIN_WAIT = 0.15


# ---------------------------------------------------------------------------
# Module-scoped OTel provider (single provider for entire test file)
# ---------------------------------------------------------------------------

_reader = InMemoryMetricReader()
_provider = MeterProvider(metric_readers=[_reader])
metrics.set_meter_provider(_provider)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_counters() -> dict[str, int]:
    """Snapshot all counter values from the module-level reader."""
    data = _reader.get_metrics_data()
    result: dict[str, int] = {}
    if data is None:
        return result
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                for dp in metric.data.data_points:
                    result[metric.name] = int(dp.value)
    return result


def _counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Compute the difference between two counter snapshots."""
    all_keys = set(before) | set(after)
    return {k: after.get(k, 0) - before.get(k, 0) for k in all_keys}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    sub = L2MetricsSubscriber()
    bus.register_subscriber(sub)
    return sub


@pytest.fixture
def snapshot():
    """Capture counters before the test; yield a callable that returns deltas."""
    before = _read_counters()

    def get_delta() -> dict[str, int]:
        return _counter_delta(before, _read_counters())

    return get_delta


# ---------------------------------------------------------------------------
# Store events
# ---------------------------------------------------------------------------


class TestL2StoreMetrics:
    def test_store_submitted_counts(self, bus, subscriber, snapshot):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L2_STORE_SUBMITTED,
                metadata={"adapter_index": 0, "key_count": 10},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.L2_STORE_SUBMITTED,
                metadata={"adapter_index": 1, "key_count": 5},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l2_store_tasks"] == 2
        assert delta["lmcache_mp.l2_store_keys"] == 15

    def test_store_completed_success(self, bus, subscriber, snapshot):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L2_STORE_COMPLETED,
                metadata={"adapter_index": 0, "succeeded_count": 8, "failed_count": 0},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l2_store_completed"] == 1
        assert delta["lmcache_mp.l2_store_succeeded_keys"] == 8
        assert delta.get("lmcache_mp.l2_store_failed_keys", 0) == 0

    def test_store_completed_with_failures(self, bus, subscriber, snapshot):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L2_STORE_COMPLETED,
                metadata={"adapter_index": 0, "succeeded_count": 3, "failed_count": 7},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l2_store_completed"] == 1
        assert delta["lmcache_mp.l2_store_succeeded_keys"] == 3
        assert delta["lmcache_mp.l2_store_failed_keys"] == 7

    def test_store_full_lifecycle(self, bus, subscriber, snapshot):
        """Simulate warmup: submit 20 keys, all succeed."""
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L2_STORE_SUBMITTED,
                metadata={"adapter_index": 0, "key_count": 20},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.L2_STORE_COMPLETED,
                metadata={"adapter_index": 0, "succeeded_count": 20, "failed_count": 0},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l2_store_tasks"] == 1
        assert delta["lmcache_mp.l2_store_keys"] == 20
        assert delta["lmcache_mp.l2_store_completed"] == 1
        assert delta["lmcache_mp.l2_store_succeeded_keys"] == 20
        assert delta.get("lmcache_mp.l2_store_failed_keys", 0) == 0


# ---------------------------------------------------------------------------
# Prefetch events
# ---------------------------------------------------------------------------


class TestL2PrefetchMetrics:
    def test_lookup_submitted_counts(self, bus, subscriber, snapshot):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOOKUP_SUBMITTED,
                metadata={"request_id": 1, "key_count": 12, "adapter_count": 2},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l2_prefetch_lookups"] == 1
        assert delta["lmcache_mp.l2_prefetch_lookup_keys"] == 12

    def test_lookup_completed_counts_hits(self, bus, subscriber, snapshot):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOOKUP_COMPLETED,
                metadata={"request_id": 1, "prefix_hit_count": 10},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l2_prefetch_hit_keys"] == 10

    def test_load_submitted_counts(self, bus, subscriber, snapshot):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOAD_SUBMITTED,
                metadata={"request_id": 1, "key_count": 10, "adapter_count": 2},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l2_prefetch_load_tasks"] == 2
        assert delta["lmcache_mp.l2_prefetch_load_keys"] == 10

    def test_load_completed_counts(self, bus, subscriber, snapshot):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOAD_COMPLETED,
                metadata={"request_id": 1, "loaded_count": 9, "failed_count": 1},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l2_prefetch_loaded_keys"] == 9
        assert delta["lmcache_mp.l2_prefetch_failed_keys"] == 1

    def test_prefetch_full_lifecycle(self, bus, subscriber, snapshot):
        """Simulate query: lookup 20 keys, 18 prefix hits, all 18 load OK."""
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOOKUP_SUBMITTED,
                metadata={"request_id": 42, "key_count": 20, "adapter_count": 1},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOOKUP_COMPLETED,
                metadata={"request_id": 42, "prefix_hit_count": 18},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOAD_SUBMITTED,
                metadata={"request_id": 42, "key_count": 18, "adapter_count": 1},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOAD_COMPLETED,
                metadata={"request_id": 42, "loaded_count": 18, "failed_count": 0},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l2_prefetch_lookups"] == 1
        assert delta["lmcache_mp.l2_prefetch_lookup_keys"] == 20
        assert delta["lmcache_mp.l2_prefetch_hit_keys"] == 18
        assert delta["lmcache_mp.l2_prefetch_load_tasks"] == 1
        assert delta["lmcache_mp.l2_prefetch_load_keys"] == 18
        assert delta["lmcache_mp.l2_prefetch_loaded_keys"] == 18
        assert delta.get("lmcache_mp.l2_prefetch_failed_keys", 0) == 0


# ---------------------------------------------------------------------------
# Subscription wiring
# ---------------------------------------------------------------------------


class TestL2MetricsSubscriptions:
    def test_subscriptions_cover_all_l2_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.L2_STORE_SUBMITTED in subs
        assert EventType.L2_STORE_COMPLETED in subs
        assert EventType.L2_PREFETCH_LOOKUP_SUBMITTED in subs
        assert EventType.L2_PREFETCH_LOOKUP_COMPLETED in subs
        assert EventType.L2_PREFETCH_LOAD_SUBMITTED in subs
        assert EventType.L2_PREFETCH_LOAD_COMPLETED in subs
        assert len(subs) == 6


# ---------------------------------------------------------------------------
# Accumulation across multiple events
# ---------------------------------------------------------------------------


class TestL2MetricsAccumulation:
    def test_multiple_store_events_accumulate(self, bus, subscriber, snapshot):
        bus.start()
        for _ in range(5):
            bus.publish(
                Event(
                    event_type=EventType.L2_STORE_SUBMITTED,
                    metadata={"adapter_index": 0, "key_count": 3},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.L2_STORE_COMPLETED,
                    metadata={
                        "adapter_index": 0,
                        "succeeded_count": 3,
                        "failed_count": 0,
                    },
                )
            )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l2_store_tasks"] == 5
        assert delta["lmcache_mp.l2_store_keys"] == 15
        assert delta["lmcache_mp.l2_store_completed"] == 5
        assert delta["lmcache_mp.l2_store_succeeded_keys"] == 15

    def test_multiple_prefetch_events_accumulate(self, bus, subscriber, snapshot):
        bus.start()
        for i in range(3):
            bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_LOOKUP_SUBMITTED,
                    metadata={"request_id": i, "key_count": 10, "adapter_count": 1},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_LOOKUP_COMPLETED,
                    metadata={"request_id": i, "prefix_hit_count": 8},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_LOAD_SUBMITTED,
                    metadata={"request_id": i, "key_count": 8, "adapter_count": 1},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_LOAD_COMPLETED,
                    metadata={"request_id": i, "loaded_count": 7, "failed_count": 1},
                )
            )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l2_prefetch_lookups"] == 3
        assert delta["lmcache_mp.l2_prefetch_lookup_keys"] == 30
        assert delta["lmcache_mp.l2_prefetch_hit_keys"] == 24
        assert delta["lmcache_mp.l2_prefetch_load_tasks"] == 3
        assert delta["lmcache_mp.l2_prefetch_load_keys"] == 24
        assert delta["lmcache_mp.l2_prefetch_loaded_keys"] == 21
        assert delta["lmcache_mp.l2_prefetch_failed_keys"] == 3
