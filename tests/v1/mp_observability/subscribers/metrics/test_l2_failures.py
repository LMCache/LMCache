# SPDX-License-Identifier: Apache-2.0

"""Tests for L2FailureMetricsSubscriber.

Verifies that ``L2_PREFETCH_FAILED`` events produce the expected
``lmcache_mp.l2_prefetch_failure`` counter with ``reason`` and
``model_name`` attributes, and that a failed ``L2_STORE_COMPLETED``
produces ``lmcache_mp.l2_store_failure`` with ``l2_name`` and
``model_name``. Uses the shared ``InMemoryMetricReader`` to assert on
counter deltas.
"""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.l2_failures import (
    L2FailureMetricsSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.counter_helpers import (
    counter_delta,
    counter_value,
    make_key,
    read_tagged_counters,
)

_DRAIN_WAIT = 0.15


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    sub = L2FailureMetricsSubscriber()
    bus.register_subscriber(sub)
    return sub


@pytest.fixture
def snapshot():
    before = read_tagged_counters()

    def get_delta():
        return counter_delta(before, read_tagged_counters())

    return get_delta


class TestL2PrefetchFailure:
    def test_reason_l1_oom(self, bus, subscriber, snapshot):
        bus.start()
        keys = [make_key("llama-7b", i) for i in range(4)]
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_FAILED,
                metadata={"reason": "l1_oom", "keys": keys},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.l2_prefetch_failure",
                reason="l1_oom",
                model_name="llama-7b",
            )
            == 4
        )

    def test_reason_not_found(self, bus, subscriber, snapshot):
        bus.start()
        keys = [make_key("mistral-7b", i) for i in range(2)]
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_FAILED,
                metadata={"reason": "not_found", "keys": keys},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.l2_prefetch_failure",
                reason="not_found",
                model_name="mistral-7b",
            )
            == 2
        )

    def test_multi_model_buckets_separately(self, bus, subscriber, snapshot):
        bus.start()
        keys = [
            make_key("llama-7b", 1),
            make_key("mistral-7b", 2),
            make_key("mistral-7b", 3),
        ]
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_FAILED,
                metadata={"reason": "not_found", "keys": keys},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.l2_prefetch_failure",
                reason="not_found",
                model_name="llama-7b",
            )
            == 1
        )
        assert (
            counter_value(
                delta,
                "lmcache_mp.l2_prefetch_failure",
                reason="not_found",
                model_name="mistral-7b",
            )
            == 2
        )

    def test_different_reasons_counted_separately(self, bus, subscriber, snapshot):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_FAILED,
                metadata={
                    "reason": "l1_oom",
                    "keys": [make_key("llama-7b", 0), make_key("llama-7b", 1)],
                },
            )
        )
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_FAILED,
                metadata={
                    "reason": "not_found",
                    "keys": [make_key("llama-7b", 2)],
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.l2_prefetch_failure",
                reason="l1_oom",
                model_name="llama-7b",
            )
            == 2
        )
        assert (
            counter_value(
                delta,
                "lmcache_mp.l2_prefetch_failure",
                reason="not_found",
                model_name="llama-7b",
            )
            == 1
        )


class TestL2StoreFailure:
    """``L2_STORE_COMPLETED`` fires for both outcomes; only failures count."""

    def _publish_store(self, bus, **metadata):
        bus.start()
        bus.publish(Event(event_type=EventType.L2_STORE_COMPLETED, metadata=metadata))
        time.sleep(_DRAIN_WAIT)
        bus.stop()

    def test_failed_store_counts_chunks(self, bus, subscriber, snapshot):
        self._publish_store(
            bus,
            adapter_index=0,
            l2_name="fs",
            succeeded_count=0,
            failed_count=6,
            failed_count_per_model={"llama-7b": 6},
        )
        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.l2_store_failure",
                l2_name="fs",
                model_name="llama-7b",
            )
            == 6
        )

    def test_successful_store_is_not_counted(self, bus, subscriber, snapshot):
        self._publish_store(
            bus,
            adapter_index=0,
            l2_name="fs",
            succeeded_count=6,
            failed_count=0,
        )
        # No l2_store_failure data point may move for a successful task,
        # whatever its attributes.
        delta = snapshot()
        assert all(
            value == 0
            for (name, _attrs), value in delta.items()
            if name == "lmcache_mp.l2_store_failure"
        )

    def test_failure_splits_across_models(self, bus, subscriber, snapshot):
        self._publish_store(
            bus,
            adapter_index=0,
            l2_name="fs",
            succeeded_count=0,
            failed_count=3,
            failed_count_per_model={"llama-7b": 1, "mistral-7b": 2},
        )
        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.l2_store_failure",
                l2_name="fs",
                model_name="llama-7b",
            )
            == 1
        )
        assert (
            counter_value(
                delta,
                "lmcache_mp.l2_store_failure",
                l2_name="fs",
                model_name="mistral-7b",
            )
            == 2
        )

    def test_backends_counted_separately(self, bus, subscriber, snapshot):
        bus.start()
        for l2_name, count in (("fs", 4), ("nixl_store", 7)):
            bus.publish(
                Event(
                    event_type=EventType.L2_STORE_COMPLETED,
                    metadata={
                        "adapter_index": 0,
                        "l2_name": l2_name,
                        "succeeded_count": 0,
                        "failed_count": count,
                        "failed_count_per_model": {"llama-7b": count},
                    },
                )
            )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.l2_store_failure",
                l2_name="fs",
                model_name="llama-7b",
            )
            == 4
        )
        assert (
            counter_value(
                delta,
                "lmcache_mp.l2_store_failure",
                l2_name="nixl_store",
                model_name="llama-7b",
            )
            == 7
        )

    def test_failure_without_per_model_still_counts(self, bus, subscriber, snapshot):
        """An emission site that omits the breakdown must stay countable."""
        self._publish_store(
            bus,
            adapter_index=0,
            l2_name="fs",
            succeeded_count=0,
            failed_count=5,
        )
        delta = snapshot()
        assert counter_value(delta, "lmcache_mp.l2_store_failure", l2_name="fs") == 5


class TestL2FailureSubscriptions:
    def test_subscribes_to_prefetch_failed_and_store_completed(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.L2_PREFETCH_FAILED in subs
        assert EventType.L2_STORE_COMPLETED in subs
        assert len(subs) == 2

    def test_no_subscription_for_normal_l2_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.L2_STORE_SUBMITTED not in subs
        assert EventType.L2_PREFETCH_LOAD_COMPLETED not in subs


class TestL2FailureEdgeCases:
    def test_empty_keys_is_noop(self, bus, subscriber, snapshot):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_FAILED,
                metadata={"reason": "l1_oom", "keys": []},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        for (name, _attrs), val in delta.items():
            assert not (name == "lmcache_mp.l2_prefetch_failure" and val != 0), (
                f"Unexpected emission for empty keys: {name}={val}"
            )
