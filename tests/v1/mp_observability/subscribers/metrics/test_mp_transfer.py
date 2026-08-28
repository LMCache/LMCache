# SPDX-License-Identifier: Apache-2.0

"""Tests for MPTransferCountersSubscriber."""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache import torch_device_type
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.mp_transfer import (
    MPTransferCountersSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.counter_helpers import (
    counter_delta,
    counter_value,
    read_tagged_counters,
)

_DRAIN_WAIT = 0.15

_SUBMITTED_STORES = "lmcache_mp.num_submitted_stores"
_FINISHED_STORES = "lmcache_mp.num_finished_stores"
_SUBMITTED_RETRIEVES = "lmcache_mp.num_submitted_retrieves"
_FINISHED_RETRIEVES = "lmcache_mp.num_finished_retrieves"

_DEVICE = f"{torch_device_type}:0"
_OTHER_DEVICE = f"{torch_device_type}:1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _submitted(
    event_type: EventType,
    device: str = _DEVICE,
    session_id: str = "req-1",
) -> Event:
    """Build an MP_{STORE,RETRIEVE}_SUBMITTED event.

    Mirrors the metadata the transfer module actually publishes: the
    CPU-synchronous sentinels carry ``device`` and nothing else.
    """
    return Event(
        event_type=event_type,
        session_id=session_id,
        metadata={"device": device},
    )


def _store_end(
    stored_count: int = 4,
    device: str = _DEVICE,
    session_id: str = "req-1",
) -> Event:
    return Event(
        event_type=EventType.MP_STORE_END,
        session_id=session_id,
        metadata={
            "stored_count": stored_count,
            "device": device,
            "engine_id": 0,
            "model_name": "test-model",
            "total_bytes": 1024,
            "num_tokens": 256,
        },
    )


def _retrieve_end(
    retrieved_count: int = 4,
    device: str = _DEVICE,
    session_id: str = "req-1",
) -> Event:
    return Event(
        event_type=EventType.MP_RETRIEVE_END,
        session_id=session_id,
        metadata={
            "retrieved_count": retrieved_count,
            "device": device,
            "engine_id": 0,
            "model_name": "test-model",
            "cache_salt": "",
            "total_bytes": 1024,
            "num_tokens": 256,
        },
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def subscriber():
    return MPTransferCountersSubscriber()


# ---------------------------------------------------------------------------
# Subscription surface
# ---------------------------------------------------------------------------


class TestSubscriptions:
    def test_subscribes_to_all_four_transfer_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert set(subs) == {
            EventType.MP_STORE_SUBMITTED,
            EventType.MP_STORE_END,
            EventType.MP_RETRIEVE_SUBMITTED,
            EventType.MP_RETRIEVE_END,
        }

    def test_does_not_subscribe_to_start_events(self, subscriber):
        # START events fire on the device stream between SUBMITTED and END;
        # counting them too would double-count every transfer.
        subs = subscriber.get_subscriptions()
        assert EventType.MP_STORE_START not in subs
        assert EventType.MP_RETRIEVE_START not in subs


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


class TestStoreCounters:
    def test_submitted_and_finished_count_one_each(self, subscriber):
        before = read_tagged_counters()
        subscriber._on_store_submitted(_submitted(EventType.MP_STORE_SUBMITTED))
        subscriber._on_store_finished(_store_end())
        delta = counter_delta(before, read_tagged_counters())

        assert counter_value(delta, _SUBMITTED_STORES, device=_DEVICE) == 1
        assert counter_value(delta, _FINISHED_STORES, device=_DEVICE) == 1

    def test_finished_counts_stores_that_committed_nothing(self, subscriber):
        # A store can complete without committing a single chunk (fail-closed
        # path). It still left the device stream, so it must be counted --
        # otherwise submitted-minus-finished would report a phantom in-flight
        # transfer forever.
        before = read_tagged_counters()
        subscriber._on_store_finished(_store_end(stored_count=0))
        delta = counter_delta(before, read_tagged_counters())

        assert counter_value(delta, _FINISHED_STORES, device=_DEVICE) == 1

    def test_stores_do_not_touch_retrieve_counters(self, subscriber):
        before = read_tagged_counters()
        subscriber._on_store_submitted(_submitted(EventType.MP_STORE_SUBMITTED))
        subscriber._on_store_finished(_store_end())
        delta = counter_delta(before, read_tagged_counters())

        assert counter_value(delta, _SUBMITTED_RETRIEVES, device=_DEVICE) == 0
        assert counter_value(delta, _FINISHED_RETRIEVES, device=_DEVICE) == 0


class TestRetrieveCounters:
    def test_submitted_and_finished_count_one_each(self, subscriber):
        before = read_tagged_counters()
        subscriber._on_retrieve_submitted(_submitted(EventType.MP_RETRIEVE_SUBMITTED))
        subscriber._on_retrieve_finished(_retrieve_end())
        delta = counter_delta(before, read_tagged_counters())

        assert counter_value(delta, _SUBMITTED_RETRIEVES, device=_DEVICE) == 1
        assert counter_value(delta, _FINISHED_RETRIEVES, device=_DEVICE) == 1

    def test_finished_counts_retrieves_that_missed(self, subscriber):
        before = read_tagged_counters()
        subscriber._on_retrieve_finished(_retrieve_end(retrieved_count=0))
        delta = counter_delta(before, read_tagged_counters())

        assert counter_value(delta, _FINISHED_RETRIEVES, device=_DEVICE) == 1

    def test_retrieves_do_not_touch_store_counters(self, subscriber):
        before = read_tagged_counters()
        subscriber._on_retrieve_submitted(_submitted(EventType.MP_RETRIEVE_SUBMITTED))
        subscriber._on_retrieve_finished(_retrieve_end())
        delta = counter_delta(before, read_tagged_counters())

        assert counter_value(delta, _SUBMITTED_STORES, device=_DEVICE) == 0
        assert counter_value(delta, _FINISHED_STORES, device=_DEVICE) == 0


# ---------------------------------------------------------------------------
# In-flight derivation (the reason the pairs exist)
# ---------------------------------------------------------------------------


class TestInFlightDerivation:
    def test_submitted_minus_finished_is_in_flight_per_device(self, subscriber):
        before = read_tagged_counters()
        # Three stores enqueued on cuda:0, two of them completed.
        for _ in range(3):
            subscriber._on_store_submitted(_submitted(EventType.MP_STORE_SUBMITTED))
        for _ in range(2):
            subscriber._on_store_finished(_store_end())
        delta = counter_delta(before, read_tagged_counters())

        in_flight = counter_value(
            delta, _SUBMITTED_STORES, device=_DEVICE
        ) - counter_value(delta, _FINISHED_STORES, device=_DEVICE)
        assert in_flight == 1

    def test_devices_are_counted_independently(self, subscriber):
        before = read_tagged_counters()
        subscriber._on_retrieve_submitted(
            _submitted(EventType.MP_RETRIEVE_SUBMITTED, device=_DEVICE)
        )
        subscriber._on_retrieve_submitted(
            _submitted(EventType.MP_RETRIEVE_SUBMITTED, device=_OTHER_DEVICE)
        )
        subscriber._on_retrieve_submitted(
            _submitted(EventType.MP_RETRIEVE_SUBMITTED, device=_OTHER_DEVICE)
        )
        subscriber._on_retrieve_finished(_retrieve_end(device=_OTHER_DEVICE))
        delta = counter_delta(before, read_tagged_counters())

        assert counter_value(delta, _SUBMITTED_RETRIEVES, device=_DEVICE) == 1
        assert counter_value(delta, _SUBMITTED_RETRIEVES, device=_OTHER_DEVICE) == 2
        assert counter_value(delta, _FINISHED_RETRIEVES, device=_OTHER_DEVICE) == 1
        assert counter_value(delta, _FINISHED_RETRIEVES, device=_DEVICE) == 0

    def test_submitted_and_finished_share_one_label_set(self, subscriber):
        # The two sides must be subtractable without a PromQL aggregation,
        # so the END side must not carry engine_id / model_name attributes
        # that the SUBMITTED side cannot supply.
        before = read_tagged_counters()
        subscriber._on_store_submitted(_submitted(EventType.MP_STORE_SUBMITTED))
        subscriber._on_store_finished(_store_end())
        delta = counter_delta(before, read_tagged_counters())

        submitted_attrs = {
            attrs for (name, attrs), v in delta.items() if name == _SUBMITTED_STORES
        }
        finished_attrs = {
            attrs for (name, attrs), v in delta.items() if name == _FINISHED_STORES
        }
        assert frozenset({("device", _DEVICE)}) in submitted_attrs
        assert frozenset({("device", _DEVICE)}) in finished_attrs


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_missing_device_still_counts_without_attr(self, subscriber):
        before = read_tagged_counters()
        subscriber._on_store_submitted(
            Event(event_type=EventType.MP_STORE_SUBMITTED, session_id="req-x")
        )
        delta = counter_delta(before, read_tagged_counters())

        assert counter_value(delta, _SUBMITTED_STORES) == 1
        assert counter_value(delta, _SUBMITTED_STORES, device=_DEVICE) == 0

    def test_non_string_device_is_stringified(self, subscriber):
        before = read_tagged_counters()
        subscriber._on_store_submitted(
            Event(
                event_type=EventType.MP_STORE_SUBMITTED,
                session_id="req-x",
                metadata={"device": 0},
            )
        )
        delta = counter_delta(before, read_tagged_counters())

        assert counter_value(delta, _SUBMITTED_STORES, device="0") == 1


# ---------------------------------------------------------------------------
# End-to-end via EventBus
# ---------------------------------------------------------------------------


class TestEventBusIntegration:
    def test_full_store_and_retrieve_cycle_via_bus(self):
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(MPTransferCountersSubscriber())

        before = read_tagged_counters()
        bus.start()
        bus.publish(_submitted(EventType.MP_STORE_SUBMITTED))
        bus.publish(_store_end())
        bus.publish(_submitted(EventType.MP_RETRIEVE_SUBMITTED))
        bus.publish(_retrieve_end())
        time.sleep(_DRAIN_WAIT)
        bus.stop()
        delta = counter_delta(before, read_tagged_counters())

        assert counter_value(delta, _SUBMITTED_STORES, device=_DEVICE) == 1
        assert counter_value(delta, _FINISHED_STORES, device=_DEVICE) == 1
        assert counter_value(delta, _SUBMITTED_RETRIEVES, device=_DEVICE) == 1
        assert counter_value(delta, _FINISHED_RETRIEVES, device=_DEVICE) == 1
