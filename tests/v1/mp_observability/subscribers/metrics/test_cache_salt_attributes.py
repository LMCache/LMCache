# SPDX-License-Identifier: Apache-2.0

"""Tests that metric subscribers emit ``cache_salt`` attributes.

Covers:

- L1MetricsSubscriber groups by ``ObjectKey.cache_salt``.
- SMMetricsSubscriber groups by ``ObjectKey.cache_salt``.
- L2MetricsSubscriber groups store events by key salt and tags prefetch
  events with the single ``cache_salt`` on the metadata.
- Batched events whose keys span multiple tenants produce multiple
  per-tenant datapoints.

Uses ``InMemoryMetricReader`` via the shared ``otel_setup`` module so that
counter values can be read back and asserted against.
"""

# Standard
from dataclasses import dataclass
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.l1 import (
    L1MetricsSubscriber,
)
from lmcache.v1.mp_observability.subscribers.metrics.l2 import (
    L2MetricsSubscriber,
)
from lmcache.v1.mp_observability.subscribers.metrics.sm import (
    SMMetricsSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.otel_setup import reader as _reader

_DRAIN_WAIT = 0.15


@dataclass(frozen=True)
class _FakeKey:
    """Minimal stand-in for ObjectKey with only the attribute the
    subscribers read."""

    token: str
    cache_salt: str


def _keys(cache_salt: str, n: int) -> list[_FakeKey]:
    return [
        _FakeKey(token=f"{cache_salt}-{i}", cache_salt=cache_salt) for i in range(n)
    ]


def _snapshot_by_salt(name: str) -> dict[str, int]:
    """Return ``{cache_salt: value}`` for every datapoint on *name*."""
    data = _reader.get_metrics_data()
    out: dict[str, int] = {}
    if data is None:
        return out
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name != name:
                    continue
                for dp in metric.data.data_points:
                    if not hasattr(dp, "value"):
                        continue
                    salt = dp.attributes.get("cache_salt", "<missing>")
                    out[salt] = out.get(salt, 0) + int(dp.value)
    return out


def _delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        salt: after.get(salt, 0) - before.get(salt, 0)
        for salt in set(before) | set(after)
    }


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


class TestL1BatchGrouping:
    def test_multi_tenant_read_batch_emits_per_tenant_counts(self, bus):
        before = _snapshot_by_salt("lmcache_mp.l1_read_keys")

        bus.register_subscriber(L1MetricsSubscriber())
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L1_READ_FINISHED,
                metadata={"keys": _keys("u1", 3) + _keys("u2", 4)},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        after = _snapshot_by_salt("lmcache_mp.l1_read_keys")
        delta = _delta(before, after)

        assert delta.get("u1", 0) == 3
        assert delta.get("u2", 0) == 4

    def test_single_tenant_eviction_batch(self, bus):
        before = _snapshot_by_salt("lmcache_mp.l1_evicted_keys")

        bus.register_subscriber(L1MetricsSubscriber())
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L1_KEYS_EVICTED,
                metadata={"keys": _keys("tenant-x", 5)},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        after = _snapshot_by_salt("lmcache_mp.l1_evicted_keys")
        delta = _delta(before, after)

        assert delta.get("tenant-x", 0) == 5


class TestSMGrouping:
    def test_read_prefetched_per_tenant(self, bus):
        before_req = _snapshot_by_salt("lmcache_mp.sm_read_requests")
        before_ok = _snapshot_by_salt("lmcache_mp.sm_read_succeed_keys")
        before_fail = _snapshot_by_salt("lmcache_mp.sm_read_failed_keys")

        bus.register_subscriber(SMMetricsSubscriber())
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.SM_READ_PREFETCHED,
                metadata={
                    "succeeded_keys": _keys("a", 2) + _keys("b", 3),
                    "failed_keys": _keys("a", 1),
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        after_req = _snapshot_by_salt("lmcache_mp.sm_read_requests")
        after_ok = _snapshot_by_salt("lmcache_mp.sm_read_succeed_keys")
        after_fail = _snapshot_by_salt("lmcache_mp.sm_read_failed_keys")

        assert _delta(before_ok, after_ok).get("a", 0) == 2
        assert _delta(before_ok, after_ok).get("b", 0) == 3
        assert _delta(before_fail, after_fail).get("a", 0) == 1
        # Request counter records one increment per distinct salt touched.
        req_delta = _delta(before_req, after_req)
        assert req_delta.get("a", 0) == 1
        assert req_delta.get("b", 0) == 1


class TestL2PrefetchSingleSalt:
    def test_prefetch_lookup_submitted_tagged(self, bus):
        before = _snapshot_by_salt("lmcache_mp.l2_prefetch_lookup_keys")

        bus.register_subscriber(L2MetricsSubscriber())
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOOKUP_SUBMITTED,
                metadata={
                    "request_id": 1,
                    "key_count": 7,
                    "adapter_count": 2,
                    "cache_salt": "prefetch-user",
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        after = _snapshot_by_salt("lmcache_mp.l2_prefetch_lookup_keys")
        delta = _delta(before, after)
        assert delta.get("prefetch-user", 0) == 7

    def test_store_submitted_groups_keys_by_salt(self, bus):
        """Task counter is per-task (tenant-agnostic); key counter splits."""
        before_keys = _snapshot_by_salt("lmcache_mp.l2_store_keys")

        bus.register_subscriber(L2MetricsSubscriber())
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L2_STORE_SUBMITTED,
                metadata={
                    "adapter_index": 0,
                    "key_count": 5,
                    "keys": _keys("x", 2) + _keys("y", 3),
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        after_keys = _snapshot_by_salt("lmcache_mp.l2_store_keys")
        key_delta = _delta(before_keys, after_keys)
        assert key_delta.get("x", 0) == 2
        assert key_delta.get("y", 0) == 3
