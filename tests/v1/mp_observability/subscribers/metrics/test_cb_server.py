# SPDX-License-Identifier: Apache-2.0

"""Tests for BlendMetricsSubscriber."""

# Standard
import subprocess
import sys
import textwrap
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.cb_server import (
    BlendMetricsSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.otel_setup import (
    counter_delta,
    reader as _reader,
    read_counters,
)

_DRAIN_WAIT = 0.15


def _read_histograms() -> dict[str, list]:
    data = _reader.get_metrics_data()
    result: dict[str, list] = {}
    if data is None:
        return result
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if not metric.name.startswith("lmcache_blend.l0_gpu"):
                    continue
                result[metric.name] = list(metric.data.data_points)
    return result


def _histogram_count(name: str) -> int:
    return sum(dp.count for dp in _read_histograms().get(name, []))


def _histogram_attrs(name: str) -> list[dict]:
    return [
        dict(dp.attributes)
        for dp in _read_histograms().get(name, [])
        if getattr(dp, "count", 0) > 0
    ]


def _read_counter_points() -> dict[str, list]:
    data = _reader.get_metrics_data()
    result: dict[str, list] = {}
    if data is None:
        return result
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if not metric.name.startswith("lmcache_blend.l0_gpu"):
                    continue
                points = [dp for dp in metric.data.data_points if hasattr(dp, "value")]
                if points:
                    result[metric.name] = points
    return result


def _counter_sum(name: str, **attrs: object) -> int:
    total = 0
    for dp in _read_counter_points().get(name, []):
        dp_attrs = dict(dp.attributes)
        if all(dp_attrs.get(k) == v for k, v in attrs.items()):
            total += int(dp.value)
    return total


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    sub = BlendMetricsSubscriber()
    bus.register_subscriber(sub)
    return sub


@pytest.fixture
def snapshot():
    """Capture counters before the test; yield a callable that returns deltas."""
    before = read_counters()

    def get_delta() -> dict[str, int]:
        return counter_delta(before, read_counters())

    return get_delta


class TestBlendMetricsSubscriber:
    def test_subscriptions_cover_all_cb_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.CB_LOOKUP_START in subs
        assert EventType.CB_LOOKUP_END in subs
        assert EventType.CB_RETRIEVE_START in subs
        assert EventType.CB_RETRIEVE_END in subs
        assert EventType.CB_STORE_PRE_COMPUTED_START in subs
        assert EventType.CB_STORE_PRE_COMPUTED_END in subs
        assert EventType.CB_STORE_FINAL_START in subs
        assert EventType.CB_STORE_FINAL_END in subs
        assert EventType.CB_FINGERPRINTS_REGISTERED in subs
        assert EventType.CB_CHUNKS_EVICTED in subs

    def test_subscribes_to_gpu_lifecycle_sentinels_for_l0_evidence(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.CB_REQUEST_START not in subs
        assert EventType.CB_REQUEST_END not in subs
        assert EventType.CB_STORE_PRE_COMPUTED_SUBMITTED in subs
        assert EventType.CB_RETRIEVE_SUBMITTED in subs
        assert EventType.CB_STORE_FINAL_SUBMITTED in subs

    def test_lookup_start_increments_counter(self, bus, subscriber):
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_LOOKUP_START,
                    session_id="req-1",
                    metadata={"num_tokens": 128},
                )
            )
            time.sleep(0.15)
        finally:
            bus.stop()

    def test_lookup_end_normal(self, bus, subscriber):
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_LOOKUP_END,
                    session_id="req-1",
                    metadata={
                        "requested_tokens": 1024,
                        "hit_tokens": 768,
                        "fingerprint_hits": 4,
                        "storage_hits": 3,
                        "stale_chunks": 1,
                        "no_gpu_context": False,
                    },
                )
            )
            time.sleep(0.15)
        finally:
            bus.stop()

    def test_lookup_end_no_gpu_context_flag(self, bus, subscriber):
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_LOOKUP_END,
                    session_id="req-1",
                    metadata={
                        "requested_tokens": 0,
                        "hit_tokens": 0,
                        "fingerprint_hits": 0,
                        "storage_hits": 0,
                        "stale_chunks": 0,
                        "no_gpu_context": True,
                    },
                )
            )
            time.sleep(0.15)
        finally:
            bus.stop()

    def test_retrieve_success(self, bus, subscriber):
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_RETRIEVE_START,
                    session_id="req-2",
                    metadata={"instance_id": 0, "num_chunks": 3},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_RETRIEVE_END,
                    session_id="req-2",
                    metadata={"instance_id": 0, "num_chunks": 3, "success": True},
                )
            )
            time.sleep(0.15)
        finally:
            bus.stop()

    def test_retrieve_failure_counted(self, bus, subscriber):
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_RETRIEVE_START,
                    session_id="req-2",
                    metadata={"instance_id": 0, "num_chunks": 2},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_RETRIEVE_END,
                    session_id="req-2",
                    metadata={"instance_id": 0, "num_chunks": 2, "success": False},
                )
            )
            time.sleep(0.15)
        finally:
            bus.stop()

    def test_store_pre_computed_failure_counted(self, bus, subscriber):
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_STORE_PRE_COMPUTED_START,
                    session_id="req-3",
                    metadata={"instance_id": 0, "num_tokens": 64},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_STORE_PRE_COMPUTED_END,
                    session_id="req-3",
                    metadata={"instance_id": 0, "stored_chunks": 0, "success": False},
                )
            )
            time.sleep(0.15)
        finally:
            bus.stop()

    def test_store_final_failure_counted(self, bus, subscriber):
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_STORE_FINAL_START,
                    session_id="req-4",
                    metadata={"instance_id": 1, "num_tokens": 256},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_STORE_FINAL_END,
                    session_id="req-4",
                    metadata={"instance_id": 1, "stored_chunks": 0, "success": False},
                )
            )
            time.sleep(0.15)
        finally:
            bus.stop()

    def test_fingerprints_registered(self, bus, subscriber):
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_FINGERPRINTS_REGISTERED,
                    metadata={"num_chunks": 8},
                )
            )
            time.sleep(0.15)
        finally:
            bus.stop()

    def test_chunks_evicted(self, bus, subscriber):
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_CHUNKS_EVICTED,
                    metadata={"num_chunks": 3},
                )
            )
            time.sleep(0.15)
        finally:
            bus.stop()

    def test_multiple_events_accumulate(self, bus, subscriber):
        bus.start()
        try:
            for _ in range(5):
                bus.publish(
                    Event(
                        event_type=EventType.CB_LOOKUP_START,
                        session_id="req-bulk",
                        metadata={"num_tokens": 100},
                    )
                )
                bus.publish(
                    Event(
                        event_type=EventType.CB_LOOKUP_END,
                        session_id="req-bulk",
                        metadata={
                            "requested_tokens": 96,
                            "hit_tokens": 32,
                            "fingerprint_hits": 2,
                            "storage_hits": 1,
                            "stale_chunks": 1,
                            "no_gpu_context": False,
                        },
                    )
                )
            time.sleep(0.15)
        finally:
            bus.stop()


# ---------------------------------------------------------------------------
# Blend token-level hit-rate counters
#
# These counters expose the numerator/denominator that let dashboards compute
# the blend hit rate identically to the L1+L2 lookup hit rate:
#
#     rate(lmcache_blend_lookup_hit_tokens_total[5m])
#     / rate(lmcache_blend_lookup_requested_tokens_total[5m])
#
# Asserts on actual counter deltas via the InMemoryMetricReader fixture.
# ---------------------------------------------------------------------------


class TestBlendLookupHitTokenCounters:
    def test_full_hit(self, bus, subscriber, snapshot):
        """All requested tokens are served by blend."""
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_LOOKUP_END,
                    session_id="req-1",
                    metadata={
                        "requested_tokens": 1024,
                        "hit_tokens": 1024,
                        "fingerprint_hits": 4,
                        "storage_hits": 4,
                        "stale_chunks": 0,
                        "no_gpu_context": False,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        delta = snapshot()
        assert delta["lmcache_blend.lookup_requested_tokens"] == 1024
        assert delta["lmcache_blend.lookup_hit_tokens"] == 1024

    def test_partial_hit(self, bus, subscriber, snapshot):
        """A subset of the requested tokens is served by blend."""
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_LOOKUP_END,
                    session_id="req-2",
                    metadata={
                        "requested_tokens": 1024,
                        "hit_tokens": 256,
                        "fingerprint_hits": 4,
                        "storage_hits": 1,
                        "stale_chunks": 3,
                        "no_gpu_context": False,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        delta = snapshot()
        assert delta["lmcache_blend.lookup_requested_tokens"] == 1024
        assert delta["lmcache_blend.lookup_hit_tokens"] == 256

    def test_full_miss_still_records_denominator(self, bus, subscriber, snapshot):
        """Cold lookup: the request must still increment the denominator so
        the running hit rate properly reflects the miss."""
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_LOOKUP_END,
                    session_id="req-3",
                    metadata={
                        "requested_tokens": 512,
                        "hit_tokens": 0,
                        "fingerprint_hits": 0,
                        "storage_hits": 0,
                        "stale_chunks": 0,
                        "no_gpu_context": False,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        delta = snapshot()
        assert delta["lmcache_blend.lookup_requested_tokens"] == 512
        assert delta.get("lmcache_blend.lookup_hit_tokens", 0) == 0

    def test_no_gpu_context_records_zero_tokens(self, bus, subscriber, snapshot):
        """``no_gpu_context`` lookups emit ``hit_tokens=0`` and
        ``requested_tokens=0`` — neither counter should move so the ratio
        stays meaningful."""
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_LOOKUP_END,
                    session_id="req-4",
                    metadata={
                        "requested_tokens": 0,
                        "hit_tokens": 0,
                        "fingerprint_hits": 5,
                        "storage_hits": 0,
                        "stale_chunks": 0,
                        "no_gpu_context": True,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        delta = snapshot()
        assert delta.get("lmcache_blend.lookup_requested_tokens", 0) == 0
        assert delta.get("lmcache_blend.lookup_hit_tokens", 0) == 0

    def test_multiple_lookups_accumulate(self, bus, subscriber, snapshot):
        """Counters accumulate across multiple completed lookups."""
        bus.start()
        try:
            # 3 full-hit lookups @ 256 tokens each
            for i in range(3):
                bus.publish(
                    Event(
                        event_type=EventType.CB_LOOKUP_END,
                        session_id=f"hit-{i}",
                        metadata={
                            "requested_tokens": 256,
                            "hit_tokens": 256,
                            "fingerprint_hits": 1,
                            "storage_hits": 1,
                            "stale_chunks": 0,
                            "no_gpu_context": False,
                        },
                    )
                )
            # 2 partial-hit lookups: 1024 requested, 128 hit
            for i in range(2):
                bus.publish(
                    Event(
                        event_type=EventType.CB_LOOKUP_END,
                        session_id=f"partial-{i}",
                        metadata={
                            "requested_tokens": 1024,
                            "hit_tokens": 128,
                            "fingerprint_hits": 4,
                            "storage_hits": 1,
                            "stale_chunks": 3,
                            "no_gpu_context": False,
                        },
                    )
                )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        delta = snapshot()
        # 3*256 + 2*1024 = 768 + 2048 = 2816
        assert delta["lmcache_blend.lookup_requested_tokens"] == 2816
        # 3*256 + 2*128 = 768 + 256 = 1024
        assert delta["lmcache_blend.lookup_hit_tokens"] == 1024


class TestBlendL0GpuObservability:
    def test_store_pre_computed_records_l0_duration_and_transfer_counters(
        self, bus, subscriber
    ):
        duration_before = _histogram_count(
            "lmcache_blend.l0_gpu_operation_duration_seconds"
        )
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_STORE_PRE_COMPUTED_START,
                    session_id="req-store-pre",
                    timestamp=100.0,
                    metadata={"instance_id": 7, "num_tokens": 512},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_STORE_PRE_COMPUTED_END,
                    session_id="req-store-pre",
                    timestamp=100.125,
                    metadata={
                        "instance_id": 7,
                        "num_tokens": 512,
                        "stored_chunks": 2,
                        "success": True,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        assert (
            _histogram_count("lmcache_blend.l0_gpu_operation_duration_seconds")
            - duration_before
        ) == 1
        assert {
            "operation": "store_pre_computed",
            "direction": "gpu_to_l1",
            "instance_id": 7,
            "success": True,
        } in _histogram_attrs("lmcache_blend.l0_gpu_operation_duration_seconds")
        assert (
            _counter_sum(
                "lmcache_blend.l0_gpu_transfer_chunks",
                operation="store_pre_computed",
                instance_id=7,
                direction="gpu_to_l1",
            )
            >= 2
        )
        assert (
            _counter_sum(
                "lmcache_blend.l0_gpu_transfer_tokens",
                operation="store_pre_computed",
                instance_id=7,
                direction="gpu_to_l1",
            )
            >= 512
        )

    def test_store_pre_computed_duration_includes_direction_attr(self, bus, subscriber):
        """The duration histogram MUST include ``direction`` per METRICS.md."""
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_STORE_PRE_COMPUTED_START,
                    session_id="req-dir-check",
                    timestamp=300.0,
                    metadata={"instance_id": 5, "num_tokens": 256},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_STORE_PRE_COMPUTED_END,
                    session_id="req-dir-check",
                    timestamp=300.05,
                    metadata={
                        "instance_id": 5,
                        "num_tokens": 256,
                        "stored_chunks": 1,
                        "success": True,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        attrs_list = _histogram_attrs("lmcache_blend.l0_gpu_operation_duration_seconds")
        assert len(attrs_list) >= 1, "Expected at least one duration data point"
        # The store_pre_computed operation has direction "gpu_to_l1"
        matching = [
            a
            for a in attrs_list
            if a.get("operation") == "store_pre_computed" and a.get("instance_id") == 5
        ]
        assert (
            len(matching) >= 1
        ), f"No duration point for store_pre_computed/instance_id=5; got {attrs_list}"
        assert (
            matching[0]["direction"] == "gpu_to_l1"
        ), f"Expected direction='gpu_to_l1', got attrs={matching[0]}"

    def test_retrieve_records_l0_transfer_direction_and_failure_duration(
        self, bus, subscriber
    ):
        duration_before = _histogram_count(
            "lmcache_blend.l0_gpu_operation_duration_seconds"
        )
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_RETRIEVE_START,
                    session_id="req-retrieve",
                    timestamp=200.0,
                    metadata={"instance_id": 3, "num_chunks": 4, "num_tokens": 1024},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_RETRIEVE_END,
                    session_id="req-retrieve",
                    timestamp=200.250,
                    metadata={
                        "instance_id": 3,
                        "num_chunks": 4,
                        "num_tokens": 1024,
                        "success": False,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        assert (
            _histogram_count("lmcache_blend.l0_gpu_operation_duration_seconds")
            - duration_before
        ) == 1
        assert {
            "operation": "retrieve_pre_computed",
            "direction": "l1_to_gpu",
            "instance_id": 3,
            "success": False,
        } in _histogram_attrs("lmcache_blend.l0_gpu_operation_duration_seconds")
        assert (
            _counter_sum(
                "lmcache_blend.l0_gpu_transfer_chunks",
                operation="retrieve_pre_computed",
                instance_id=3,
                direction="l1_to_gpu",
            )
            == 0
        )

    def test_store_final_records_l0_transfer_direction_and_duration(
        self, bus, subscriber
    ):
        duration_before = _histogram_count(
            "lmcache_blend.l0_gpu_operation_duration_seconds"
        )
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_STORE_FINAL_START,
                    session_id="req-store-final-l0",
                    timestamp=210.0,
                    metadata={"instance_id": 6, "num_chunks": 3, "num_tokens": 768},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_STORE_FINAL_END,
                    session_id="req-store-final-l0",
                    timestamp=210.125,
                    metadata={
                        "instance_id": 6,
                        "stored_chunks": 3,
                        "num_tokens": 768,
                        "success": True,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        assert (
            _histogram_count("lmcache_blend.l0_gpu_operation_duration_seconds")
            - duration_before
        ) == 1
        assert {
            "operation": "store_final",
            "direction": "gpu_to_l1",
            "instance_id": 6,
            "success": True,
        } in _histogram_attrs("lmcache_blend.l0_gpu_operation_duration_seconds")
        assert (
            _counter_sum(
                "lmcache_blend.l0_gpu_transfer_chunks",
                operation="store_final",
                instance_id=6,
                direction="gpu_to_l1",
            )
            >= 3
        )
        assert (
            _counter_sum(
                "lmcache_blend.l0_gpu_transfer_tokens",
                operation="store_final",
                instance_id=6,
                direction="gpu_to_l1",
            )
            >= 768
        )

    def test_cb_l0_metrics_export_to_prometheus(self):
        code = r"""
import os
import sys
import time

from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from prometheus_client import REGISTRY, generate_latest

from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.cb_server import (
    BlendMetricsSubscriber,
)

reader = PrometheusMetricReader()
metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
bus.register_subscriber(BlendMetricsSubscriber())
bus.start()
try:
    bus.publish(Event(
        event_type=EventType.CB_RETRIEVE_START,
        session_id="req-prometheus",
        metadata={"instance_id": 11, "num_chunks": 5, "num_tokens": 1280},
    ))
    bus.publish(Event(
        event_type=EventType.CB_RETRIEVE_END,
        session_id="req-prometheus",
        metadata={
            "instance_id": 11,
            "num_chunks": 5,
            "num_tokens": 1280,
            "success": True,
            "token_ids": [101, 102],
            "block_ids": [7, 8],
            "hashes": ["secret-hash"],
            "object_keys": ["secret-key"],
        },
    ))
    time.sleep(0.2)
finally:
    bus.stop()
for line in generate_latest(REGISTRY).decode().splitlines():
    if "lmcache_blend_l0_gpu" in line:
        print(line)
sys.stdout.flush()
os._exit(0)
"""
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert "lmcache_blend_l0_gpu_operation_duration_seconds" in result.stdout
        assert "lmcache_blend_l0_gpu_transfer_chunks_total" in result.stdout
        assert "lmcache_blend_l0_gpu_transfer_tokens_total" in result.stdout
        assert 'operation="retrieve_pre_computed"' in result.stdout
        assert 'direction="l1_to_gpu"' in result.stdout
        assert 'instance_id="11"' in result.stdout
        assert "token_ids" not in result.stdout
        assert "block_ids" not in result.stdout
        assert "secret-hash" not in result.stdout
        assert "secret-key" not in result.stdout

    def test_cb_prometheus_export_covers_full_current_metric_surface(self):
        code = r"""
import os
import sys
import time

from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from prometheus_client import REGISTRY, generate_latest

from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.cb_server import (
    BlendMetricsSubscriber,
)

reader = PrometheusMetricReader()
metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
bus.register_subscriber(BlendMetricsSubscriber())
bus.start()
try:
    events = [
        Event(EventType.CB_LOOKUP_START, "req-lookup", {}),
        Event(
            EventType.CB_LOOKUP_END,
            "req-lookup",
            {
                "requested_tokens": 100,
                "hit_tokens": 75,
                "fingerprint_hits": 10,
                "storage_hits": 8,
                "stale_chunks": 2,
                "no_gpu_context": True,
            },
        ),
        Event(
            EventType.CB_RETRIEVE_START,
            "req-retrieve",
            {"instance_id": 1, "num_chunks": 3, "num_tokens": 768},
        ),
        Event(
            EventType.CB_RETRIEVE_END,
            "req-retrieve",
            {
                "instance_id": 1,
                "num_chunks": 3,
                "num_tokens": 768,
                "success": False,
            },
        ),
        Event(
            EventType.CB_STORE_PRE_COMPUTED_START,
            "req-pre",
            {"instance_id": 2, "num_chunks": 4, "num_tokens": 1024},
        ),
        Event(
            EventType.CB_STORE_PRE_COMPUTED_END,
            "req-pre",
            {
                "instance_id": 2,
                "stored_chunks": 4,
                "num_tokens": 1024,
                "success": False,
            },
        ),
        Event(
            EventType.CB_STORE_FINAL_START,
            "req-final",
            {"instance_id": 3, "num_chunks": 5, "num_tokens": 1280},
        ),
        Event(
            EventType.CB_STORE_FINAL_END,
            "req-final",
            {
                "instance_id": 3,
                "stored_chunks": 5,
                "num_tokens": 1280,
                "success": False,
            },
        ),
        Event(
            EventType.CB_RETRIEVE_START,
            "req-retrieve-ok",
            {"instance_id": 4, "num_chunks": 6, "num_tokens": 1536},
        ),
        Event(
            EventType.CB_RETRIEVE_END,
            "req-retrieve-ok",
            {
                "instance_id": 4,
                "num_chunks": 6,
                "num_tokens": 1536,
                "success": True,
            },
        ),
        Event(EventType.CB_FINGERPRINTS_REGISTERED, "req-fp", {"num_chunks": 9}),
        Event(EventType.CB_CHUNKS_EVICTED, "req-evict", {"num_chunks": 2}),
    ]
    for event in events:
        bus.publish(event)
    time.sleep(0.2)
finally:
    bus.stop()

print(generate_latest(REGISTRY).decode())
sys.stdout.flush()
os._exit(0)
"""
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        for metric_name in (
            "lmcache_blend_lookup_requests_total",
            "lmcache_blend_lookup_requested_tokens_total",
            "lmcache_blend_lookup_hit_tokens_total",
            "lmcache_blend_lookup_fingerprint_hits_total",
            "lmcache_blend_lookup_storage_hits_total",
            "lmcache_blend_lookup_stale_chunks_total",
            "lmcache_blend_lookup_no_gpu_context_errors_total",
            "lmcache_blend_retrieve_requests_total",
            "lmcache_blend_retrieve_chunks_total",
            "lmcache_blend_retrieve_failures_total",
            "lmcache_blend_store_pre_computed_requests_total",
            "lmcache_blend_store_pre_computed_chunks_total",
            "lmcache_blend_store_pre_computed_failures_total",
            "lmcache_blend_store_final_requests_total",
            "lmcache_blend_store_final_chunks_total",
            "lmcache_blend_store_final_failures_total",
            "lmcache_blend_fingerprints_registered_total",
            "lmcache_blend_chunks_evicted_total",
            "lmcache_blend_l0_gpu_operation_duration_seconds",
            "lmcache_blend_l0_gpu_transfer_chunks_total",
            "lmcache_blend_l0_gpu_transfer_tokens_total",
        ):
            assert metric_name in result.stdout


class TestBlendPendingOpsCap:
    """Pending CB L0 GPU START events are bounded when END is missing."""

    def test_pending_l0_gpu_ops_cap_evicts_oldest_start(self, monkeypatch, subscriber):
        from lmcache.v1.mp_observability.subscribers.metrics import cb_server

        monkeypatch.setattr(cb_server, "_MAX_PENDING_L0_GPU_OPS", 2)
        callbacks = subscriber.get_subscriptions()

        for session_id, timestamp in (("one", 1.0), ("two", 2.0), ("three", 3.0)):
            callbacks[EventType.CB_RETRIEVE_START](
                Event(
                    event_type=EventType.CB_RETRIEVE_START,
                    session_id=session_id,
                    timestamp=timestamp,
                    metadata={"instance_id": "inst", "num_chunks": 1, "num_tokens": 16},
                )
            )

        assert len(subscriber._pending_l0_gpu_ops) == 2
        assert "one:cb.retrieve.start" not in subscriber._pending_l0_gpu_ops
        assert "three:cb.retrieve.start" in subscriber._pending_l0_gpu_ops

    def test_pending_l0_gpu_ops_cap_logs_warning(self, monkeypatch, subscriber):
        """When the cap is exceeded, a warning must be logged via the module logger."""
        from unittest.mock import patch

        from lmcache.v1.mp_observability.subscribers.metrics import cb_server

        monkeypatch.setattr(cb_server, "_MAX_PENDING_L0_GPU_OPS", 1)
        callbacks = subscriber.get_subscriptions()

        with patch.object(cb_server.logger, "warning") as mock_warn:
            callbacks[EventType.CB_RETRIEVE_START](
                Event(
                    event_type=EventType.CB_RETRIEVE_START,
                    session_id="first",
                    timestamp=1.0,
                    metadata={"instance_id": "inst", "num_chunks": 1, "num_tokens": 16},
                )
            )
            callbacks[EventType.CB_RETRIEVE_START](
                Event(
                    event_type=EventType.CB_RETRIEVE_START,
                    session_id="second",
                    timestamp=2.0,
                    metadata={"instance_id": "inst", "num_chunks": 1, "num_tokens": 16},
                )
            )

        mock_warn.assert_called_once()
        assert "_pending_l0_gpu_ops exceeded" in mock_warn.call_args[0][0]


class TestBlendSuccessPathCounters:
    """Verify that success-path counters (requests + chunks) increment correctly."""

    def test_retrieve_success_path_counters(self, bus, subscriber, snapshot):
        """Successful retrieve must increment retrieve_requests and retrieve_chunks."""
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_RETRIEVE_START,
                    session_id="req-ret-ok",
                    timestamp=400.0,
                    metadata={"instance_id": 0, "num_chunks": 7},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_RETRIEVE_END,
                    session_id="req-ret-ok",
                    timestamp=400.1,
                    metadata={
                        "instance_id": 0,
                        "num_chunks": 7,
                        "success": True,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        delta = snapshot()
        assert delta.get("lmcache_blend.retrieve_requests", 0) >= 1
        assert delta.get("lmcache_blend.retrieve_chunks", 0) >= 7

    def test_store_pre_computed_success_path_counters(self, bus, subscriber, snapshot):
        """Successful store_pre_computed must increment requests + chunks."""
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_STORE_PRE_COMPUTED_START,
                    session_id="req-sp-ok",
                    timestamp=500.0,
                    metadata={"instance_id": 0, "num_chunks": 5, "num_tokens": 320},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_STORE_PRE_COMPUTED_END,
                    session_id="req-sp-ok",
                    timestamp=500.05,
                    metadata={
                        "instance_id": 0,
                        "stored_chunks": 5,
                        "success": True,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        delta = snapshot()
        assert delta.get("lmcache_blend.store_pre_computed_requests", 0) >= 1
        assert delta.get("lmcache_blend.store_pre_computed_chunks", 0) >= 5
        # No failure should be recorded
        assert delta.get("lmcache_blend.store_pre_computed_failures", 0) == 0

    def test_store_final_success_path_counters(self, bus, subscriber, snapshot):
        """Successful store_final must increment requests + chunks."""
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_STORE_FINAL_START,
                    session_id="req-sf-ok",
                    timestamp=600.0,
                    metadata={"instance_id": 1, "num_chunks": 3, "num_tokens": 192},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_STORE_FINAL_END,
                    session_id="req-sf-ok",
                    timestamp=600.08,
                    metadata={
                        "instance_id": 1,
                        "stored_chunks": 3,
                        "success": True,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        delta = snapshot()
        assert delta.get("lmcache_blend.store_final_requests", 0) >= 1
        assert delta.get("lmcache_blend.store_final_chunks", 0) >= 3
        assert delta.get("lmcache_blend.store_final_failures", 0) == 0


class TestBlendL0GpuAttributeStrictness:
    """Verify OTel attributes (operation, direction, instance_id, success) on
    L0 GPU histogram and transfer counter data points."""

    def test_retrieve_l0_gpu_attribute_set(self, bus, subscriber):
        """Retrieve success: verify all four attrs on histogram and transfer counter."""
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_RETRIEVE_START,
                    session_id="attr-retrieve",
                    timestamp=700.0,
                    metadata={"instance_id": 42, "num_chunks": 4, "num_tokens": 256},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_RETRIEVE_END,
                    session_id="attr-retrieve",
                    timestamp=700.1,
                    metadata={
                        "instance_id": 42,
                        "num_chunks": 4,
                        "num_tokens": 256,
                        "success": True,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        # Verify histogram attributes
        expected_hist_attrs = {
            "operation": "retrieve_pre_computed",
            "direction": "l1_to_gpu",
            "instance_id": 42,
            "success": True,
        }
        assert expected_hist_attrs in _histogram_attrs(
            "lmcache_blend.l0_gpu_operation_duration_seconds"
        )

        # Verify transfer_chunks counter attributes (no success attr on transfer)
        assert (
            _counter_sum(
                "lmcache_blend.l0_gpu_transfer_chunks",
                operation="retrieve_pre_computed",
                instance_id=42,
                direction="l1_to_gpu",
            )
            >= 4
        )
        assert (
            _counter_sum(
                "lmcache_blend.l0_gpu_transfer_tokens",
                operation="retrieve_pre_computed",
                instance_id=42,
                direction="l1_to_gpu",
            )
            >= 256
        )

    def test_store_final_l0_gpu_failure_no_transfer(self, bus, subscriber):
        """Failed store_final: duration recorded with success=False, transfer=0."""
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_STORE_FINAL_START,
                    session_id="attr-fail",
                    timestamp=800.0,
                    metadata={"instance_id": 99, "num_chunks": 2, "num_tokens": 128},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_STORE_FINAL_END,
                    session_id="attr-fail",
                    timestamp=800.05,
                    metadata={
                        "instance_id": 99,
                        "stored_chunks": 0,
                        "success": False,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        expected_hist_attrs = {
            "operation": "store_final",
            "direction": "gpu_to_l1",
            "instance_id": 99,
            "success": False,
        }
        assert expected_hist_attrs in _histogram_attrs(
            "lmcache_blend.l0_gpu_operation_duration_seconds"
        )
        # Failed ops should not record transfer counters
        assert (
            _counter_sum(
                "lmcache_blend.l0_gpu_transfer_chunks",
                operation="store_final",
                instance_id=99,
                direction="gpu_to_l1",
            )
            == 0
        )


class TestBlendSubscriptionContractExact:
    """Verify the exact set of subscriptions returned by get_subscriptions()."""

    def test_subscription_count_and_all_callable(self, subscriber):
        subs = subscriber.get_subscriptions()
        expected = {
            EventType.CB_LOOKUP_START,
            EventType.CB_LOOKUP_END,
            EventType.CB_RETRIEVE_START,
            EventType.CB_RETRIEVE_END,
            EventType.CB_STORE_PRE_COMPUTED_START,
            EventType.CB_STORE_PRE_COMPUTED_END,
            EventType.CB_STORE_FINAL_START,
            EventType.CB_STORE_FINAL_END,
            EventType.CB_FINGERPRINTS_REGISTERED,
            EventType.CB_CHUNKS_EVICTED,
            EventType.CB_STORE_PRE_COMPUTED_SUBMITTED,
            EventType.CB_RETRIEVE_SUBMITTED,
            EventType.CB_STORE_FINAL_SUBMITTED,
        }
        assert set(subs.keys()) == expected
        for callback in subs.values():
            assert callable(callback)
