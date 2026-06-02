# SPDX-License-Identifier: Apache-2.0

"""Tests for EngineMetricsSubscriber."""

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
from lmcache.v1.mp_observability.subscribers.metrics.engine import (
    EngineMetricsSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.otel_setup import reader as _reader

_DRAIN_WAIT = 0.15
_BYTES_METRIC = "lmcache_mp.l0_l1_load_bytes"
_CHUNKS_METRIC = "lmcache_mp.num_chunks_loaded"
_REQUESTS_METRIC = "lmcache_mp.l0_l1_load_requests"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _retrieve_end(
    retrieved_count: int,
    total_bytes: int = 0,
    engine_id: int = 0,
    model_name: str = "test-model",
    cache_salt: str = "",
    device: str = "cuda:0",
) -> Event:
    return Event(
        event_type=EventType.MP_RETRIEVE_END,
        session_id="req-1",
        metadata={
            "retrieved_count": retrieved_count,
            "device": device,
            "engine_id": engine_id,
            "model_name": model_name,
            "cache_salt": cache_salt,
            "total_bytes": total_bytes,
        },
    )


def _attrs(
    worker_id: str, model_name: str = "test-model", cache_salt: str = ""
) -> tuple:
    """Build the sorted-tuple attribute key the subscriber emits."""
    return tuple(
        sorted(
            {
                "worker_id": worker_id,
                "model_name": model_name,
                "cache_salt": cache_salt,
            }.items()
        )
    )


def _load_attrs(
    worker_id: str,
    model_name: str = "test-model",
    cache_salt: str = "",
    device: str = "cuda:0",
) -> tuple:
    """Build the sorted-tuple attribute key for CPU-to-GPU load counters."""
    return tuple(
        sorted(
            {
                "worker_id": worker_id,
                "model_name": model_name,
                "cache_salt": cache_salt,
                "device": device,
            }.items()
        )
    )


def _read_counter_by_attrs(metric_name: str) -> dict[tuple, int]:
    data = _reader.get_metrics_data()
    result: dict[tuple, int] = {}
    if data is None:
        return result
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name != metric_name:
                    continue
                for dp in metric.data.data_points:
                    if not hasattr(dp, "value"):
                        continue
                    key = tuple(sorted(dict(dp.attributes).items()))
                    result[key] = int(dp.value)
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def subscriber():
    return EngineMetricsSubscriber()


# ---------------------------------------------------------------------------
# Subscription surface
# ---------------------------------------------------------------------------


class TestSubscriptions:
    def test_subscribes_to_retrieve_end_only(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.MP_RETRIEVE_END in subs
        # Store path is not of interest here; the counter is load-only.
        assert EventType.MP_STORE_END not in subs


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestNumChunksLoaded:
    def test_single_retrieve_adds_retrieved_count(self, subscriber):
        before = _read_counter_by_attrs(_CHUNKS_METRIC)
        subscriber._on_retrieve_end(_retrieve_end(retrieved_count=8, engine_id=3))
        after = _read_counter_by_attrs(_CHUNKS_METRIC)

        key = _attrs(worker_id="3")
        assert after.get(key, 0) == before.get(key, 0) + 8

    def test_different_workers_are_independent(self, subscriber):
        before = _read_counter_by_attrs(_CHUNKS_METRIC)
        subscriber._on_retrieve_end(_retrieve_end(retrieved_count=5, engine_id=0))
        subscriber._on_retrieve_end(_retrieve_end(retrieved_count=7, engine_id=1))
        subscriber._on_retrieve_end(_retrieve_end(retrieved_count=3, engine_id=0))
        after = _read_counter_by_attrs(_CHUNKS_METRIC)

        worker_0 = _attrs(worker_id="0")
        worker_1 = _attrs(worker_id="1")
        assert after.get(worker_0, 0) == before.get(worker_0, 0) + 8
        assert after.get(worker_1, 0) == before.get(worker_1, 0) + 7

    def test_carries_model_name_and_cache_salt(self, subscriber):
        before = _read_counter_by_attrs(_CHUNKS_METRIC)
        subscriber._on_retrieve_end(
            _retrieve_end(
                retrieved_count=4,
                engine_id=2,
                model_name="llama-3.1-8b",
                cache_salt="tenant-A",
            )
        )
        after = _read_counter_by_attrs(_CHUNKS_METRIC)
        key = _attrs(worker_id="2", model_name="llama-3.1-8b", cache_salt="tenant-A")
        assert after.get(key, 0) == before.get(key, 0) + 4

    def test_different_models_or_salts_accumulate_independently(self, subscriber):
        before = _read_counter_by_attrs(_CHUNKS_METRIC)
        # Same worker, different (model, salt) pairs.
        subscriber._on_retrieve_end(
            _retrieve_end(
                retrieved_count=5,
                engine_id=0,
                model_name="model-A",
                cache_salt="salt-1",
            )
        )
        subscriber._on_retrieve_end(
            _retrieve_end(
                retrieved_count=3,
                engine_id=0,
                model_name="model-A",
                cache_salt="salt-2",
            )
        )
        subscriber._on_retrieve_end(
            _retrieve_end(
                retrieved_count=7,
                engine_id=0,
                model_name="model-B",
                cache_salt="salt-1",
            )
        )
        after = _read_counter_by_attrs(_CHUNKS_METRIC)
        a1 = _attrs(worker_id="0", model_name="model-A", cache_salt="salt-1")
        a2 = _attrs(worker_id="0", model_name="model-A", cache_salt="salt-2")
        b1 = _attrs(worker_id="0", model_name="model-B", cache_salt="salt-1")
        assert after.get(a1, 0) == before.get(a1, 0) + 5
        assert after.get(a2, 0) == before.get(a2, 0) + 3
        assert after.get(b1, 0) == before.get(b1, 0) + 7


class TestL0L1LoadCounters:
    def test_retrieve_records_completed_request_and_bytes(self, subscriber):
        before_requests = _read_counter_by_attrs(_REQUESTS_METRIC)
        before_bytes = _read_counter_by_attrs(_BYTES_METRIC)

        subscriber._on_retrieve_end(
            _retrieve_end(retrieved_count=2, total_bytes=4096, engine_id=6)
        )

        after_requests = _read_counter_by_attrs(_REQUESTS_METRIC)
        after_bytes = _read_counter_by_attrs(_BYTES_METRIC)
        key = _load_attrs(worker_id="6")
        assert after_requests.get(key, 0) == before_requests.get(key, 0) + 1
        assert after_bytes.get(key, 0) == before_bytes.get(key, 0) + 4096

    def test_load_counters_carry_device_model_and_cache_salt(self, subscriber):
        before_requests = _read_counter_by_attrs(_REQUESTS_METRIC)
        before_bytes = _read_counter_by_attrs(_BYTES_METRIC)

        subscriber._on_retrieve_end(
            _retrieve_end(
                retrieved_count=1,
                total_bytes=2048,
                engine_id=4,
                model_name="llama-3.1-8b",
                cache_salt="tenant-A",
                device="cuda:3",
            )
        )

        after_requests = _read_counter_by_attrs(_REQUESTS_METRIC)
        after_bytes = _read_counter_by_attrs(_BYTES_METRIC)
        key = _load_attrs(
            worker_id="4",
            model_name="llama-3.1-8b",
            cache_salt="tenant-A",
            device="cuda:3",
        )
        assert after_requests.get(key, 0) == before_requests.get(key, 0) + 1
        assert after_bytes.get(key, 0) == before_bytes.get(key, 0) + 2048

    def test_zero_total_bytes_records_request_but_not_bytes(self, subscriber):
        before_requests = _read_counter_by_attrs(_REQUESTS_METRIC)
        before_bytes = _read_counter_by_attrs(_BYTES_METRIC)

        subscriber._on_retrieve_end(
            _retrieve_end(retrieved_count=3, total_bytes=0, engine_id=7)
        )

        after_requests = _read_counter_by_attrs(_REQUESTS_METRIC)
        after_bytes = _read_counter_by_attrs(_BYTES_METRIC)
        key = _load_attrs(worker_id="7")
        assert after_requests.get(key, 0) == before_requests.get(key, 0) + 1
        assert after_bytes.get(key, 0) == before_bytes.get(key, 0)

    def test_l0_l1_load_counters_export_to_prometheus(self):
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
from lmcache.v1.mp_observability.subscribers.metrics.engine import (
    EngineMetricsSubscriber,
)

reader = PrometheusMetricReader()
metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
bus.register_subscriber(EngineMetricsSubscriber())
bus.start()
bus.publish(Event(event_type=EventType.MP_RETRIEVE_END, metadata={
    "retrieved_count": 2,
    "total_bytes": 4096,
    "device": "cuda:0",
    "engine_id": 6,
    "model_name": "test-model",
    "cache_salt": "tenant-A",
}))
time.sleep(0.2)
bus.stop()
for line in generate_latest(REGISTRY).decode().splitlines():
    if "lmcache_mp_l0_l1_load" in line:
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

        assert "lmcache_mp_l0_l1_load_requests_total" in result.stdout
        assert "lmcache_mp_l0_l1_load_bytes_total" in result.stdout
        assert 'worker_id="6"' in result.stdout
        assert 'model_name="test-model"' in result.stdout
        assert 'cache_salt="tenant-A"' in result.stdout
        assert 'device="cuda:0"' in result.stdout


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_count_is_noop(self, subscriber):
        before_chunks = _read_counter_by_attrs(_CHUNKS_METRIC)
        before_requests = _read_counter_by_attrs(_REQUESTS_METRIC)
        before_bytes = _read_counter_by_attrs(_BYTES_METRIC)

        subscriber._on_retrieve_end(
            _retrieve_end(retrieved_count=0, total_bytes=1024, engine_id=4)
        )

        after_chunks = _read_counter_by_attrs(_CHUNKS_METRIC)
        after_requests = _read_counter_by_attrs(_REQUESTS_METRIC)
        after_bytes = _read_counter_by_attrs(_BYTES_METRIC)
        key = _attrs(worker_id="4")
        load_key = _load_attrs(worker_id="4")
        assert after_chunks.get(key, 0) == before_chunks.get(key, 0)
        assert after_requests.get(load_key, 0) == before_requests.get(load_key, 0)
        assert after_bytes.get(load_key, 0) == before_bytes.get(load_key, 0)

    def test_missing_engine_id_still_records_without_attr(self, subscriber):
        # Some future emission site may forget engine_id; we should
        # record the count anyway (so operators notice the total) but
        # drop the worker_id attribute.
        before = _read_counter_by_attrs(_CHUNKS_METRIC)
        subscriber._on_retrieve_end(
            Event(
                event_type=EventType.MP_RETRIEVE_END,
                metadata={"retrieved_count": 2},
            )
        )
        after = _read_counter_by_attrs(_CHUNKS_METRIC)
        empty_key: tuple = ()
        assert after.get(empty_key, 0) == before.get(empty_key, 0) + 2


# ---------------------------------------------------------------------------
# End-to-end via EventBus
# ---------------------------------------------------------------------------


class TestEventBusIntegration:
    def test_retrieve_end_via_bus_increments_counter(self):
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        sub = EngineMetricsSubscriber()
        bus.register_subscriber(sub)

        before = _read_counter_by_attrs(_CHUNKS_METRIC)
        bus.start()
        bus.publish(_retrieve_end(retrieved_count=12, engine_id=9))
        time.sleep(_DRAIN_WAIT)
        bus.stop()
        after = _read_counter_by_attrs(_CHUNKS_METRIC)

        key = _attrs(worker_id="9")
        assert after.get(key, 0) == before.get(key, 0) + 12
