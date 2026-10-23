# SPDX-License-Identifier: Apache-2.0
"""Tests for MP coordinator metrics initialization."""

# Standard
from unittest.mock import patch
import socket
import time

# Third Party
import pytest

# First Party
from lmcache.v1.cache_identity import ObjectKeyIdentity
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator import observability
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.background_loops import (
    BackgroundLoops,
    BackgroundLoopStats,
    EventLoopMonitor,
    EventLoopStats,
)
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.controllers.base import ControllerStatus
from lmcache.v1.mp_coordinator.ingest.event_broadcaster import CacheEventBroadcaster
from lmcache.v1.mp_coordinator.ingest.event_gate import EventGate
from lmcache.v1.mp_coordinator.ingest.kafka_event_source import (
    KafkaCacheEventSource,
    KafkaSourceStats,
)
from lmcache.v1.mp_coordinator.metric_sources import register_metric_source
from lmcache.v1.mp_coordinator.observability import init_coordinator_metrics
from lmcache.v1.mp_coordinator.persistence.quiesce import QuiesceLock
from lmcache.v1.mp_coordinator.views.instance_registry import (
    InstanceRegistry,
    MPInstance,
)
from lmcache.v1.mp_coordinator.views.key_directory import KeyDirectory

# Local
from .otel_reader import labels, private_meter, read_values


def _key(hash_byte: int) -> ObjectKey:
    return ObjectKey(
        identity=ObjectKeyIdentity(
            chunk_hash=bytes([hash_byte]) * 4, model_name="m", kv_rank=0
        )
    )


def _batch(
    tier: Tier,
    keys: list[ObjectKey],
    size_bytes: int,
    seq: int,
    backend: str = "dram",
) -> CacheEventBatch:
    return CacheEventBatch(
        instance_id="node-a",
        incarnation=1,
        seq=seq,
        event_type=CacheEventType.STORE,
        tier=tier,
        backend=backend,
        entries=[
            CacheEventEntry(key=k.to_encoded_object_key(), size_bytes=size_bytes)
            for k in keys
        ],
    )


def _stream_batch(
    instance_id: str, seq: int, dropped_events: int = 0, ts: float = 0.0
) -> CacheEventBatch:
    return CacheEventBatch(
        instance_id=instance_id,
        incarnation=1,
        seq=seq,
        event_type=CacheEventType.STORE,
        tier=Tier.L1,
        backend="dram",
        entries=[CacheEventEntry(key=_key(1).to_encoded_object_key(), size_bytes=1)],
        dropped_events=dropped_events,
        ts=ts,
    )


def _by_tier(
    values: dict[str, dict[frozenset[tuple[str, object]], float]],
) -> dict[str, dict[str, float]]:
    """Re-key :func:`read_values` output by each point's ``tier``."""
    return {
        name: {str(dict(attrs)["tier"]): value for attrs, value in points.items()}
        for name, points in values.items()
        if all("tier" in dict(attrs) for attrs in points)
    }


def test_disabled_metrics_are_not_initialized() -> None:
    config = MPCoordinatorConfig(metrics_enabled=False)

    with patch(
        "lmcache.v1.mp_coordinator.observability.init_otel_metrics"
    ) as mock_init:
        init_coordinator_metrics(config)

    mock_init.assert_not_called()


def test_prometheus_metrics_reuse_coordinator_http_server() -> None:
    config = MPCoordinatorConfig(metrics_enabled=True)

    with patch(
        "lmcache.v1.mp_coordinator.observability.init_otel_metrics"
    ) as mock_init:
        init_coordinator_metrics(config)

    mock_init.assert_called_once_with(
        otlp_endpoint=None,
        resource_attributes={
            "service.name": "lmcache-mp-coordinator",
            "service.instance.id": socket.gethostname(),
        },
        start_http_server=False,
    )


def test_otlp_metrics_reuse_shared_initializer() -> None:
    config = MPCoordinatorConfig(
        metrics_enabled=True,
        otlp_endpoint="http://collector:4317",
    )

    with patch(
        "lmcache.v1.mp_coordinator.observability.init_otel_metrics"
    ) as mock_init:
        init_coordinator_metrics(config)

    mock_init.assert_called_once_with(
        otlp_endpoint="http://collector:4317",
        resource_attributes={
            "service.name": "lmcache-mp-coordinator",
            "service.instance.id": socket.gethostname(),
        },
        start_http_server=False,
    )


def test_key_directory_gauges_report_the_registered_directory() -> None:
    directory = KeyDirectory()
    directory.consume(
        _batch(tier=Tier.L1, keys=[_key(1), _key(2)], size_bytes=100, seq=1)
    )
    directory.consume(
        _batch(tier=Tier.L2, backend="fs", keys=[_key(1)], size_bytes=400, seq=2)
    )
    meter, reader = private_meter()

    observability.register_coordinator_metrics(meter)

    points = _by_tier(read_values(reader))
    expected_placements = {"l1": 2, "l2": 1}
    expected_bytes = {"l1": 200, "l2": 400}
    assert points["lmcache_coordinator.key_directory.placements"] == expected_placements
    assert points["lmcache_coordinator.key_directory.placement_bytes"] == expected_bytes


def test_key_directory_gauges_keep_the_pre_rename_names_for_one_release() -> None:
    directory = KeyDirectory()
    directory.consume(_batch(tier=Tier.L1, keys=[_key(1)], size_bytes=100, seq=1))
    meter, reader = private_meter()

    observability.register_coordinator_metrics(meter)

    points = _by_tier(read_values(reader))
    assert points["lmcache_mp.key_directory_placement_count"] == {"l1": 1, "l2": 0}
    assert points["lmcache_mp.key_directory_placement_size_bytes"] == {
        "l1": 100,
        "l2": 0,
    }


def test_key_directory_gauges_follow_later_changes() -> None:
    directory = KeyDirectory()
    meter, reader = private_meter()
    observability.register_coordinator_metrics(meter)

    directory.consume(_batch(tier=Tier.L1, keys=[_key(1)], size_bytes=100, seq=1))

    points = _by_tier(read_values(reader))
    assert points["lmcache_coordinator.key_directory.placements"] == {"l1": 1, "l2": 0}


def test_event_gate_gauges_report_each_tracked_server() -> None:
    gate = EventGate(CacheEventBroadcaster(), QuiesceLock())
    gate.ingest(_stream_batch("node-a", seq=1))
    gate.ingest(_stream_batch("node-a", seq=4, dropped_events=6))
    gate.ingest(_stream_batch("node-b", seq=1))
    meter, reader = private_meter()

    observability.register_coordinator_metrics(meter)

    values = read_values(reader)
    assert values["lmcache_coordinator.ingest.server_event_batches_missing"] == {
        labels(instance_id="node-a"): 2,
        labels(instance_id="node-b"): 0,
    }
    assert values["lmcache_coordinator.ingest.server_events_dropped"] == {
        labels(instance_id="node-a"): 6,
        labels(instance_id="node-b"): 0,
    }
    assert values["lmcache_coordinator.ingest.server_view_incomplete"] == {
        labels(instance_id="node-a"): 1,
        labels(instance_id="node-b"): 0,
    }


def test_event_gate_gauges_drop_a_server_that_leaves() -> None:
    gate = EventGate(CacheEventBroadcaster(), QuiesceLock())
    gate.ingest(_stream_batch("node-a", seq=1))
    gate.ingest(_stream_batch("node-b", seq=1))
    meter, reader = private_meter()
    observability.register_coordinator_metrics(meter)

    gate.drop_instance("node-a")

    values = read_values(reader)
    assert values["lmcache_coordinator.ingest.server_view_incomplete"] == {
        labels(instance_id="node-b"): 0
    }


def test_event_gate_counters_report_the_fleet_totals() -> None:
    gate = EventGate(CacheEventBroadcaster(), QuiesceLock())
    gate.ingest(_stream_batch("node-a", seq=1))
    gate.ingest(_stream_batch("node-a", seq=4, dropped_events=6))
    gate.ingest(_stream_batch("node-a", seq=4))  # duplicate
    meter, reader = private_meter()

    observability.register_coordinator_metrics(meter)

    values = read_values(reader)
    assert values["lmcache_coordinator.ingest.event_batches_received"] == {
        labels(result="applied"): 2,
        labels(result="duplicate"): 1,
        labels(result="stale"): 0,
    }
    assert values["lmcache_coordinator.ingest.event_batches_missing"] == {labels(): 2}
    assert values["lmcache_coordinator.ingest.events_dropped_by_servers"] == {
        labels(): 6
    }


class _FailingConsumer:
    def consume(self, batch: CacheEventBatch) -> None:
        raise RuntimeError("consume bug")

    def fence_instance(self, instance_id: str) -> None:
        raise RuntimeError("fence bug")


def test_broadcaster_counters_report_each_consumer() -> None:
    broadcaster = CacheEventBroadcaster()
    broadcaster.register_consumer(_FailingConsumer())
    broadcaster.register_consumer(KeyDirectory())
    broadcaster.broadcast(_stream_batch("node-a", seq=1))
    broadcaster.fence_instance("node-a")
    meter, reader = private_meter()

    observability.register_coordinator_metrics(meter)

    values = read_values(reader)
    assert values["lmcache_coordinator.ingest.batches_delivered"] == {
        labels(consumer="_FailingConsumer"): 1,
        labels(consumer="KeyDirectory"): 1,
    }
    assert set(values["lmcache_coordinator.ingest.batch_apply_time_seconds"]) == {
        labels(consumer="_FailingConsumer"),
        labels(consumer="KeyDirectory"),
    }
    assert values["lmcache_coordinator.ingest.batch_apply_failures"] == {
        labels(consumer="_FailingConsumer", op="consume"): 1,
        labels(consumer="_FailingConsumer", op="fence"): 1,
        labels(consumer="KeyDirectory", op="consume"): 0,
        labels(consumer="KeyDirectory", op="fence"): 0,
    }


def test_event_gate_freshness_gauges_report_each_server() -> None:
    gate = EventGate(CacheEventBroadcaster(), QuiesceLock())
    gate.ingest(_stream_batch("node-a", seq=1, ts=time.time() - 5.0))
    gate.ingest(_stream_batch("node-b", seq=1))  # no send time
    meter, reader = private_meter()

    observability.register_coordinator_metrics(meter)

    values = read_values(reader)
    since = values["lmcache_coordinator.ingest.server_seconds_since_last_event"]
    assert set(since) == {labels(instance_id="node-a"), labels(instance_id="node-b")}
    delay = values["lmcache_coordinator.ingest.server_event_delivery_delay_seconds"]
    assert set(delay) == {labels(instance_id="node-a")}
    assert 5.0 <= delay[labels(instance_id="node-a")] < 10.0


class _StubKafkaSource(KafkaCacheEventSource):
    def __init__(self, stats: KafkaSourceStats) -> None:
        self.current = stats
        register_metric_source(self)

    def stats(self) -> KafkaSourceStats:
        return self.current


def test_kafka_source_metrics_report_reads_partitions_and_lag() -> None:
    _StubKafkaSource(
        KafkaSourceStats(
            records_applied=7,
            records_undecodable=1,
            records_failed=2,
            partitions_assigned=2,
            unread_records={0: 0, 1: 40},
        )
    )
    meter, reader = private_meter()

    observability.register_coordinator_metrics(meter)

    values = read_values(reader)
    assert values["lmcache_coordinator.kafka.records_read"] == {
        labels(result="applied"): 7,
        labels(result="undecodable"): 1,
        labels(result="failed"): 2,
    }
    assert values["lmcache_coordinator.kafka.partitions_assigned"] == {labels(): 2}
    assert values["lmcache_coordinator.kafka.unread_records"] == {
        labels(partition="0"): 0,
        labels(partition="1"): 40,
    }


class _StubLoops(BackgroundLoops):
    def __init__(self, stats: dict[str, BackgroundLoopStats]) -> None:
        self.current = stats
        register_metric_source(self)

    def stats(self) -> dict[str, BackgroundLoopStats]:
        return self.current


def test_background_loop_metrics_report_each_loop() -> None:
    _StubLoops(
        {
            "health": BackgroundLoopStats(
                interval_seconds=10.0,
                last_success_timestamp=1700000000.0,
                passes_ok=5,
                passes_failed=1,
            ),
            "eviction": BackgroundLoopStats(interval_seconds=5.0, passes_failed=2),
        }
    )
    meter, reader = private_meter()

    observability.register_coordinator_metrics(meter)

    values = read_values(reader)
    prefix = "lmcache_coordinator.background_loop."
    # A loop that never succeeded reports no last-success time.
    assert values[prefix + "last_success_timestamp_seconds"] == {
        labels(loop="health"): 1700000000.0
    }
    assert values[prefix + "interval_seconds"] == {
        labels(loop="health"): 10.0,
        labels(loop="eviction"): 5.0,
    }
    assert values[prefix + "passes"] == {
        labels(loop="health", result="ok"): 5,
        labels(loop="health", result="failed"): 1,
        labels(loop="eviction", result="ok"): 0,
        labels(loop="eviction", result="failed"): 2,
    }


def test_controller_metric_reports_each_controller_s_state() -> None:
    status = ControllerStatus(["FleetEvictionController", "Broken"])
    meter, reader = private_meter()
    observability.register_coordinator_metrics(meter)

    status.mark_up("FleetEvictionController")

    assert read_values(reader)["lmcache_coordinator.controller.up"] == {
        labels(controller="FleetEvictionController"): 1,
        labels(controller="Broken"): 0,
    }


class _StubMonitor(EventLoopMonitor):
    def stats(self) -> EventLoopStats:
        return EventLoopStats(blocked_seconds=1.5)


def test_event_loop_metric_reports_the_blocked_time() -> None:
    _StubMonitor()
    meter, reader = private_meter()

    observability.register_coordinator_metrics(meter)

    assert read_values(reader)["lmcache_coordinator.event_loop.blocked_seconds"] == {
        labels(): 1.5
    }


def test_registry_metric_reports_each_server_s_heartbeat_age() -> None:
    registry = InstanceRegistry()
    registry.register(
        MPInstance(
            instance_id="mp-0",
            ip="10.0.0.1",
            http_port=8000,
            registration_time=time.time(),
            last_heartbeat_time=time.monotonic() - 7.0,
        )
    )
    meter, reader = private_meter()

    observability.register_coordinator_metrics(meter)

    ages = read_values(reader)[
        "lmcache_coordinator.registry.server_seconds_since_heartbeat"
    ]
    assert set(ages) == {labels(instance_id="mp-0")}
    assert 7.0 <= ages[labels(instance_id="mp-0")] < 10.0


def test_a_component_s_metrics_are_registered_once() -> None:
    KeyDirectory()
    first_meter, first_reader = private_meter()
    observability.register_coordinator_metrics(first_meter)
    second_meter, second_reader = private_meter()

    observability.register_coordinator_metrics(second_meter)

    assert "lmcache_coordinator.key_directory.placements" in read_values(first_reader)
    assert read_values(second_reader) == {}


class _NoMetrics:
    def stats(self) -> None:
        return None


def test_a_registered_component_without_metrics_is_rejected() -> None:
    register_metric_source(_NoMetrics())
    meter, _reader = private_meter()

    with pytest.raises(TypeError, match="_NoMetrics"):
        observability.register_coordinator_metrics(meter)
