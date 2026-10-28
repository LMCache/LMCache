# SPDX-License-Identifier: Apache-2.0
"""Tests for MP coordinator metrics initialization."""

# Standard
from unittest.mock import patch
import socket
import time

# Third Party
import pytest

# First Party
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
from lmcache.v1.mp_coordinator.controllers.prefetch_manager import PrefetchManager
from lmcache.v1.mp_coordinator.ingest.event_broadcaster import CacheEventBroadcaster
from lmcache.v1.mp_coordinator.ingest.event_gate import EventGate
from lmcache.v1.mp_coordinator.ingest.kafka_event_source import (
    KafkaCacheEventSource,
    KafkaSourceStats,
)
from lmcache.v1.mp_coordinator.metric_sources import register_metric_source
from lmcache.v1.mp_coordinator.observability import init_coordinator_metrics
from lmcache.v1.mp_coordinator.persistence.checkpoint import (
    CheckpointMonitor,
    CheckpointWrite,
)
from lmcache.v1.mp_coordinator.persistence.metadata import MetadataPersister
from lmcache.v1.mp_coordinator.persistence.quiesce import QuiesceLock
from lmcache.v1.mp_coordinator.persistence.store import NullArtifactStore
from lmcache.v1.mp_coordinator.views.instance_registry import (
    InstanceRegistry,
    MPInstance,
)
from lmcache.v1.mp_coordinator.views.key_directory import KeyDirectory
from lmcache.v1.mp_coordinator.views.usage_manager import CacheUsageManager

# Local
from .otel_reader import labels, private_meter, read_values


def _key(hash_byte: int) -> ObjectKey:
    return ObjectKey(chunk_hash=bytes([hash_byte]) * 4, model_name="m", kv_rank=0)


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


def _mp_instance(instance_id: str) -> MPInstance:
    return MPInstance(
        instance_id=instance_id,
        ip="10.0.0.1",
        http_port=8000,
        registration_time=time.time(),
        last_heartbeat_time=time.monotonic(),
    )


def test_registry_metrics_report_membership_and_its_churn() -> None:
    registry = InstanceRegistry()
    registry.register(_mp_instance("mp-0"))
    registry.register(_mp_instance("mp-0"))
    registry.update_heartbeat("ghost", time.monotonic())
    meter, reader = private_meter()

    observability.register_coordinator_metrics(meter)

    values = read_values(reader)
    assert values["lmcache_coordinator.registry.servers_registered"] == {labels(): 1}
    assert values["lmcache_coordinator.registry.server_membership_changes"] == {
        labels(change="registered"): 1,
        labels(change="re_registered"): 1,
        labels(change="deregistered"): 0,
        labels(change="timed_out"): 0,
        labels(change="unknown_heartbeat"): 1,
    }


def test_key_directory_metrics_report_keys_and_each_server_s_l1_keys() -> None:
    directory = KeyDirectory()
    directory.consume(
        _batch(tier=Tier.L1, keys=[_key(1), _key(2)], size_bytes=100, seq=1)
    )
    meter, reader = private_meter()

    observability.register_coordinator_metrics(meter)

    values = read_values(reader)
    assert values["lmcache_coordinator.key_directory.keys_tracked"] == {labels(): 2}
    assert values["lmcache_coordinator.key_directory.server_l1_keys"] == {
        labels(instance_id="node-a"): 2
    }


def test_blend_index_metrics_report_only_while_blend_lookup_is_on() -> None:
    directory = KeyDirectory()
    meter, reader = private_meter()
    observability.register_coordinator_metrics(meter)
    names = (
        "lmcache_coordinator.blend_index.chunks_indexed",
        "lmcache_coordinator.blend_index.unique_chunk_contents",
        "lmcache_coordinator.blend_index.namespaces",
    )

    assert not set(names) & set(read_values(reader))

    directory.enable_blend_lookup(chunk_size=2, probe_stride=1)

    values = read_values(reader)
    assert {name: values[name] for name in names} == {
        name: {labels(): 0} for name in names
    }


def test_lookup_metrics_report_counts_and_the_found_per_request_histogram() -> None:
    directory = KeyDirectory()
    directory.record_lookup("prefix", chunks_requested=4, chunks_found=3)
    meter, reader = private_meter()

    observability.register_coordinator_metrics(meter)

    values = read_values(reader)
    prefix = "lmcache_coordinator.key_directory."
    assert values[prefix + "lookups"] == {
        labels(lookup_type="prefix"): 1,
        labels(lookup_type="non_prefix"): 0,
    }
    assert values[prefix + "lookup_chunks_requested"][labels(lookup_type="prefix")] == 4
    assert values[prefix + "lookup_chunks_found"][labels(lookup_type="prefix")] == 3
    histogram = prefix + "lookup_chunks_found_per_request"
    buckets = values[histogram + "_bucket"]
    assert buckets[labels(lookup_type="prefix", le="2")] == 0
    assert buckets[labels(lookup_type="prefix", le="4")] == 1
    assert buckets[labels(lookup_type="prefix", le="+Inf")] == 1
    assert values[histogram + "_sum"][labels(lookup_type="prefix")] == 3
    assert values[histogram + "_count"] == {
        labels(lookup_type="prefix"): 1,
        labels(lookup_type="non_prefix"): 0,
    }


def test_usage_metric_reports_the_fleet_bytes_per_tier() -> None:
    usage = CacheUsageManager()
    usage.consume(
        _batch(tier=Tier.L2, backend="fs", keys=[_key(1)], size_bytes=400, seq=1)
    )
    meter, reader = private_meter()

    observability.register_coordinator_metrics(meter)

    assert read_values(reader)["lmcache_coordinator.usage.bytes_stored"] == {
        labels(tier="l1"): 0,
        labels(tier="l2"): 400,
    }


def test_checkpoint_metrics_report_writes_phases_and_the_restore() -> None:
    monitor = CheckpointMonitor()
    monitor.record_restore("restored")
    monitor.record_write(
        CheckpointWrite(
            result="ok",
            pause_ingest_seconds=0.001,
            copy_seconds=0.002,
            write_seconds=0.003,
            size_bytes=4096,
        )
    )
    monitor.record_write(CheckpointWrite(result="quiesce_timeout", error="boom"))
    meter, reader = private_meter()

    observability.register_coordinator_metrics(meter)

    values = read_values(reader)
    prefix = "lmcache_coordinator.checkpoint_file."
    assert values[prefix + "writes"] == {
        labels(result="ok"): 1,
        labels(result="quiesce_timeout"): 1,
        labels(result="error"): 0,
    }
    assert values[prefix + "size_bytes"] == {labels(): 4096}
    assert set(values[prefix + "last_write_timestamp_seconds"]) == {labels()}
    assert values[prefix + "write_duration_seconds_count"] == {
        labels(phase="pause_ingest"): 1,
        labels(phase="copy"): 1,
        labels(phase="write"): 1,
    }
    assert values[prefix + "restored_on_start"] == {
        labels(outcome="cold"): 0,
        labels(outcome="restored"): 1,
        labels(outcome="ignored"): 0,
    }


def test_metadata_metric_reports_saves_by_result() -> None:
    MetadataPersister(NullArtifactStore()).save()
    meter, reader = private_meter()

    observability.register_coordinator_metrics(meter)

    assert read_values(reader)["lmcache_coordinator.metadata_file.writes"] == {
        labels(result="ok"): 1,
        labels(result="error"): 0,
    }


def test_prefetch_metrics_report_submits_and_polls() -> None:
    prefetch = PrefetchManager()
    prefetch.record_unknown_server()
    meter, reader = private_meter()

    observability.register_coordinator_metrics(meter)

    values = read_values(reader)
    assert values["lmcache_coordinator.prefetch.requests_submitted"] == {
        labels(result="submitted"): 0,
        labels(result="noop"): 0,
        labels(result="unknown_server"): 1,
        labels(result="server_error"): 0,
    }
    assert (
        values["lmcache_coordinator.prefetch.status_polls"][labels(result="pending")]
        == 0
    )
    assert values["lmcache_coordinator.prefetch.requests_abandoned"] == {labels(): 0}
    assert values["lmcache_coordinator.prefetch.submit_duration_seconds_count"] == {
        labels(): 0
    }
