# SPDX-License-Identifier: Apache-2.0
"""Tests for MP coordinator metrics initialization."""

# Standard
from unittest.mock import patch
import socket

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator import observability
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.observability import init_coordinator_metrics
from lmcache.v1.mp_coordinator.views.key_directory import KeyDirectory

# Local
from .otel_reader import private_meter, read_values


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

    observability.register_key_directory_metrics(directory, meter)

    points = _by_tier(read_values(reader))
    expected_placements = {"l1": 2, "l2": 1}
    expected_bytes = {"l1": 200, "l2": 400}
    assert points["lmcache_coordinator.key_directory.placements"] == expected_placements
    assert points["lmcache_coordinator.key_directory.placement_bytes"] == expected_bytes


def test_key_directory_gauges_keep_the_pre_rename_names_for_one_release() -> None:
    directory = KeyDirectory()
    directory.consume(_batch(tier=Tier.L1, keys=[_key(1)], size_bytes=100, seq=1))
    meter, reader = private_meter()

    observability.register_key_directory_metrics(directory, meter)

    points = _by_tier(read_values(reader))
    assert points["lmcache_mp.key_directory_placement_count"] == {"l1": 1, "l2": 0}
    assert points["lmcache_mp.key_directory_placement_size_bytes"] == {
        "l1": 100,
        "l2": 0,
    }


def test_key_directory_gauges_follow_later_changes() -> None:
    directory = KeyDirectory()
    meter, reader = private_meter()
    observability.register_key_directory_metrics(directory, meter)

    directory.consume(_batch(tier=Tier.L1, keys=[_key(1)], size_bytes=100, seq=1))

    points = _by_tier(read_values(reader))
    assert points["lmcache_coordinator.key_directory.placements"] == {"l1": 1, "l2": 0}
