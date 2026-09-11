# SPDX-License-Identifier: Apache-2.0
"""Tests for node-local MP memory-pressure reporting."""

# Standard
from dataclasses import dataclass, replace

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.storage_usage import (
    StorageHealth,
    StorageTier,
    StorageTierUsageSnapshot,
    StorageUsageSnapshot,
)
from lmcache.v1.distributed.tiers import Tier
from lmcache.v1.multiprocess.memory_pressure import (
    MemoryPressureLevel,
    MemoryPressureService,
    MemoryPressureUnavailable,
    classify_memory_pressure,
)


def _usage(
    *,
    tier: StorageTier = Tier.L1,
    adapter_id: int | None = None,
    backend_type: str | None = None,
    used_bytes: int | None = 20,
    capacity_bytes: int | None = 100,
    trigger_watermark: float | None = 0.8,
    health: StorageHealth = StorageHealth.OK,
    collection_errors: tuple[str, ...] = (),
) -> StorageTierUsageSnapshot:
    return StorageTierUsageSnapshot(
        tier=tier,
        adapter_id=adapter_id,
        backend_type=backend_type,
        used_bytes=used_bytes,
        capacity_bytes=capacity_bytes,
        trigger_watermark=trigger_watermark,
        eviction_policy="LRU" if trigger_watermark is not None else None,
        health=health,
        collection_errors=collection_errors,
    )


@dataclass
class _UsageSource:
    value: StorageUsageSnapshot

    def get_memory_usage_snapshot(self) -> StorageUsageSnapshot:
        return self.value


class _BrokenUsageSource:
    def get_memory_usage_snapshot(self) -> StorageUsageSnapshot:
        raise RuntimeError("snapshot failed")


def test_classification_follows_configured_watermark() -> None:
    assert (
        classify_memory_pressure(
            used_bytes=80,
            capacity_bytes=100,
            trigger_watermark=0.8,
        )
        is MemoryPressureLevel.HIGH
    )
    assert (
        classify_memory_pressure(
            used_bytes=79,
            capacity_bytes=100,
            trigger_watermark=0.8,
        )
        is MemoryPressureLevel.NORMAL
    )


def test_classification_uses_capacity_for_critical_and_unknown() -> None:
    assert (
        classify_memory_pressure(
            used_bytes=100,
            capacity_bytes=100,
            trigger_watermark=0.8,
        )
        is MemoryPressureLevel.CRITICAL
    )
    assert (
        classify_memory_pressure(
            used_bytes=123,
            capacity_bytes=None,
            trigger_watermark=None,
        )
        is MemoryPressureLevel.UNKNOWN
    )


def test_report_preserves_each_l2_adapter_and_uses_hottest_level() -> None:
    source = _UsageSource(
        StorageUsageSnapshot(
            l1=_usage(used_bytes=20),
            l2=(
                _usage(
                    tier=Tier.L2,
                    adapter_id=0,
                    backend_type="dax",
                    used_bytes=90,
                ),
                _usage(
                    tier=Tier.L2,
                    adapter_id=1,
                    backend_type="fs",
                    used_bytes=10,
                ),
            ),
        )
    )

    report = MemoryPressureService(source, "mp-a", clock=lambda: 1.25).snapshot()

    assert report.instance_id == "mp-a"
    assert report.timestamp_ms == 1250
    assert report.overall_level is MemoryPressureLevel.HIGH
    assert report.complete is True
    assert [tier.adapter_id for tier in report.tiers] == [None, 0, 1]
    assert [tier.backend_type for tier in report.tiers[1:]] == ["dax", "fs"]
    assert report.tiers[1].used_ratio == pytest.approx(0.9)
    assert report.tiers[2].used_ratio == pytest.approx(0.1)


def test_unknown_capacity_keeps_usage_and_does_not_hide_known_pressure() -> None:
    source = _UsageSource(
        StorageUsageSnapshot(
            l1=_usage(used_bytes=80),
            l2=(
                _usage(
                    tier=Tier.L2,
                    adapter_id=4,
                    backend_type="remote",
                    used_bytes=123,
                    capacity_bytes=None,
                    trigger_watermark=None,
                ),
            ),
        )
    )

    report = MemoryPressureService(source, "mp-a", clock=lambda: 0).snapshot()

    assert report.overall_level is MemoryPressureLevel.HIGH
    assert report.complete is False
    assert report.tiers[1].used_bytes == 123
    assert report.tiers[1].capacity_bytes is None
    assert report.tiers[1].used_ratio is None
    assert report.tiers[1].level is MemoryPressureLevel.UNKNOWN


def test_health_is_reported_but_does_not_change_capacity_pressure() -> None:
    source = _UsageSource(
        StorageUsageSnapshot(
            l1=_usage(health=StorageHealth.FAILED),
            l2=(),
        )
    )

    report = MemoryPressureService(source, "mp-a", clock=lambda: 0).snapshot()

    assert report.tiers[0].health is StorageHealth.FAILED
    assert report.tiers[0].level is MemoryPressureLevel.NORMAL
    assert report.overall_level is MemoryPressureLevel.NORMAL


def test_isolated_l2_watermark_is_not_applied_to_aggregate_usage() -> None:
    isolated = _usage(
        tier=Tier.L2,
        adapter_id=0,
        used_bytes=90,
        trigger_watermark=0.8,
    )
    isolated = replace(isolated, eviction_policy="IsolatedLRU")
    source = _UsageSource(
        StorageUsageSnapshot(
            l1=_usage(used_bytes=20),
            l2=(isolated,),
        )
    )

    report = MemoryPressureService(source, "mp-a", clock=lambda: 0).snapshot()

    assert report.tiers[1].trigger_watermark == 0.8
    assert report.tiers[1].used_ratio == pytest.approx(0.9)
    assert report.tiers[1].level is MemoryPressureLevel.NORMAL


def test_top_level_source_failure_is_unavailable() -> None:
    service = MemoryPressureService(_BrokenUsageSource(), "mp-a")

    with pytest.raises(MemoryPressureUnavailable):
        service.snapshot()
