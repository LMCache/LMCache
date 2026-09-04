# SPDX-License-Identifier: Apache-2.0
"""Local memory-pressure snapshots for an MP cache server.

The coordinator owns fleet metadata and control decisions, while each MP server
owns local memory safety and status collection.  This module turns the storage
manager's transport-neutral usage snapshot into a small, stable status model;
the HTTP layer only exposes the result.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol
import time

# First Party
from lmcache.v1.distributed.storage_usage import (
    StorageHealth,
    StorageTier,
    StorageTierUsageSnapshot,
    StorageUsageSnapshot,
)
from lmcache.v1.distributed.tiers import Tier


class MemoryPressureLevel(str, Enum):
    """Capacity pressure relative to the active local eviction policy."""

    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MemoryTierPressure:
    """Pressure and collection state for one local storage tier."""

    tier: StorageTier
    adapter_id: int | None
    backend_type: str | None
    used_bytes: int | None
    capacity_bytes: int | None
    used_ratio: float | None
    trigger_watermark: float | None
    eviction_policy: str | None
    health: StorageHealth
    level: MemoryPressureLevel
    collection_errors: tuple[str, ...]


@dataclass(frozen=True)
class InstanceMemoryPressureReport:
    """Point-in-time memory-pressure report for one MP server."""

    instance_id: str
    timestamp_ms: int
    overall_level: MemoryPressureLevel
    complete: bool
    tiers: tuple[MemoryTierPressure, ...]


class MemoryPressureUnavailable(RuntimeError):
    """Raised when the storage manager cannot produce any source snapshot."""


class _StorageUsageSource(Protocol):
    def get_memory_usage_snapshot(self) -> StorageUsageSnapshot:
        """Return current L1 and per-adapter L2 usage."""


class MemoryPressureService:
    """Build current pressure reports from a node-local storage manager."""

    def __init__(
        self,
        storage_manager: _StorageUsageSource,
        instance_id: str,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Initialize the service.

        Args:
            storage_manager: Source of canonical local storage usage snapshots.
            instance_id: Coordinator/telemetry identity for this MP server.
            clock: Wall-clock provider, injectable for deterministic tests.
        """
        self._storage_manager = storage_manager
        self._instance_id = instance_id
        self._clock = clock

    def snapshot(self) -> InstanceMemoryPressureReport:
        """Return one live local pressure report.

        Returns:
            A report with one L1 entry and one entry per active L2 adapter.

        Raises:
            MemoryPressureUnavailable: If the storage manager cannot produce
                its top-level snapshot. Individual tier failures remain in the
                report as ``unknown`` entries instead.
        """
        try:
            usage = self._storage_manager.get_memory_usage_snapshot()
        except Exception as exc:
            raise MemoryPressureUnavailable(
                "memory usage snapshot is unavailable"
            ) from exc

        tiers = tuple(
            _tier_pressure(tier_usage) for tier_usage in (usage.l1, *usage.l2)
        )
        known_levels = [
            tier.level
            for tier in tiers
            if tier.level is not MemoryPressureLevel.UNKNOWN
        ]
        overall_level = (
            max(known_levels, key=_pressure_rank)
            if known_levels
            else MemoryPressureLevel.UNKNOWN
        )
        return InstanceMemoryPressureReport(
            instance_id=self._instance_id,
            timestamp_ms=int(self._clock() * 1000),
            overall_level=overall_level,
            complete=all(
                tier.level is not MemoryPressureLevel.UNKNOWN for tier in tiers
            ),
            tiers=tiers,
        )


def classify_memory_pressure(
    *,
    used_bytes: int | None,
    capacity_bytes: int | None,
    trigger_watermark: float | None,
) -> MemoryPressureLevel:
    """Classify capacity pressure without inventing a second eviction policy.

    ``high`` begins at the tier's configured eviction watermark. ``critical``
    means usage has reached the tier's reported logical capacity. When
    capacity is known but no global watermark is configured, below-capacity
    usage is ``normal``; callers still receive the raw ratio to apply their own
    policy.

    Args:
        used_bytes: Current allocation, or ``None`` when collection failed.
        capacity_bytes: Reported logical capacity, or ``None`` for
            unknown/unlimited.
        trigger_watermark: Configured local eviction watermark, if any.

    Returns:
        The capacity-pressure level for this tier.
    """
    ratio = _usage_ratio(used_bytes, capacity_bytes)
    if ratio is None:
        return MemoryPressureLevel.UNKNOWN
    if ratio >= 1.0:
        return MemoryPressureLevel.CRITICAL
    if trigger_watermark is not None and ratio >= trigger_watermark:
        return MemoryPressureLevel.HIGH
    return MemoryPressureLevel.NORMAL


def _tier_pressure(usage: StorageTierUsageSnapshot) -> MemoryTierPressure:
    ratio = _usage_ratio(usage.used_bytes, usage.capacity_bytes)
    classification_watermark = usage.trigger_watermark
    if usage.tier is Tier.L2 and usage.eviction_policy == "IsolatedLRU":
        # IsolatedLRU applies its watermark to each cache_salt quota, not to
        # aggregate adapter capacity. Per-salt state belongs to /quota.
        classification_watermark = None
    return MemoryTierPressure(
        tier=usage.tier,
        adapter_id=usage.adapter_id,
        backend_type=usage.backend_type,
        used_bytes=usage.used_bytes,
        capacity_bytes=usage.capacity_bytes,
        used_ratio=ratio,
        trigger_watermark=usage.trigger_watermark,
        eviction_policy=usage.eviction_policy,
        health=usage.health,
        level=classify_memory_pressure(
            used_bytes=usage.used_bytes,
            capacity_bytes=usage.capacity_bytes,
            trigger_watermark=classification_watermark,
        ),
        collection_errors=usage.collection_errors,
    )


def _usage_ratio(used_bytes: int | None, capacity_bytes: int | None) -> float | None:
    if (
        used_bytes is None
        or used_bytes < 0
        or capacity_bytes is None
        or capacity_bytes <= 0
    ):
        return None
    return used_bytes / capacity_bytes


def _pressure_rank(level: MemoryPressureLevel) -> int:
    return {
        MemoryPressureLevel.UNKNOWN: -1,
        MemoryPressureLevel.NORMAL: 0,
        MemoryPressureLevel.HIGH: 1,
        MemoryPressureLevel.CRITICAL: 2,
    }[level]


__all__ = [
    "InstanceMemoryPressureReport",
    "MemoryPressureLevel",
    "MemoryPressureService",
    "MemoryPressureUnavailable",
    "MemoryTierPressure",
    "classify_memory_pressure",
]
