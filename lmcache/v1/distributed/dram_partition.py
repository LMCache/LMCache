# SPDX-License-Identifier: Apache-2.0
"""DRAM Partition Coordinator for coordinated L1 + DramL2 sizing.

Splits a total DRAM pool between L1 (raw KV staging/hot cache) and
DramL2 (compressed cold store), and exposes the computed byte-level
allocations for config construction.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DramPartitionConfig:
    """User-facing memory budget knobs.

    When ``total_memory_budget_gb`` is set (> 0), it overrides the
    separate ``--l1-size-gb`` and L2 adapter ``max_size_gb`` settings.
    The coordinator splits the budget using ``l1_fraction``.
    """

    total_memory_budget_gb: float = 0.0
    """Total DRAM budget in GiB. 0 means disabled (use manual sizing)."""

    l1_fraction: float = 0.3
    """Fraction of total budget allocated to L1 slab (raw KV hot cache).
    The remainder (1 - l1_fraction) goes to DramL2 compressed store."""

    l2_high_watermark: float = 0.8
    """Usage fraction that triggers L2 eviction (overrides per-adapter
    eviction_config.trigger_watermark when budget coordination is active)."""

    l1_high_watermark: float = 0.8
    """Usage fraction that triggers L1 eviction (overrides the global
    eviction_config.trigger_watermark when budget coordination is active)."""

    def __post_init__(self):
        if self.total_memory_budget_gb < 0:
            raise ValueError(
                f"total_memory_budget_gb must be >= 0, "
                f"got {self.total_memory_budget_gb}"
            )
        if not 0.0 < self.l1_fraction < 1.0:
            raise ValueError(f"l1_fraction must be in (0, 1), got {self.l1_fraction}")
        if not 0.0 < self.l2_high_watermark <= 1.0:
            raise ValueError(
                f"l2_high_watermark must be in (0, 1], got {self.l2_high_watermark}"
            )
        if not 0.0 < self.l1_high_watermark <= 1.0:
            raise ValueError(
                f"l1_high_watermark must be in (0, 1], got {self.l1_high_watermark}"
            )

    @property
    def enabled(self) -> bool:
        """Whether coordinated budget management is active."""
        return self.total_memory_budget_gb > 0


@dataclass(frozen=True)
class DramAllocation:
    """Computed byte-level allocation from a MemoryBudgetConfig.

    Immutable snapshot returned by :meth:`DramPartitionCoordinator.allocate`.
    """

    l1_size_bytes: int
    """Bytes allocated to the L1 slab (raw KV staging + hot cache)."""

    l2_max_bytes: int
    """Bytes allocated to DramL2 compressed store (max_capacity_bytes)."""

    l1_high_watermark: float
    """L1 eviction trigger watermark."""

    l2_high_watermark: float
    """L2 eviction trigger watermark."""


class DramPartitionCoordinator:
    """Coordinates L1 + DramL2 memory allocation from a unified budget.

    Usage::

        config = DramPartitionConfig(total_memory_budget_gb=16.0, l1_fraction=0.3)
        coordinator = DramPartitionCoordinator(config)
        alloc = coordinator.allocate()
        # alloc.l1_size_bytes → L1MemoryManagerConfig.size_in_bytes
        # alloc.l2_max_bytes → DramL2AdapterConfig.max_size_gb
    """

    def __init__(self, config: DramPartitionConfig):
        if not config.enabled:
            raise ValueError(
                "DramPartitionCoordinator requires total_memory_budget_gb > 0"
            )
        self._config = config

    @property
    def config(self) -> DramPartitionConfig:
        return self._config

    def allocate(self) -> DramAllocation:
        """Compute the byte-level allocation.

        Returns:
            DramAllocation with computed sizes.
        """
        total_bytes = int(self._config.total_memory_budget_gb * (1 << 30))
        l1_bytes = int(total_bytes * self._config.l1_fraction)
        l2_bytes = total_bytes - l1_bytes  # remainder to avoid rounding loss
        logger.info(
            "Memory budget: L1=%.2f GiB, L2=%.2f GiB, total=%.2f GiB",
            l1_bytes / (1 << 30),
            l2_bytes / (1 << 30),
            total_bytes / (1 << 30),
        )

        return DramAllocation(
            l1_size_bytes=l1_bytes,
            l2_max_bytes=l2_bytes,
            l1_high_watermark=self._config.l1_high_watermark,
            l2_high_watermark=self._config.l2_high_watermark,
        )
