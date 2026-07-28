# SPDX-License-Identifier: Apache-2.0
"""DRAM Partition Coordinator for coordinated L1 + DramL2 sizing.

Splits a total DRAM pool between L1 (raw KV staging/hot cache) and
DramL2 (compressed cold store), validates the L1 staging minimum
required for concurrent prefetch operations, and exposes the computed
byte-level allocations for config construction.
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

    l1_staging_min_bytes: int = 0
    """Minimum L1 bytes required for staging (informational)."""


@dataclass
class StagingParams:
    """Parameters for computing L1 staging minimum.

    The L1 slab must hold at least::

        N_prefetch * 2.5 * chunk_size + N_write * chunk_size

    where:
    - N_prefetch: max concurrent prefetch requests
    - N_write: max concurrent write requests (typically 1 per vLLM instance)
    - chunk_size: size of one KV chunk in bytes
    - 2.5x multiplier: 1x for the write destination + 1.5x for the
      serde temp buffer (estimate_serialized_size upper bound per KVCacheClip)
    """

    chunk_size_bytes: int = 0
    """Size of one KV chunk (2 * num_layers * page_size * num_kv_heads *
    head_dim * element_size). 0 means skip staging validation."""

    max_prefetch_in_flight: int = 8
    """Maximum concurrent prefetch requests (from StorageManagerConfig)."""

    max_write_in_flight: int = 1
    """Maximum concurrent write requests."""


class DramPartitionCoordinator:
    """Coordinates L1 + DramL2 memory allocation from a unified budget.

    Usage::

        config = DramPartitionConfig(total_memory_budget_gb=16.0, l1_fraction=0.3)
        coordinator = DramPartitionCoordinator(config)
        alloc = coordinator.allocate(
            staging_params=StagingParams(chunk_size_bytes=5_242_880)
        )
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

    def allocate(self, staging_params: StagingParams | None = None) -> DramAllocation:
        """Compute the byte-level allocation.

        Args:
            staging_params: Optional staging parameters for L1 minimum
                validation. If None or chunk_size_bytes == 0, staging
                validation is skipped.

        Returns:
            DramAllocation with computed sizes.

        Raises:
            ValueError: If L1 allocation is below the staging minimum.
        """
        total_bytes = int(self._config.total_memory_budget_gb * (1 << 30))
        l1_bytes = int(total_bytes * self._config.l1_fraction)
        l2_bytes = total_bytes - l1_bytes  # remainder to avoid rounding loss

        staging_min = 0
        if staging_params and staging_params.chunk_size_bytes > 0:
            staging_min = self._compute_staging_min(staging_params)
            if l1_bytes < staging_min:
                raise ValueError(
                    f"L1 allocation ({l1_bytes / (1 << 30):.2f} GiB) is below "
                    f"the staging minimum ({staging_min / (1 << 30):.2f} GiB) "
                    f"required for {staging_params.max_prefetch_in_flight} "
                    f"concurrent prefetches. Increase total_memory_budget_gb "
                    f"or l1_fraction."
                )
            logger.info(
                "Memory budget: L1=%.2f GiB (staging_min=%.2f GiB), "
                "L2=%.2f GiB, total=%.2f GiB",
                l1_bytes / (1 << 30),
                staging_min / (1 << 30),
                l2_bytes / (1 << 30),
                total_bytes / (1 << 30),
            )

        return DramAllocation(
            l1_size_bytes=l1_bytes,
            l2_max_bytes=l2_bytes,
            l1_high_watermark=self._config.l1_high_watermark,
            l2_high_watermark=self._config.l2_high_watermark,
            l1_staging_min_bytes=staging_min,
        )

    @staticmethod
    def _compute_staging_min(params: StagingParams) -> int:
        """L1_min = N_prefetch * 2.5 * chunk_size + N_write * chunk_size"""
        prefetch_bytes = int(
            params.max_prefetch_in_flight * 2.5 * params.chunk_size_bytes
        )
        write_bytes = params.max_write_in_flight * params.chunk_size_bytes
        return prefetch_bytes + write_bytes
