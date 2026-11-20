# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the L1 fragmentation signal (issue #6724).

Covers the passthrough at every layer: ``AddressManager`` /
``TensorMemoryAllocator`` semantics (including zero-capacity arenas), the
``L1MemoryManager`` fast path and address-manager fallback, the GDS and
shadow managers, and the ``lmcache_mp`` gauges that surface the signal.
"""

# Standard
from unittest.mock import patch

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.config import (
    GdsL1Config,
    L1ManagerConfig,
    L1MemoryManagerConfig,
)
from lmcache.v1.distributed.l1_manager import create_l1_manager
from lmcache.v1.distributed.memory_manager import (
    GDSL1MemoryManager,
    L1MemoryManager,
    ShadowL1MemoryManager,
)
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import (
    AddressManager,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
import lmcache.v1.distributed.memory_manager.l1_memory_manager as l1_memory_manager

# Importing this sets the process-wide MeterProvider with an
# InMemoryMetricReader; the L1 gauges bind their instruments to it.
from tests.v1.mp_observability.subscribers.metrics.otel_setup import reader as _reader

_ALIGN = 4096
_ARENA = 16 * 1024


def _make_manager(allocator: MemoryAllocatorInterface) -> L1MemoryManager:
    config = L1MemoryManagerConfig(
        size_in_bytes=_ARENA,
        use_lazy=False,
        align_bytes=_ALIGN,
        shm_name="",
    )
    with patch.object(
        l1_memory_manager,
        "create_memory_allocator",
        return_value=allocator,
    ):
        return L1MemoryManager(config)


class _NonAllocatingAllocator(MemoryAllocatorInterface):
    """Concrete allocator whose allocation entry points must never run."""

    def allocate(
        self,
        shapes,
        dtypes,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: str | None = None,
    ) -> MemoryObj | None:
        raise AssertionError("allocate is not exercised by these tests")

    def batched_allocate(
        self,
        shapes,
        dtypes,
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: str | None = None,
    ) -> list[MemoryObj] | None:
        raise AssertionError("batched_allocate is not exercised by these tests")

    def free(self, memory_obj: MemoryObj, allocator_type: str | None = None) -> None:
        raise AssertionError("free is not exercised by these tests")

    def batched_free(
        self,
        memory_objs: list[MemoryObj],
        allocator_type: str | None = None,
        update_stats: bool = True,
    ) -> None:
        raise AssertionError("batched_free is not exercised by these tests")


class _StatsOnlyAllocator(_NonAllocatingAllocator):
    """Allocator exposing the stats directly (the hasattr fast path)."""

    def get_fragmentation_stats(self) -> tuple[int, int]:
        return (123456, 7)


# =============================================================================
# AddressManager / allocator semantics
# =============================================================================


def test_zero_capacity_address_manager_reports_no_blocks() -> None:
    """The zero-size placeholder block is not a free block."""
    assert AddressManager(0, _ALIGN).get_fragmentation_stats() == (0, 0)


def test_empty_tensor_allocator_reports_no_blocks() -> None:
    allocator = TensorMemoryAllocator(
        torch.empty(0, dtype=torch.uint8), align_bytes=_ALIGN
    )
    assert allocator.get_fragmentation_stats() == (0, 0)


def test_tensor_allocator_reports_real_fragmentation() -> None:
    """A real arena reports sub-arena holes after interleaved frees."""
    unit = _ALIGN
    tensor = torch.empty(6 * unit, dtype=torch.uint8)
    allocator = TensorMemoryAllocator(tensor, align_bytes=unit)
    assert allocator.get_fragmentation_stats() == (6 * unit, 1)

    objects = [allocator.allocate(torch.Size([unit]), torch.uint8) for _ in range(6)]
    assert objects[0] is not None
    assert allocator.get_fragmentation_stats() == (0, 0)

    for odd_object in objects[1::2]:
        assert odd_object is not None
        allocator.free(odd_object)
    assert allocator.get_fragmentation_stats() == (unit, 3)

    for even_object in objects[0::2]:
        assert even_object is not None
        allocator.free(even_object)
    assert allocator.get_fragmentation_stats() == (6 * unit, 1)


# =============================================================================
# L1MemoryManager passthrough
# =============================================================================


def test_manager_passthrough_reads_allocator_stats() -> None:
    """The fast path returns the allocator's own fragmentation stats."""
    manager = _make_manager(_StatsOnlyAllocator())
    assert manager.get_fragmentation_stats() == (123456, 7)


def test_manager_raises_for_unsupported_allocator() -> None:
    """No stats and no address manager is an explicit NotImplementedError."""
    manager = _make_manager(_NonAllocatingAllocator())
    with pytest.raises(NotImplementedError):
        manager.get_fragmentation_stats()


def test_manager_falls_back_to_address_manager_stats() -> None:
    """Lazy allocators read the stats through their address manager."""

    class _LazyWithArena(LazyMemoryAllocator):
        def __init__(self) -> None:
            self.address_manager = AddressManager(_ARENA, _ALIGN)

        def get_address_manager(self) -> AddressManager:
            return self.address_manager

        def close(self) -> None:
            pass

    manager = _make_manager(_LazyWithArena())
    assert manager.get_fragmentation_stats() == (_ARENA, 1)


# =============================================================================
# GDS and shadow managers
# =============================================================================


def test_gds_manager_reports_address_space_stats() -> None:
    """The GDS slab reports its AddressManager; no GDS hardware is needed."""
    manager = GDSL1MemoryManager(
        GdsL1Config(file_location="/unused", size_in_bytes=_ARENA)
    )
    try:
        assert manager.get_fragmentation_stats() == (_ARENA, 1)
    finally:
        manager.close()


def test_shadow_manager_reports_zeros() -> None:
    """A shadow L1 owns no bytes, so it reports the ``(0, 0)`` convention."""
    assert ShadowL1MemoryManager().get_fragmentation_stats() == (0, 0)


# =============================================================================
# Gauges
# =============================================================================


def _value_for(metric: str, attrs: dict | None = None) -> float:
    """Look up the current value of ``metric`` for the exact attribute set."""
    data = _reader.get_metrics_data()
    if data is None:
        return 0
    key = tuple(sorted((attrs or {}).items()))
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric_data in scope_metrics.metrics:
                for dp in metric_data.data.data_points:
                    if (
                        metric_data.name == metric
                        and tuple(sorted(dict(dp.attributes).items())) == key
                    ):
                        return dp.value
    return 0


def _dram_l1(allocator: MemoryAllocatorInterface):
    """Create a real ``DramL1Manager`` around ``allocator`` (no real buffer)."""
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=_ARENA,
            use_lazy=False,
            align_bytes=_ALIGN,
            shm_name="",
        )
    )
    with patch.object(
        l1_memory_manager,
        "create_memory_allocator",
        return_value=allocator,
    ):
        return create_l1_manager(config)


_DRAM_ATTRS = {"l1_tag": "_default", "backend": "dram"}


def test_gauges_report_manager_stats_with_l1_labels() -> None:
    """The gauges surface the manager's stats tagged by L1 tag and medium."""
    manager = _dram_l1(_StatsOnlyAllocator())
    try:
        assert (
            _value_for("lmcache_mp.l1_largest_free_block_bytes", _DRAM_ATTRS) == 123456
        )
        assert _value_for("lmcache_mp.l1_free_block_count", _DRAM_ATTRS) == 7
    finally:
        manager.close()


def test_gauges_track_a_real_arena() -> None:
    """End to end: interleaved frees on a real arena show up in the gauges."""
    unit = _ALIGN
    allocator = TensorMemoryAllocator(
        torch.empty(6 * unit, dtype=torch.uint8), align_bytes=unit
    )
    manager = _dram_l1(allocator)
    try:
        objects = [
            allocator.allocate(torch.Size([unit]), torch.uint8) for _ in range(6)
        ]
        assert objects[0] is not None
        for odd_object in objects[1::2]:
            assert odd_object is not None
            allocator.free(odd_object)
        assert _value_for("lmcache_mp.l1_largest_free_block_bytes", _DRAM_ATTRS) == unit
        assert _value_for("lmcache_mp.l1_free_block_count", _DRAM_ATTRS) == 3
        for even_object in objects[0::2]:
            assert even_object is not None
            allocator.free(even_object)
    finally:
        manager.close()
