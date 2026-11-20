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
from lmcache.v1 import memory_management
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    GdsL1Config,
    L1ManagerConfig,
    L1MemoryManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ManagerInterface
from lmcache.v1.distributed.l1_manager import create_l1_manager
from lmcache.v1.distributed.memory_manager import (
    GDSL1MemoryManager,
    L1MemoryManager,
    ShadowL1MemoryManager,
)
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
    PagedTensorMemoryAllocator,
)
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import (
    AddressManager,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)

# Importing this sets the process-wide MeterProvider with an
# InMemoryMetricReader; the L1 gauges bind their instruments to it.
from tests.v1.mp_observability.subscribers.metrics.otel_setup import reader as _reader
import lmcache.v1.distributed.memory_manager.l1_memory_manager as l1_memory_manager

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

    def get_memory_usage(self) -> tuple[int, int]:
        # Keeps the usage gauges' callbacks working during a scrape.
        return (0, _ARENA)

    def batched_free(
        self,
        memory_objs: list[MemoryObj],
        allocator_type: str | None = None,
        update_stats: bool = True,
    ) -> None:
        # BaseL1Manager.close() frees an empty batch through the manager;
        # freeing anything real would mean the test leaked an object.
        if memory_objs:
            raise AssertionError("batched_free is not exercised by these tests")


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


def test_empty_paged_allocator_reports_no_blocks() -> None:
    """torch.split leaves one empty slice for an empty buffer; it is not a page."""
    shapes, dtypes = [torch.Size([_ALIGN])], [torch.uint8]
    allocator = PagedTensorMemoryAllocator(
        torch.empty(0, dtype=torch.uint8), shapes, dtypes
    )
    assert allocator.get_fragmentation_stats() == (0, 0)
    assert allocator.address_manager.get_free_size() == 0


def test_paged_allocator_reports_page_granular_stats() -> None:
    """Same-size pages cannot fragment; free pages are one page each."""
    shapes, dtypes = [torch.Size([_ALIGN])], [torch.uint8]
    allocator = PagedTensorMemoryAllocator(
        torch.empty(3 * _ALIGN, dtype=torch.uint8), shapes, dtypes
    )
    assert allocator.get_fragmentation_stats() == (_ALIGN, 3)

    obj = allocator.allocate(shapes[0], dtypes[0])
    assert obj is not None
    assert allocator.get_fragmentation_stats() == (_ALIGN, 2)


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


def _gauge_snapshot() -> dict[tuple, float]:
    """Collect once: ``{(metric, sorted attrs): value}`` from one scrape."""
    data = _reader.get_metrics_data()
    result: dict[tuple, float] = {}
    if data is None:
        return result
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric_data in scope_metrics.metrics:
                for dp in metric_data.data.data_points:
                    # Skip non-numeric (e.g. histogram) data points.
                    if not hasattr(dp, "value"):
                        continue
                    attrs = tuple(sorted(dict(dp.attributes or {}).items()))
                    result[(metric_data.name, attrs)] = dp.value
    return result


def _dram_l1(allocator: MemoryAllocatorInterface) -> L1ManagerInterface:
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
        snapshot = _gauge_snapshot()
        attrs = tuple(sorted(_DRAM_ATTRS.items()))
        assert snapshot[("lmcache_mp.l1_largest_free_block_bytes", attrs)] == 123456
        assert snapshot[("lmcache_mp.l1_free_block_count", attrs)] == 7
    finally:
        manager.close()


def test_gauges_track_a_real_arena(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end: writes and deletes through a fully real DRAM L1.

    Only the pinned buffer's allocation and release are patched to plain
    CPU storage; the allocator factory, L1 manager and write/delete path
    are real. All four gauge values come from a single scrape.
    """
    unit = _ALIGN
    arena = 6 * unit
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda *args, **kwargs: torch.empty(arena, dtype=torch.uint8),
    )
    monkeypatch.setattr(
        memory_management, "_free_cpu_memory", lambda *args, **kwargs: None
    )
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=arena, use_lazy=False, align_bytes=unit, shm_name=""
        )
    )
    manager = create_l1_manager(config)
    try:
        keys = [
            ObjectKey(
                chunk_hash=ObjectKey.IntHash2Bytes(index),
                model_name="test_model",
                kv_rank=0,
            )
            for index in range(6)
        ]
        layout = MemoryLayoutDesc(shapes=[torch.Size([unit])], dtypes=[torch.uint8])
        results = manager.reserve_write(
            keys=keys, is_temporary=[False] * len(keys), layout_desc=layout
        )
        written = [key for key, (_, mem_obj) in results.items() if mem_obj is not None]
        assert len(written) == 6
        manager.finish_write(written)
        for key, error in manager.delete(keys[1::2]).items():
            assert error == L1Error.SUCCESS

        snapshot = _gauge_snapshot()
        attrs = tuple(sorted(_DRAM_ATTRS.items()))
        assert snapshot[("lmcache_mp.l1_memory_usage_bytes", attrs)] == 3 * unit
        assert snapshot[("lmcache_mp.l1_usage_ratio", attrs)] == 0.5
        assert snapshot[("lmcache_mp.l1_largest_free_block_bytes", attrs)] == unit
        assert snapshot[("lmcache_mp.l1_free_block_count", attrs)] == 3
    finally:
        manager.close()
