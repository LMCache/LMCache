# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the L1 fragmentation-signal passthrough (issue #6724)."""

# Standard
from unittest.mock import patch

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.memory_manager import L1MemoryManager
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import (
    AddressManager,
    MemoryAllocatorInterface,
)
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


class _StatsOnlyAllocator(MemoryAllocatorInterface):
    """Allocator exposing the stats directly (the hasattr fast path)."""

    def get_fragmentation_stats(self) -> tuple[int, int]:
        return (123456, 7)

    def close(self) -> None:
        pass


class _NoStatsAddressManager:
    """Minimal address-manager double without fragmentation support."""

    def get_heap_size(self) -> int:
        return _ARENA

    def get_free_size(self) -> int:
        return _ARENA


class _StatslessLazyAllocator(LazyMemoryAllocator):
    """Lazy allocator whose address manager lacks the stats method."""

    def __init__(self) -> None:
        self.address_manager = _NoStatsAddressManager()

    def get_address_manager(self) -> _NoStatsAddressManager:
        return self.address_manager

    def close(self) -> None:
        pass


def test_manager_passthrough_reads_allocator_stats() -> None:
    """The fast path returns the allocator's own fragmentation stats."""
    manager = _make_manager(_StatsOnlyAllocator())
    assert manager.get_fragmentation_stats() == (123456, 7)


def test_manager_raises_for_unsupported_allocator() -> None:
    """No stats and no address manager is an explicit NotImplementedError."""

    class _Opaque(MemoryAllocatorInterface):
        def close(self) -> None:
            pass

    manager = _make_manager(_Opaque())
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


def test_tensor_allocator_reports_real_fragmentation() -> None:
    """A real arena reports sub-aligned holes after interleaved frees."""
    unit = 4096
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
