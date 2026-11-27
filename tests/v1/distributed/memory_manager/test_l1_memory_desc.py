# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for L1 fixed-buffer registration descriptors."""

# Standard
from unittest.mock import patch

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.memory_manager import L1MemoryManager
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.memory_management import AddressManager, MemoryAllocatorInterface
import lmcache.v1.distributed.memory_manager.l1_memory_manager as l1_memory_manager

_ALIGN = 4096
_FINAL_SIZE = 16 * 1024


class _FakeAddressManager(AddressManager):
    def __init__(self, heap_size: int) -> None:
        self.heap_size = heap_size

    def get_heap_size(self) -> int:
        return self.heap_size


class _FakeMixedMemoryAllocator(MixedMemoryAllocator):
    def __init__(self, shm_name: str | None) -> None:
        self.buffer = torch.empty(_FINAL_SIZE, dtype=torch.uint8)
        self.shm_name = shm_name

    def close(self) -> None:
        pass


class _FakeLazyMemoryAllocator(LazyMemoryAllocator):
    def __init__(self, heap_size: int) -> None:
        self.buffer = torch.empty(_FINAL_SIZE, dtype=torch.uint8)
        self.address_manager = _FakeAddressManager(heap_size)

    def get_underlying_buffer(self) -> torch.Tensor:
        return self.buffer

    def get_address_manager(self) -> _FakeAddressManager:
        return self.address_manager

    def close(self) -> None:
        pass


def _make_manager(allocator: MemoryAllocatorInterface) -> L1MemoryManager:
    config = L1MemoryManagerConfig(
        size_in_bytes=_FINAL_SIZE,
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


def test_anonymous_mixed_allocator_exposes_full_stable_range() -> None:
    manager = _make_manager(_FakeMixedMemoryAllocator(shm_name=None))
    try:
        desc = manager.get_l1_memory_desc()

        assert desc.size == _FINAL_SIZE
        assert desc.stable_registration_size == _FINAL_SIZE
    finally:
        manager.close()


def test_shm_mixed_allocator_exposes_no_stable_range() -> None:
    manager = _make_manager(_FakeMixedMemoryAllocator(shm_name="lmcache_l1_pool_test"))
    try:
        desc = manager.get_l1_memory_desc()

        assert desc.size == _FINAL_SIZE
        assert desc.stable_registration_size is None
    finally:
        manager.close()


@pytest.mark.parametrize(
    ("heap_size", "expected_stable_size"),
    [
        (0, None),
        (_FINAL_SIZE // 2, _FINAL_SIZE // 2),
        (_FINAL_SIZE * 2, _FINAL_SIZE),
    ],
)
def test_lazy_allocator_exposes_bounded_heap_snapshot(
    heap_size: int,
    expected_stable_size: int | None,
) -> None:
    manager = _make_manager(_FakeLazyMemoryAllocator(heap_size))
    try:
        desc = manager.get_l1_memory_desc()

        assert desc.size == _FINAL_SIZE
        assert desc.stable_registration_size == expected_stable_size
    finally:
        manager.close()
