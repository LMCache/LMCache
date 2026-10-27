# SPDX-License-Identifier: Apache-2.0
# Third Party
import torch

# First Party
from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
    PagedTensorMemoryAllocator,
)
from lmcache.v1.memory_management import MemoryFormat


def _make_allocator(
    num_pages: int = 4,
) -> tuple[PagedTensorMemoryAllocator, torch.Size, torch.dtype]:
    """Create a paged allocator backed by CPU memory.

    Args:
        num_pages: Number of physical pages in the buffer.

    Returns:
        The allocator, allocation shape, and allocation dtype.
    """
    shape = torch.Size([16, 512])
    dtype = torch.float16
    page_bytes = shape.numel() * dtype.itemsize
    buffer = torch.empty(num_pages * page_bytes, dtype=torch.uint8)
    allocator = PagedTensorMemoryAllocator(
        buffer,
        [shape],
        [dtype],
        MemoryFormat.KV_2LTD,
    )
    return allocator, shape, dtype


def test_duplicate_ref_count_down_does_not_alias_pages() -> None:
    """A duplicate release must not enqueue the same page twice."""
    allocator, shape, dtype = _make_allocator()
    obj = allocator.allocate(shape, dtype)
    assert obj is not None

    obj.ref_count_down()
    obj.ref_count_down()

    live = []
    for _ in range(4):
        allocated = allocator.allocate(shape, dtype)
        assert allocated is not None
        live.append(allocated)

    assert allocator.allocate(shape, dtype) is None
    assert len({o.data_ptr for o in live}) == len(live)


def test_duplicate_direct_free_does_not_grow_free_pool() -> None:
    """Calling free twice must not increase the free pool."""
    allocator, shape, dtype = _make_allocator()
    obj = allocator.allocate(shape, dtype)
    assert obj is not None
    assert len(allocator.free_blocks) == 3

    allocator.free(obj)
    assert len(allocator.free_blocks) == 4
    allocator.free(obj)
    assert len(allocator.free_blocks) == 4


def test_reallocated_page_restores_state() -> None:
    """A recycled page must be valid with a clean pin state."""
    allocator, shape, dtype = _make_allocator(num_pages=2)
    obj = allocator.allocate(shape, dtype)
    assert obj is not None
    obj.ref_count_down()
    assert not obj.is_valid()

    # The free deque is FIFO: consume the other page, then recycle obj's page.
    assert allocator.allocate(shape, dtype) is not None
    reused = allocator.allocate(shape, dtype)
    assert reused is obj
    assert reused.is_valid()
    assert reused.get_ref_count() == 1
    assert reused.meta.pin_count == 0
    assert reused.meta.cached_positions is None
