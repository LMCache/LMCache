# SPDX-License-Identifier: Apache-2.0
"""Request-exclusive allocation contracts."""

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.server_bench.allocator import BlockAllocator


def test_sequential_allocation_and_release() -> None:
    allocator = BlockAllocator(2, 8)
    first = allocator.allocate(2)
    assert first.as_block_ids() == [[1, 2], [1, 2]]
    assert [group.group_id for group in first.groups] == [0, 1]
    assert allocator.allocate(2).as_block_ids() == [[3, 4], [3, 4]]
    allocator.release(first)
    assert allocator.allocate(2) == first


def test_failed_reservations_are_atomic() -> None:
    allocator = BlockAllocator(2, 8)
    allocator.allocate(2)
    with pytest.raises(ValueError, match="positive"):
        allocator.allocate(0)
    with pytest.raises(RuntimeError, match="only 5 free"):
        allocator.allocate(6)
    assert allocator.allocate(5).as_block_ids() == [list(range(3, 8))] * 2


def test_random_allocations_are_reproducible_and_exclusive() -> None:
    first = BlockAllocator(2, 32, allocation_order="random", seed=42)
    second = BlockAllocator(2, 32, allocation_order="random", seed=42)
    allocation = first.allocate(8)
    assert allocation == second.allocate(8)
    assert any(set(group.block_ids) != set(range(1, 9)) for group in allocation.groups)
    assert any(
        list(group.block_ids) != sorted(group.block_ids) for group in allocation.groups
    )
    with pytest.raises(RuntimeError, match="only 23 free"):
        first.allocate(24)
    remaining = first.allocate(23)
    assert remaining == second.allocate(23)
    for group, rest in zip(allocation.groups, remaining.groups, strict=True):
        assert not set(group.block_ids) & set(rest.block_ids)
        assert len(set(group.block_ids)) == 8
        assert set(group.block_ids) | set(rest.block_ids) == set(range(1, 32))
    first.release(allocation)
    reused = first.allocate(8)
    for group, original in zip(reused.groups, allocation.groups, strict=True):
        assert set(group.block_ids) == set(original.block_ids)


def test_null_block_is_never_allocated() -> None:
    """The default null block 0 stays out of every allocation (issue #6741)."""
    sequential = BlockAllocator(2, 16)
    random_order = BlockAllocator(2, 16, allocation_order="random", seed=7)
    for allocator in (sequential, random_order):
        allocated_ids: set[int] = set()
        while True:
            try:
                allocation = allocator.allocate(1)
            except RuntimeError:
                break
            for group in allocation.groups:
                assert 0 not in group.block_ids
                allocated_ids.update(group.block_ids)
        assert allocated_ids == set(range(1, 16))


def test_custom_null_block_id_is_excluded() -> None:
    """A configured non-zero null block id is excluded instead of block 0."""
    allocator = BlockAllocator(1, 8, null_block_id=5)
    allocation = allocator.allocate(7)
    assert allocation.as_block_ids() == [[0, 1, 2, 3, 4, 6, 7]]
    with pytest.raises(RuntimeError, match="only 0 free"):
        allocator.allocate(1)


@pytest.mark.parametrize("null_block_id", [-1, -2, 8])
def test_out_of_pool_sentinel_reserves_nothing(null_block_id: int) -> None:
    """Sentinels the server also accepts outside the pool keep every block."""
    allocator = BlockAllocator(1, 8, null_block_id=null_block_id)
    assert allocator.allocate(8).as_block_ids() == [list(range(8))]


def test_exhaustion_when_the_only_block_is_null() -> None:
    """A pool whose single block is the null block allocates nothing."""
    allocator = BlockAllocator(1, 1)
    with pytest.raises(RuntimeError, match="only 0 free"):
        allocator.allocate(1)


def test_empty_allocator_stays_constructible() -> None:
    """The zero-capacity placeholder used before startup stays constructible."""
    assert BlockAllocator(0, 0).allocate(1).as_block_ids() == []
