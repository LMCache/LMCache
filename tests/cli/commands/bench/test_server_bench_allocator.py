# SPDX-License-Identifier: Apache-2.0
"""Request-exclusive allocation contracts."""

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.server_bench.allocator import BlockAllocator


def test_sequential_allocation_and_release() -> None:
    allocator = BlockAllocator(2, 8)
    first = allocator.allocate(2)
    assert first.as_block_ids() == [[0, 1], [0, 1]]
    assert [group.group_id for group in first.groups] == [0, 1]
    assert allocator.allocate(2).as_block_ids() == [[2, 3], [2, 3]]
    allocator.release(first)
    assert allocator.allocate(2) == first


def test_failed_reservations_are_atomic() -> None:
    allocator = BlockAllocator(2, 8)
    allocator.allocate(2)
    with pytest.raises(ValueError, match="positive"):
        allocator.allocate(0)
    with pytest.raises(RuntimeError, match="only 6 free"):
        allocator.allocate(7)
    assert allocator.allocate(6).as_block_ids() == [list(range(2, 8))] * 2


def test_random_allocations_are_reproducible_and_exclusive() -> None:
    first = BlockAllocator(2, 32, allocation_order="random", seed=42)
    second = BlockAllocator(2, 32, allocation_order="random", seed=42)
    allocation = first.allocate(8)
    assert allocation == second.allocate(8)
    assert any(set(group.block_ids) != set(range(8)) for group in allocation.groups)
    assert any(
        list(group.block_ids) != sorted(group.block_ids) for group in allocation.groups
    )
    with pytest.raises(RuntimeError, match="only 24 free"):
        first.allocate(25)
    remaining = first.allocate(24)
    assert remaining == second.allocate(24)
    for group, rest in zip(allocation.groups, remaining.groups, strict=True):
        assert not set(group.block_ids) & set(rest.block_ids)
        assert len(set(group.block_ids)) == 8
        assert set(group.block_ids) | set(rest.block_ids) == set(range(32))
    first.release(allocation)
    reused = first.allocate(8)
    for group, original in zip(reused.groups, allocation.groups, strict=True):
        assert set(group.block_ids) == set(original.block_ids)
