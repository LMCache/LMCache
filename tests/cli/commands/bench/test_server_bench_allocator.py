# SPDX-License-Identifier: Apache-2.0
"""Block allocation tests."""

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.server_bench.allocator import BlockAllocator


def test_independent_pools_and_release() -> None:
    allocator = BlockAllocator(2, 2, 8)
    explicit = allocator.allocate(2, [[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    assert allocator.allocate(2) == (((2, 3), (0, 1)), ((0, 1), (0, 1)))
    allocator.release(explicit)
    assert allocator.allocate(2, explicit) == explicit


def test_failed_reservations_are_atomic() -> None:
    allocator = BlockAllocator(2, 2, 8)
    allocator.allocate(2)
    with pytest.raises(ValueError, match="worker 1/group 1.*occupied"):
        allocator.allocate(2, [[[2, 3], [2, 3]], [[2, 3], [0, 1]]])
    with pytest.raises(RuntimeError, match="only 6 free"):
        allocator.allocate(7)
    assert allocator.allocate(6) == (((2, 3, 4, 5, 6, 7),) * 2,) * 2
