# SPDX-License-Identifier: Apache-2.0
# Contributed by Metrum AI.
"""
Unit tests for NixlObjPool, the slot pool of NixlStoreL2Adapter.

The pool must hand out slots in the same order as the original list-based
pool (never-used slots ascending, then freed slots first-in first-out) at
O(k) cost per call, and must reuse quarantined slots (those of a failed
transfer) only when nothing else is free.
"""

# Standard
import random
import threading

# Third Party
import pytest

nixl = pytest.importorskip("nixl")

# First Party
from lmcache.v1.distributed.l2_adapters.nixl_store_l2_adapter import (  # noqa: E402
    NixlObjPool,
)


class _ListPool:
    """Reference model: the original list-based pool (head pop, tail extend)."""

    def __init__(self, num_total_objs: int) -> None:
        self.indices = list(range(num_total_objs))

    def batched_allocate(self, num_objs: int) -> list[int]:
        if num_objs > len(self.indices):
            return []
        allocated = self.indices[:num_objs]
        self.indices = self.indices[num_objs:]
        return allocated

    def batched_free(self, obj_indices: list[int]) -> None:
        self.indices.extend(obj_indices)


@pytest.mark.parametrize("pool_size", [1, 2, 7, 64, 1000])
@pytest.mark.parametrize("seed", range(4))
def test_matches_original_list_pool(pool_size: int, seed: int) -> None:
    """Random alloc/free sequences give exactly the original pool's results."""
    rng = random.Random(pool_size * 1000 + seed)
    pool, ref = NixlObjPool(pool_size), _ListPool(pool_size)
    held: list[list[int]] = []
    for _ in range(5000):
        if held and rng.random() < 0.45:
            batch = held.pop(rng.randrange(len(held)))
            if rng.random() < 0.5:
                batch = batch[::-1]
            pool.batched_free(batch)
            ref.batched_free(batch)
        else:
            num_objs = rng.randint(0, max(1, pool_size // 4) + 2)
            got = pool.batched_allocate(num_objs)
            assert got == ref.batched_allocate(num_objs)
            if got:
                held.append(got)
        assert pool.num_free() == len(ref.indices)
        assert pool.get_slot_usage()[0] == pytest.approx(
            (pool_size - len(ref.indices)) / pool_size
        )


def test_fresh_slots_ascending_then_freed_slots_fifo() -> None:
    pool = NixlObjPool(8)
    assert pool.batched_allocate(4) == [0, 1, 2, 3]
    pool.batched_free([2, 0])
    pool.batched_free([3])
    assert pool.batched_allocate(6) == [4, 5, 6, 7, 2, 0]
    assert pool.batched_allocate(1) == [3]
    assert pool.batched_allocate(1) == []


def test_quarantined_slots_are_reused_last() -> None:
    pool = NixlObjPool(6)
    assert pool.batched_allocate(4) == [0, 1, 2, 3]
    pool.batched_quarantine([0, 1])
    pool.batched_free([2])
    assert pool.num_free() == 5
    assert pool.get_slot_usage() == (pytest.approx(1 / 6), pytest.approx(1 / 6))
    assert pool.batched_allocate(3) == [4, 5, 2]
    assert pool.batched_allocate(2) == [0, 1]
    assert pool.num_free() == 0


def test_non_positive_request_returns_empty_and_keeps_pool() -> None:
    pool = NixlObjPool(5)
    assert pool.batched_allocate(0) == []
    assert pool.batched_allocate(-2) == []
    assert pool.num_free() == 5
    assert pool.batched_allocate(5) == [0, 1, 2, 3, 4]
    assert pool.batched_allocate(0) == []


def test_exhaustion_is_all_or_nothing() -> None:
    pool = NixlObjPool(4)
    assert pool.batched_allocate(5) == []
    assert pool.num_free() == 4
    assert pool.batched_allocate(4) == [0, 1, 2, 3]
    assert pool.batched_allocate(1) == []
    assert pool.get_slot_usage() == (1.0, 1.0)


def test_empty_pool() -> None:
    pool = NixlObjPool(0)
    assert pool.total_objs == 0
    assert pool.batched_allocate(1) == []
    assert pool.get_slot_usage() == (0.0, 0.0)


def test_large_pool_does_not_materialize_slots() -> None:
    """Construction and allocation cost must not depend on the pool size."""
    pool = NixlObjPool(10**12)
    assert pool.num_free() == 10**12
    assert pool.batched_allocate(3) == [0, 1, 2]
    pool.batched_free([1])
    assert pool.batched_allocate(2) == [3, 4]
    assert pool.num_free() == 10**12 - 4


def test_concurrent_allocate_and_free_never_hand_out_a_slot_twice() -> None:
    pool_size, num_threads, ops = 512, 8, 3000
    pool = NixlObjPool(pool_size)
    owner: dict[int, int] = {}
    guard = threading.Lock()
    errors: list[tuple[str, int]] = []

    def worker(tid: int) -> None:
        rng = random.Random(tid)
        mine: list[list[int]] = []
        for _ in range(ops):
            if mine and rng.random() < 0.5:
                batch = mine.pop(rng.randrange(len(mine)))
                with guard:
                    for idx in batch:
                        if owner.pop(idx, None) != tid:
                            errors.append(("free-not-owner", idx))
                if rng.random() < 0.1:
                    pool.batched_quarantine(batch)
                else:
                    pool.batched_free(batch)
            else:
                batch = pool.batched_allocate(rng.randint(0, 24))
                if batch:
                    with guard:
                        for idx in batch:
                            if idx in owner:
                                errors.append(("double-allocation", idx))
                            owner[idx] = tid
                    mine.append(batch)
        for batch in mine:
            with guard:
                for idx in batch:
                    owner.pop(idx, None)
            pool.batched_free(batch)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert pool.num_free() == pool_size
    assert sorted(pool.batched_allocate(pool_size)) == list(range(pool_size))
