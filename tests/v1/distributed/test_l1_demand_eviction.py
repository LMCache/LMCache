# SPDX-License-Identifier: Apache-2.0
"""Demand eviction against real L1 allocators, TTL locks, and eviction policies."""

# Standard
from collections.abc import Callable, Iterator
from pathlib import Path
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    GdsL1Config,
    L1ManagerConfig,
    L1MemoryManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager, L1OperationResult
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L1EvictionController,
)

PAGE = 4096


def make_key(index: int, salt: str = "") -> ObjectKey:
    return ObjectKey(ObjectKey.IntHash2Bytes(index), "test", 0, cache_salt=salt)


def make_layout(size: int = PAGE) -> MemoryLayoutDesc:
    return MemoryLayoutDesc([torch.Size([1, 1, size])], [torch.uint8])


def populate(manager: L1Manager, keys: list[ObjectKey], size: int = PAGE) -> None:
    result = manager.reserve_write(keys, [False] * len(keys), make_layout(size))
    assert all(error == L1Error.SUCCESS for error, _ in result.values())
    manager.finish_write(keys)


@pytest.fixture(params=["cpu", "gds"])
def manager_factory(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[Callable[[int], L1Manager]]:
    """GDS exercises slab address accounting only; no device IO is involved."""
    managers: list[L1Manager] = []

    def create(pages: int) -> L1Manager:
        config = L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=pages * PAGE, use_lazy=False, align_bytes=PAGE
            ),
            gds_l1_config=(
                GdsL1Config(str(tmp_path), pages * PAGE)
                if request.param == "gds"
                else None
            ),
        )
        manager = L1Manager(config)
        managers.append(manager)
        return manager

    yield create
    for manager in managers:
        assert manager.memcheck()
        manager.close()


def test_evicts_only_deficit_below_watermark(
    manager_factory: Callable[[int], L1Manager],
) -> None:
    manager = manager_factory(28)
    controller = L1EvictionController(manager, EvictionConfig("LRU"))
    residents = [make_key(i) for i in range(18)]
    populate(manager, residents)
    used, total = manager.get_memory_usage()
    assert used / total < 0.8

    requested = [make_key(i) for i in range(100, 111)]
    # A store without demand eviction still fails without deleting residents.
    result = manager.reserve_write(requested, [False] * 11, make_layout())
    assert all(error == L1Error.OUT_OF_MEMORY for error, _ in result.values())
    result = manager.reserve_write(
        requested,
        [False] * 11,
        make_layout(),
        eviction_candidate_selector=controller.get_eviction_candidates,
    )
    assert all(error == L1Error.SUCCESS for error, _ in result.values())
    # LRU inserts a batch in reverse order to preserve prefix chunks.
    assert manager.get_object_state(residents[-1]) is None
    assert all(manager.get_object_state(k) is not None for k in residents[:-1])
    assert manager.get_memory_usage() == (28 * PAGE, 28 * PAGE)
    assert residents[-1] not in controller.get_eviction_candidates(requested)
    manager.finish_write_and_delete(requested)
    assert manager.get_staging_memory_usage() == 0


def test_fast_path_does_not_consult_policy(
    manager_factory: Callable[[int], L1Manager],
) -> None:
    manager = manager_factory(2)
    key = make_key(0)

    def unexpected_selection(keys: list[ObjectKey]) -> list[ObjectKey]:
        raise AssertionError("An allocation that fits must not invoke eviction")

    result = manager.reserve_write(
        [key], [False], make_layout(), eviction_candidate_selector=unexpected_selection
    )
    assert result[key][0] == L1Error.SUCCESS


@pytest.mark.parametrize("requested_count,locked_count", [(5, 0), (3, 2), (1, 4)])
def test_impossible_reservation_preserves_residents(
    manager_factory: Callable[[int], L1Manager],
    requested_count: int,
    locked_count: int,
) -> None:
    manager = manager_factory(4)
    controller = L1EvictionController(manager, EvictionConfig("LRU"))
    residents = [make_key(i) for i in range(4)]
    populate(manager, residents)
    manager.reserve_read(residents[:locked_count])
    requested = [make_key(i) for i in range(100, 100 + requested_count)]
    result = manager.reserve_write(
        requested,
        [False] * requested_count,
        make_layout(),
        eviction_candidate_selector=controller.get_eviction_candidates,
    )
    assert all(error == L1Error.OUT_OF_MEMORY for error, _ in result.values())
    assert all(manager.get_object_state(k) is not None for k in residents)
    assert manager.get_staging_memory_usage() == 0
    manager.finish_read(residents[:locked_count])


def test_alignment_and_locked_victim(
    manager_factory: Callable[[int], L1Manager],
) -> None:
    manager = manager_factory(4)
    controller = L1EvictionController(manager, EvictionConfig("LRU"))
    residents = [make_key(i) for i in range(3)]
    populate(manager, residents, size=1)
    manager.reserve_read([residents[0]])
    key = make_key(100)
    result = manager.reserve_write(
        [key],
        [False],
        make_layout(PAGE + 1),
        eviction_candidate_selector=controller.get_eviction_candidates,
    )
    error, obj = result[key]
    assert error == L1Error.SUCCESS and obj is not None
    assert obj.get_physical_size() == 2 * PAGE
    assert manager.get_object_state(residents[-1]) is None
    assert manager.get_object_state(residents[-2]) is not None
    assert manager.get_object_state(residents[0]) is not None
    manager.finish_read([residents[0]])


def test_live_staging_writer_is_not_evicted(
    manager_factory: Callable[[int], L1Manager],
) -> None:
    manager = manager_factory(3)
    controller = L1EvictionController(manager, EvictionConfig("LRU"))
    pinned, victim, requested = [make_key(i) for i in range(3)]
    manager.reserve_write([pinned], [False], make_layout(), tag="writer-a")
    staged = manager.reserve_write([pinned], [False], make_layout(), tag="writer-b")
    manager.finish_write([pinned], tag="writer-a")
    populate(manager, [victim])
    result = manager.reserve_write(
        [requested],
        [False],
        make_layout(),
        eviction_candidate_selector=controller.get_eviction_candidates,
    )
    assert result[requested][0] == L1Error.SUCCESS
    assert manager.get_object_state(pinned) is not None
    assert manager.get_object_state(victim) is None
    assert manager.get_staging_memory_usage() == 2 * PAGE
    assert staged[pinned][0] == L1Error.SUCCESS
    assert manager.finish_write_and_delete([pinned], tag="writer-b")[pinned] == (
        L1Error.SUCCESS
    )


@pytest.mark.parametrize("policy", ["noop", "IsolatedLRU"])
def test_policy_constraints_are_respected(
    manager_factory: Callable[[int], L1Manager], policy: str
) -> None:
    manager = manager_factory(2)
    config = EvictionConfig("noop" if policy == "noop" else "IsolatedLRU")
    controller = L1EvictionController(manager, config)
    foreign, own, requested = (
        make_key(0, "other"),
        make_key(1, "own"),
        make_key(2, "own"),
    )
    populate(manager, [foreign])
    populate(manager, [own])
    result = manager.reserve_write(
        [requested],
        [False],
        make_layout(),
        eviction_candidate_selector=controller.get_eviction_candidates,
    )
    assert manager.get_object_state(foreign) is not None
    if policy == "noop":
        assert result[requested][0] == L1Error.OUT_OF_MEMORY
        assert manager.get_object_state(own) is not None
    else:
        assert result[requested][0] == L1Error.SUCCESS
        assert manager.get_object_state(own) is None


def test_fragmentation_does_not_drain_cache(
    manager_factory: Callable[[int], L1Manager],
) -> None:
    manager = manager_factory(4)
    controller = L1EvictionController(manager, EvictionConfig("LRU"))
    residents = [make_key(i) for i in range(4)]
    populate(manager, residents)
    manager.delete(residents[::2])
    key = make_key(100)
    result = manager.reserve_write(
        [key],
        [False],
        make_layout(2 * PAGE),
        eviction_candidate_selector=controller.get_eviction_candidates,
    )
    assert result[key][0] == L1Error.OUT_OF_MEMORY
    assert all(manager.get_object_state(k) is not None for k in residents[1::2])


def test_failed_retry_does_not_evict_additional_residents(
    manager_factory: Callable[[int], L1Manager],
) -> None:
    manager = manager_factory(4)
    controller = L1EvictionController(manager, EvictionConfig("LRU"))
    residents = [make_key(i) for i in range(3)]
    populate(manager, residents)
    # The oldest candidate borders the free page, but is pinned by a reader.
    # Evicting the next candidate covers the byte deficit without producing a
    # contiguous two-page region. Do not keep evicting to fix fragmentation.
    manager.reserve_read([residents[-1]])
    key = make_key(100)
    result = manager.reserve_write(
        [key],
        [False],
        make_layout(2 * PAGE),
        eviction_candidate_selector=controller.get_eviction_candidates,
    )
    assert result[key][0] == L1Error.OUT_OF_MEMORY
    assert manager.get_object_state(residents[1]) is None
    assert manager.get_object_state(residents[0]) is not None
    assert manager.get_object_state(residents[2]) is not None
    assert manager.get_memory_usage()[0] == 2 * PAGE
    assert manager.get_staging_memory_usage() == 0
    manager.finish_read([residents[-1]])


def test_requested_residents_and_duplicate_candidates_are_not_reclaimed(
    manager_factory: Callable[[int], L1Manager],
) -> None:
    manager = manager_factory(3)
    protected, locked, victim, requested = [make_key(i) for i in range(4)]
    populate(manager, [protected, locked, victim])
    manager.reserve_read([locked])

    def select(keys: list[ObjectKey]) -> list[ObjectKey]:
        # A selector can return stale/duplicate keys. Eligibility belongs to
        # the L1 manager and is rechecked in the same reservation transaction.
        return [make_key(999), protected, locked, victim, victim]

    result = manager.reserve_write(
        [protected, requested],
        [False, False],
        make_layout(),
        eviction_candidate_selector=select,
    )
    assert result[protected][0] == L1Error.KEY_NOT_WRITABLE
    assert result[requested][0] == L1Error.SUCCESS
    assert manager.get_object_state(protected) is not None
    assert manager.get_object_state(locked) is not None
    assert manager.get_object_state(victim) is None
    manager.finish_read([locked])


def test_concurrent_store_cannot_take_reclaimed_space(
    manager_factory: Callable[[int], L1Manager],
) -> None:
    manager = manager_factory(1)
    controller = L1EvictionController(manager, EvictionConfig("LRU"))
    populate(manager, [make_key(0)])
    selected, release, competing = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    requested, other = make_key(1), make_key(2)
    results: dict[str, dict[ObjectKey, L1OperationResult]] = {}

    def select(keys: list[ObjectKey]) -> list[ObjectKey]:
        selected.set()
        assert release.wait(5)
        return controller.get_eviction_candidates(keys)

    def reserve() -> None:
        results["prefetch"] = manager.reserve_write(
            [requested], [False], make_layout(), eviction_candidate_selector=select
        )

    def compete() -> None:
        competing.set()
        results["store"] = manager.reserve_write([other], [False], make_layout())

    prefetch = threading.Thread(target=reserve, daemon=True)
    store = threading.Thread(target=compete, daemon=True)
    prefetch.start()
    try:
        assert selected.wait(5)
        store.start()
        assert competing.wait(5)
    finally:
        release.set()
        prefetch.join(5)
        if store.ident is not None:
            store.join(5)
    assert not prefetch.is_alive() and not store.is_alive()
    assert results["prefetch"][requested][0] == L1Error.SUCCESS
    assert results["store"][other][0] == L1Error.OUT_OF_MEMORY
