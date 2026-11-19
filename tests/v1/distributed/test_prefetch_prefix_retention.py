# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for complete-prefix retention after prefetch completion."""

# Standard
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from time import monotonic, sleep

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    FetchingPolicy,
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchLockMode,
    PrefetchResult,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.eviction import L1EvictionPolicy
from lmcache.v1.distributed.eviction_policy.lru import LRUEvictionPolicy
from lmcache.v1.distributed.l1_manager import DramL1Manager
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    RetainPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.utils import (
    L1ManagerDescriptor,
    L2AdapterDescriptor,
)
from lmcache.v1.memory_management import MemoryObj, MemoryObjMetadata, TensorMemoryObj

pytestmark = pytest.mark.no_shared_allocator

LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([1024])], dtypes=[torch.float32])


@dataclass
class Cache:
    """Real managers and eviction policies sharing one prefetch controller."""

    controller: PrefetchController
    managers: list[DramL1Manager]
    policies: list[LRUEvictionPolicy]
    adapter: MockL2Adapter


@contextmanager
def cache_context(num_l1: int = 1) -> Iterator[Cache]:
    """Create small CPU pools, with deterministic eviction controlled by tests."""
    with ExitStack() as stack:
        configs = [
            L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=1024 * 1024,
                    use_lazy=False,
                    shm_name="",
                ),
                write_ttl_seconds=60,
                read_ttl_seconds=60,
            )
            for _ in range(num_l1)
        ]
        managers: list[DramL1Manager] = []
        policies: list[LRUEvictionPolicy] = []
        for config in configs:
            manager = DramL1Manager(config)
            stack.callback(manager.close)
            policy = LRUEvictionPolicy()
            manager.register_listener(L1EvictionPolicy(policy))
            managers.append(manager)
            policies.append(policy)

        adapter_config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)
        adapter = MockL2Adapter(adapter_config)
        stack.callback(adapter.close)
        controller = PrefetchController(
            l1_managers=[manager for manager in managers],
            l1_manager_descriptors=[
                L1ManagerDescriptor(index=index, config=config)
                for index, config in enumerate(configs)
            ],
            l2_adapters=[adapter],
            adapter_descriptors=[L2AdapterDescriptor(index=0, config=adapter_config)],
            policy=RetainPrefetchPolicy(),
        )
        controller.start()
        stack.callback(controller.stop)
        yield Cache(controller, managers, policies, adapter)


def make_groups(
    group_ids: tuple[int, ...], num_ranks: int = 1
) -> list[GroupedObjectKeys]:
    """Build four chunk-ordered keys for each requested group and KV rank."""
    return [
        GroupedObjectKeys(
            keys=[
                ObjectKey(ObjectKey.IntHash2Bytes(chunk), "retention", rank, group)
                for chunk in range(4)
            ],
            object_group_id=group,
            layout_desc=LAYOUT,
        )
        for group in group_ids
        for rank in range(num_ranks)
    ]


def seed_l2(adapter: MockL2Adapter, keys: list[ObjectKey]) -> None:
    """Store real CPU memory objects and wait until the mock adapter has them."""
    objects: list[MemoryObj] = []
    for _ in keys:
        tensor = torch.zeros(LAYOUT.shapes[0], dtype=LAYOUT.dtypes[0])
        objects.append(
            TensorMemoryObj(
                raw_data=tensor,
                metadata=MemoryObjMetadata(
                    shape=LAYOUT.shapes[0],
                    dtype=LAYOUT.dtypes[0],
                    address=0,
                    phy_size=tensor.nelement() * tensor.element_size(),
                    ref_count=0,
                ),
                parent_allocator=None,
            )
        )
    adapter.submit_store_task(keys, objects)
    deadline = monotonic() + 10
    while not all(adapter.debug_has_key(key) for key in keys):
        assert monotonic() < deadline, "L2 seed did not complete"
        sleep(0.01)


def seed_l1(manager: DramL1Manager, keys: list[ObjectKey]) -> None:
    """Admit permanent objects through the public L1 write interface."""
    reserved = manager.reserve_write(
        keys, is_temporary=[False] * len(keys), layout_desc=LAYOUT
    )
    assert all(reserved[key][0] == L1Error.SUCCESS for key in keys)
    written = manager.finish_write(keys)
    assert all(result == L1Error.SUCCESS for result in written.values())


def prefetch(controller: PrefetchController, spec: PrefetchTaskSpec) -> PrefetchResult:
    """Submit a public prefetch request and consume its completed result."""
    request_id = controller.submit_prefetch_request(spec)
    assert controller.wait_prefetch_result(request_id, timeout=10)
    result = controller.query_prefetch_result(request_id)
    assert result is not None
    return result


def evict(cache: Cache, l1_index: int, ratio: float) -> list[ObjectKey]:
    """Apply the real LRU policy's eligible victims through public L1 deletion."""
    manager = cache.managers[l1_index]
    actions = cache.policies[l1_index].get_eviction_actions(
        ratio, key_eligible_filter=manager.is_key_evictable
    )
    keys = [key for action in actions for key in action.keys]
    assert keys, "No eligible eviction victims"
    deleted = manager.delete(keys)
    assert all(deleted[key] == L1Error.SUCCESS for key in keys)
    return keys


def resident_keys(manager: DramL1Manager, keys: list[ObjectKey]) -> set[ObjectKey]:
    """Probe physical residency, releasing every read lock acquired by the probe."""
    results = manager.reserve_read(keys)
    present = [key for key in keys if results[key][0] == L1Error.SUCCESS]
    assert all(
        results[key][0] in (L1Error.SUCCESS, L1Error.KEY_NOT_EXIST) for key in keys
    )
    released = manager.finish_read(present)
    assert all(result == L1Error.SUCCESS for result in released.values())
    return set(present)


@pytest.mark.parametrize(
    ("group_ids", "num_ranks"),
    [((0,), 1), ((0, 1), 1), ((1, 0), 1), ((0, 1), 2)],
    ids=["single-group", "two-groups", "reversed-groups", "two-ranks"],
)
@pytest.mark.parametrize("lock_mode", [PrefetchLockMode.NO_LOCK, PrefetchLockMode.LOCK])
@pytest.mark.parametrize("fetching_policy", ["prefix", "whole_chunk"])
def test_l2_prefetch_retains_complete_prefix_after_eviction(
    group_ids: tuple[int, ...],
    num_ranks: int,
    lock_mode: PrefetchLockMode,
    fetching_policy: FetchingPolicy,
) -> None:
    """Evicting half the loaded objects preserves two complete prefix chunks."""
    groups = make_groups(group_ids, num_ranks)
    keys = [key for group in groups for key in group.keys]
    with cache_context() as cache:
        manager = cache.managers[0]
        seed_l2(cache.adapter, keys)
        loaded = prefetch(
            cache.controller,
            PrefetchTaskSpec(
                key_groups=groups,
                lock_mode=lock_mode,
                fetching_policy=fetching_policy,
            ),
        )
        assert loaded.l1_hit_count == 0
        assert loaded.l2_hit_count == len(keys)
        assert [row.count_leading_ones() for row in loaded.hit_cells] == [4] * len(
            groups
        )

        if lock_mode == PrefetchLockMode.LOCK:
            assert all(not manager.is_key_evictable(key) for key in keys)
            assert (
                cache.policies[0].get_eviction_actions(
                    0.5, key_eligible_filter=manager.is_key_evictable
                )
                == []
            )
            released = manager.finish_read(keys)
            assert all(result == L1Error.SUCCESS for result in released.values())
        assert all(manager.is_key_evictable(key) for key in keys)
        assert len(evict(cache, 0, 0.5)) == len(keys) // 2

        remaining = prefetch(
            cache.controller,
            PrefetchTaskSpec(
                key_groups=groups,
                fetching_policy=fetching_policy,
                lock_mode=PrefetchLockMode.NO_LOCK,
                allowed_l2_memory_tags=frozenset(),
            ),
        )
        assert remaining.l2_hit_count == 0
        assert remaining.l1_hit_count == len(keys) // 2
        assert [row.count_leading_ones() for row in remaining.hit_cells] == [2] * len(
            groups
        )


@pytest.mark.parametrize("scenario", ["sliding-window", "split-l1"])
def test_filtered_rows_keep_their_original_chunk_positions(
    scenario: str,
) -> None:
    """A row starting at chunk two must not be zipped with another row's start."""
    groups = make_groups((0, 1))
    if scenario == "sliding-window":
        groups[1] = replace(groups[1], sliding_window_size=2)
    keys = groups[0].keys + groups[1].keys[2:]
    with cache_context(num_l1=1 if scenario == "sliding-window" else 2) as cache:
        seed_l1(cache.managers[0], keys)
        if scenario == "split-l1":
            seed_l1(cache.managers[1], groups[1].keys[:2])
        found = prefetch(
            cache.controller,
            PrefetchTaskSpec(
                key_groups=groups,
                fetching_policy="prefix" if scenario == "sliding-window" else "full",
                lock_mode=PrefetchLockMode.NO_LOCK,
                allowed_l2_memory_tags=frozenset(),
            ),
        )
        expected_rows = [
            [0, 1, 2, 3],
            [2, 3] if scenario == "sliding-window" else [0, 1, 2, 3],
        ]
        assert [row.get_indices_list() for row in found.hit_cells] == expected_rows
        assert len(evict(cache, 0, 1 / 3)) == 2
        assert resident_keys(cache.managers[0], keys) == set(
            groups[0].keys[:3] + groups[1].keys[2:3]
        )


def test_unselected_copy_in_another_l1_stays_cold() -> None:
    """A resident duplicate outside a manager's selected hits is not refreshed."""
    groups = make_groups((0, 1))
    with cache_context(num_l1=2) as cache:
        seed_l1(cache.managers[0], groups[0].keys)
        duplicate = groups[0].keys[0]
        seed_l1(cache.managers[1], [duplicate] + groups[1].keys)
        spec = PrefetchTaskSpec(
            key_groups=groups,
            lock_mode=PrefetchLockMode.NO_LOCK,
            allowed_l2_memory_tags=frozenset(),
        )
        found = prefetch(cache.controller, spec)
        assert [row.count_leading_ones() for row in found.hit_cells] == [4, 4]
        assert evict(cache, 1, 0.2) == [duplicate]
        remaining = prefetch(cache.controller, spec)
        assert [row.count_leading_ones() for row in remaining.hit_cells] == [4, 4]


def test_empty_hits_leave_eviction_state_empty() -> None:
    """An all-miss completion creates no resident or eviction-tracked objects."""
    with cache_context() as cache:
        result = prefetch(
            cache.controller,
            PrefetchTaskSpec(
                key_groups=make_groups((0, 1)),
                lock_mode=PrefetchLockMode.NO_LOCK,
                allowed_l2_memory_tags=frozenset(),
            ),
        )
        assert result.l1_hit_count == result.l2_hit_count == 0
        assert [row.get_indices_list() for row in result.hit_cells] == [[], []]
        assert cache.policies[0].get_num_tracked_keys() == 0
        assert cache.managers[0].get_staging_memory_usage() == 0
