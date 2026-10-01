# SPDX-License-Identifier: Apache-2.0
"""Internal write-overflow foundations, not multi-L1 serving qualification."""

# Standard
from collections.abc import Callable, Iterator
from unittest.mock import Mock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.storage_controllers.write_policy import OrderedWritePolicy
from lmcache.v1.distributed.storage_manager import StorageManager
from tests.v1.distributed.utils import single_row_spec
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator

LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([1024])], dtypes=[torch.float32])
StorageFactory = Callable[
    [tuple[int, ...]], tuple[StorageManager, tuple[L1Manager, ...], OrderedWritePolicy]
]


def key(index: int) -> ObjectKey:
    """Return a deterministic test key."""
    return ObjectKey(ObjectKey.IntHash2Bytes(index), "overflow-test", 0)


@pytest.fixture
def storage_factory(monkeypatch: pytest.MonkeyPatch) -> Iterator[StorageFactory]:
    """Use real CPU allocators and L1 lifecycle; replace only host pinning."""
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)
    stores: list[StorageManager] = []

    def create(
        sizes: tuple[int, ...] = (4096, 4096),
    ) -> tuple[StorageManager, tuple[L1Manager, ...], OrderedWritePolicy]:
        configs = [
            L1ManagerConfig(
                L1MemoryManagerConfig(size_in_bytes=size, use_lazy=False, shm_name="")
            )
            for size in sizes
        ]
        managers = tuple(L1Manager(config) for config in configs)
        policy = OrderedWritePolicy(tuple(m.l1_manager_id for m in managers))
        store = StorageManager(
            StorageManagerConfig(configs[0], EvictionConfig("noop")),
            _l1_managers=managers,
            _write_policy=policy,
        )
        stores.append(store)
        return store, managers, policy

    yield create
    for store in reversed(stores):
        store.close()


def test_primary_success_does_not_try_fallback(
    storage_factory: StorageFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, (primary, fallback), _ = storage_factory((4096, 4096))
    attempt = Mock(wraps=fallback.reserve_write)
    monkeypatch.setattr(fallback, "reserve_write", attempt)
    objects = store.reserve_write([key(1)], LAYOUT)
    assert objects[key(1)].get_l1_manager() == primary.l1_manager_id
    store.finish_write_by_owner(store.prepare_write_completion(objects))
    assert primary.get_object_state(key(1)) is not None
    assert fallback.get_object_state(key(1)) is None
    attempt.assert_not_called()


def test_overflow_order_and_captured_owner_survive_policy_change(
    storage_factory: StorageFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, managers, policy = storage_factory((4096, 4096, 4096))
    for index, manager in enumerate(managers[:2]):
        assert (
            manager.reserve_write([key(index)], [False], LAYOUT)[key(index)][0]
            == L1Error.SUCCESS
        )
        manager.finish_write([key(index)])
    attempts = []
    for manager in managers:
        attempt = Mock(wraps=manager.reserve_write)
        monkeypatch.setattr(manager, "reserve_write", attempt)
        attempts.append(attempt)
    objects = store.reserve_write([key(2)], LAYOUT)
    assert objects[key(2)].get_l1_manager() == managers[2].l1_manager_id
    completion = store.prepare_write_completion(objects)
    policy.manager_ids = tuple(reversed(policy.manager_ids))
    store.finish_write_by_owner(completion)
    assert [m.get_object_state(key(2)) is not None for m in managers] == [
        False,
        False,
        True,
    ]
    assert all(attempt.call_count == 1 for attempt in attempts)


def test_mixed_results_retry_only_oom_subset(
    storage_factory: StorageFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, (primary, fallback), _ = storage_factory((8192, 8192))
    success, conflict, overflow = key(1), key(2), key(3)
    original_reserve = primary.reserve_write

    def mixed(
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        tag: str,
    ) -> dict[ObjectKey, tuple[L1Error, memory_management.MemoryObj | None]]:
        assert keys == [success, conflict, overflow]
        assert is_temporary == [False, False, False]
        assert layout_desc is LAYOUT
        response = original_reserve([success], [False], layout_desc, tag=tag)
        return {
            success: response[success],
            conflict: (L1Error.KEY_NOT_WRITABLE, None),
            overflow: (L1Error.OUT_OF_MEMORY, None),
        }

    monkeypatch.setattr(primary, "reserve_write", mixed)
    attempt = Mock(wraps=fallback.reserve_write)
    monkeypatch.setattr(fallback, "reserve_write", attempt)
    objects = store.reserve_write([success, conflict, overflow], LAYOUT)
    assert list(objects) == [success, overflow]
    assert objects[success].get_l1_manager() == primary.l1_manager_id
    assert objects[overflow].get_l1_manager() == fallback.l1_manager_id
    attempt.assert_called_once_with(
        keys=[overflow],
        is_temporary=[False],
        layout_desc=LAYOUT,
        tag="storage_manager",
    )
    store.finish_write_by_owner(store.prepare_write_completion(objects))
    assert primary.get_object_state(success) is not None
    assert primary.get_object_state(overflow) is None
    assert fallback.get_object_state(overflow) is not None
    assert fallback.get_object_state(conflict) is None


@pytest.mark.parametrize("sizes", [(4096, 4096), (4096, 8192)])
def test_overflow_preserves_batch_atomicity(
    storage_factory: StorageFactory,
    monkeypatch: pytest.MonkeyPatch,
    sizes: tuple[int, ...],
) -> None:
    store, (primary, fallback), _ = storage_factory(sizes)
    attempts = [Mock(wraps=m.reserve_write) for m in (primary, fallback)]
    for manager, attempt in zip((primary, fallback), attempts, strict=True):
        monkeypatch.setattr(manager, "reserve_write", attempt)
    keys = [key(1), key(2)]
    objects = store.reserve_write(keys, LAYOUT)
    for attempt in attempts:
        assert attempt.call_count == 1
        assert attempt.call_args.kwargs["keys"] == keys
    assert primary.get_memory_usage()[0] == 0
    if sizes[1] == 4096:
        # Combined unused bytes suffice, but neither candidate fits the batch.
        assert objects == {}
        assert fallback.get_memory_usage()[0] == 0
    else:
        assert set(objects) == set(keys)
        assert all(
            o.get_l1_manager() == fallback.l1_manager_id for o in objects.values()
        )
        store.finish_write_by_owner(store.prepare_write_completion(objects))
        assert fallback.get_memory_usage()[0] == 8192


@pytest.mark.parametrize(
    "error",
    [
        L1Error.KEY_NOT_WRITABLE,
        L1Error.KEY_IS_LOCKED,
        L1Error.KEY_IN_WRONG_STATE,
        L1Error.KEY_NOT_READABLE,
    ],
)
def test_terminal_errors_do_not_overflow(
    storage_factory: StorageFactory, monkeypatch: pytest.MonkeyPatch, error: L1Error
) -> None:
    store, (primary, fallback), _ = storage_factory((4096, 4096))
    monkeypatch.setattr(
        primary, "reserve_write", Mock(return_value={key(1): (error, None)})
    )
    attempt = Mock(wraps=fallback.reserve_write)
    monkeypatch.setattr(fallback, "reserve_write", attempt)
    assert store.reserve_write([key(1)], LAYOUT) == {}
    attempt.assert_not_called()


def test_exception_is_not_capacity_failure(
    storage_factory: StorageFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, (primary, fallback), _ = storage_factory((4096, 4096))
    monkeypatch.setattr(
        primary, "reserve_write", Mock(side_effect=RuntimeError("failure"))
    )
    attempt = Mock(wraps=fallback.reserve_write)
    monkeypatch.setattr(fallback, "reserve_write", attempt)
    with pytest.raises(RuntimeError, match="failure"):
        store.reserve_write([key(1)], LAYOUT)
    attempt.assert_not_called()


def test_same_key_copies_finish_only_the_selected_owner(
    storage_factory: StorageFactory,
) -> None:
    store, (primary, fallback), _ = storage_factory((4096, 4096))
    objects = []
    for manager in (primary, fallback):
        error, obj = manager.reserve_write(
            [key(1)], [False], LAYOUT, tag="storage_manager"
        )[key(1)]
        assert error == L1Error.SUCCESS and obj is not None
        objects.append(obj)
    assert objects[0].get_l1_manager() != objects[1].get_l1_manager()
    store.finish_write_by_owner(store.prepare_write_completion({key(1): objects[1]}))
    assert primary.reserve_read([key(1)])[key(1)][0] == L1Error.KEY_NOT_EXIST
    assert primary.report_status()["write_locked_count"] == 1
    assert fallback.reserve_read([key(1)])[key(1)][0] == L1Error.SUCCESS
    fallback.finish_read([key(1)])
    assert primary.report_status()["write_locked_count"] == 1


def test_unknown_or_unset_owner_is_not_guessed(storage_factory: StorageFactory) -> None:
    store, _, _ = storage_factory((4096, 4096))
    obj = memory_management.BytesBufferMemoryObj(b"test")
    with pytest.raises(ValueError, match="registered L1 owner"):
        store.prepare_write_completion({key(1): obj})
    obj.set_l1_manager(2**62)
    with pytest.raises(ValueError, match="registered L1 owner"):
        store.prepare_write_completion({key(1): obj})
    with pytest.raises(ValueError, match="registered L1 owner"):
        store.finish_write_by_owner([(2**62, [key(1)])])


def test_multi_manager_harness_rejects_unsupported_serving(
    storage_factory: StorageFactory,
) -> None:
    store, _, _ = storage_factory((4096, 4096))
    with pytest.raises(ValueError, match="owner-routed writes only"):
        store.finish_write([key(1)])
    with pytest.raises(ValueError, match="owner-routed writes only"):
        store.submit_prefetch_task(single_row_spec([key(1)], LAYOUT))
    with pytest.raises(ValueError, match="owner-routed writes only"):
        with store.read_prefetched_results([key(1)]):
            pass
    with pytest.raises(ValueError, match="owner-routed writes only"):
        store.finish_read_prefetched([key(1)])
    with pytest.raises(ValueError, match="owner-routed writes only"):
        _ = store.l1_memory_desc
    for operation in (store.unsafe_read, store.touch_l1_keys, store.delete_l1_keys):
        with pytest.raises(ValueError, match="owner-routed writes only"):
            operation([key(1)])
    with pytest.raises(ValueError, match="owner-routed writes only"):
        store.add_l2_adapter(MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=1))
    with pytest.raises(ValueError, match="owner-routed writes only"):
        store.get_l1_devdax_arena_statuses()
    with pytest.raises(ValueError, match="owner-routed writes only"):
        store.add_l1_devdax_device("/unused", 4096)
    with pytest.raises(ValueError, match="owner-routed writes only"):
        store.remove_l1_devdax_device("/unused")


def test_single_manager_legacy_finish_and_read(storage_factory: StorageFactory) -> None:
    store, (manager,), _ = storage_factory((4096,))
    objects = store.reserve_write([key(1)], LAYOUT)
    tensor = objects[key(1)].tensor
    assert tensor is not None
    tensor.fill_(7)
    store.finish_write([key(1)])
    handle = store.submit_prefetch_task(single_row_spec([key(1)], LAYOUT), skip_l2=True)
    result = store.query_prefetch_status(handle)
    assert result is not None and result.hit_cells[0].popcount() == 1
    with store.read_prefetched_results([key(1)]) as read:
        assert read is not None
        tensor = read[0].tensor
        assert tensor is not None and torch.all(tensor == 7)
    store.finish_read_prefetched([key(1)])
    assert manager.report_status()["read_locked_count"] == 0


def test_policy_rejects_repeated_or_unknown_candidates(
    storage_factory: StorageFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, managers, policy = storage_factory((4096, 4096))
    attempt = Mock(wraps=managers[0].reserve_write)
    monkeypatch.setattr(managers[0], "reserve_write", attempt)
    for candidates in [(managers[0].l1_manager_id,) * 2, (2**62,)]:
        policy.manager_ids = candidates
        with pytest.raises(ValueError, match="distinct registered"):
            store.reserve_write([key(1)], LAYOUT)
    attempt.assert_not_called()


@pytest.mark.parametrize("with_l2", [False, True])
def test_internal_multi_manager_rejects_unwired_controllers(
    storage_factory: StorageFactory, with_l2: bool
) -> None:
    _, managers, _ = storage_factory((4096, 4096))
    config = StorageManagerConfig(
        L1ManagerConfig(L1MemoryManagerConfig(4096, False, shm_name="")),
        EvictionConfig("noop" if with_l2 else "LRU"),
        L2AdaptersConfig(
            [MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=1)]
            if with_l2
            else []
        ),
    )
    with pytest.raises(ValueError, match="no L2 and noop eviction"):
        StorageManager(config, _l1_managers=managers)
