# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for StorageManager's fused raw-block restore path."""

# Standard
from typing import cast
from unittest.mock import MagicMock, call, create_autospec, patch
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (
    RawBlockL2Adapter,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.memory_management import MemoryObj


class _TestStorageManager(StorageManager):
    """Minimal StorageManager harness without background controllers."""

    def __init__(
        self,
        l1_manager: MagicMock,
        adapters: list[L2AdapterInterface],
        store_policy: str,
    ) -> None:
        self._l1_manager = l1_manager
        self._event_bus = MagicMock()
        self._store_policy_name = store_policy
        self._lifecycle_lock = threading.Lock()
        self._adapter_lease_condition = threading.Condition()
        self._adapter_lease_counts = {}
        self._draining_adapter_ids = set()
        self._adapters_lock = threading.Lock()
        self._l2_adapters = dict(enumerate(adapters))
        self._adapter_descriptors = {
            index: MagicMock(index=index) for index in self._l2_adapters
        }


def _make_key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test-model",
        kv_rank=0,
    )


def _make_layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc(
        shapes=[torch.Size([16])],
        dtypes=[torch.uint8],
    )


def _make_objects(count: int) -> list[MemoryObj]:
    return [create_autospec(MemoryObj, instance=True) for _ in range(count)]


def _make_raw_adapter() -> RawBlockL2Adapter:
    return create_autospec(RawBlockL2Adapter, instance=True)


def _make_manager(
    adapters: list[L2AdapterInterface],
    *,
    store_policy: str = "skip_l1",
) -> tuple[StorageManager, MagicMock]:
    l1_manager = MagicMock()
    l1_manager.finish_write.side_effect = lambda keys: {
        key: L1Error.SUCCESS for key in keys
    }
    l1_manager.delete.side_effect = lambda keys, force=False: {
        key: L1Error.SUCCESS for key in keys
    }
    manager = _TestStorageManager(l1_manager, adapters, store_policy)
    return manager, l1_manager


def _successful_reservations(
    keys: list[ObjectKey],
    objects: list[MemoryObj],
) -> dict[ObjectKey, tuple[L1Error, MemoryObj]]:
    return {
        key: (L1Error.SUCCESS, memory_obj)
        for key, memory_obj in zip(keys, objects, strict=True)
    }


def test_fused_raw_block_restore_loads_full_prefix_and_finishes_temps():
    keys = [_make_key(i) for i in range(3)]
    objects = _make_objects(len(keys))
    adapter = _make_raw_adapter()
    adapter.lookup_and_lock_sync.return_value = [True, True, True]
    adapter.load_sync.return_value = [True, True, True]
    manager, l1_manager = _make_manager([adapter])
    l1_manager.reserve_write.return_value = _successful_reservations(keys, objects)

    assert manager.supports_fused_raw_block_retrieve() is True
    assert manager.load_raw_block_prefix(keys, _make_layout()) == (keys, objects)
    l1_manager.reserve_write.assert_called_once_with(
        keys=keys,
        is_temporary=[True, True, True],
        layout_desc=_make_layout(),
        mode="new",
    )
    adapter.load_sync.assert_called_once_with(keys, objects)
    adapter.submit_unlock.assert_called_once_with(keys)
    l1_manager.finish_write.assert_not_called()
    l1_manager.delete.assert_not_called()

    manager.finish_raw_block_restore(keys)
    l1_manager.finish_write.assert_called_once_with(keys)
    l1_manager.delete.assert_called_once_with(keys, force=True)


def test_fused_raw_block_restore_returns_contiguous_load_prefix():
    keys = [_make_key(i) for i in range(4)]
    objects = _make_objects(len(keys))
    adapter = _make_raw_adapter()
    adapter.lookup_and_lock_sync.return_value = [True, True, True, True]
    adapter.load_sync.return_value = [True, True, False, True]
    manager, l1_manager = _make_manager([adapter])
    l1_manager.reserve_write.return_value = _successful_reservations(keys, objects)

    assert manager.load_raw_block_prefix(keys, _make_layout()) == (
        keys[:2],
        objects[:2],
    )
    l1_manager.finish_write.assert_called_once_with(keys[2:])
    l1_manager.delete.assert_called_once_with(keys[2:], force=True)
    adapter.submit_unlock.assert_called_once_with(keys)


def test_fused_raw_block_restore_zero_prefix_skips_allocation_and_load():
    keys = [_make_key(i) for i in range(3)]
    adapter = _make_raw_adapter()
    adapter.lookup_and_lock_sync.return_value = [False, True, True]
    manager, l1_manager = _make_manager([adapter])

    assert manager.load_raw_block_prefix(keys, _make_layout()) == ([], [])
    l1_manager.reserve_write.assert_not_called()
    adapter.load_sync.assert_not_called()
    adapter.submit_unlock.assert_called_once_with(keys)


def test_fused_raw_block_restore_rejects_unsupported_adapter_topologies():
    raw_adapter_0 = _make_raw_adapter()
    raw_adapter_1 = _make_raw_adapter()
    other_adapter = create_autospec(L2AdapterInterface, instance=True)

    for adapters in (
        [],
        [other_adapter],
        [raw_adapter_0, raw_adapter_1],
    ):
        manager, l1_manager = _make_manager(adapters)
        assert manager.supports_fused_raw_block_retrieve() is False
        assert manager.load_raw_block_prefix([_make_key(0)], _make_layout()) is None
        l1_manager.reserve_write.assert_not_called()


def test_fused_raw_block_restore_requires_explicit_skip_l1_policy():
    adapter = _make_raw_adapter()
    key = _make_key(0)
    default_manager, default_l1 = _make_manager(
        [adapter],
        store_policy="default",
    )

    assert default_manager.supports_fused_raw_block_retrieve() is False
    assert default_manager.load_raw_block_prefix([key], _make_layout()) is None
    default_l1.reserve_write.assert_not_called()
    adapter.lookup_and_lock_sync.assert_not_called()

    skip_l1_manager, _ = _make_manager(
        [_make_raw_adapter()],
        store_policy="skip_l1",
    )
    assert skip_l1_manager.supports_fused_raw_block_retrieve() is True


def test_fused_raw_block_restore_cleans_non_prefix_allocations():
    keys = [_make_key(i) for i in range(3)]
    objects = _make_objects(len(keys))
    adapter = _make_raw_adapter()
    adapter.lookup_and_lock_sync.return_value = [True, True, True]
    adapter.load_sync.return_value = [True]
    manager, l1_manager = _make_manager([adapter])
    l1_manager.reserve_write.return_value = {
        keys[0]: (L1Error.SUCCESS, objects[0]),
        keys[1]: (L1Error.OUT_OF_MEMORY, None),
        keys[2]: (L1Error.SUCCESS, objects[2]),
    }

    assert manager.load_raw_block_prefix(keys, _make_layout()) == (
        keys[:1],
        objects[:1],
    )
    adapter.load_sync.assert_called_once_with(keys[:1], objects[:1])
    l1_manager.finish_write.assert_called_once_with(keys[2:])
    l1_manager.delete.assert_called_once_with(keys[2:], force=True)
    adapter.submit_unlock.assert_called_once_with(keys)


def test_fused_raw_block_restore_cleans_all_reservations_on_load_error():
    keys = [_make_key(i) for i in range(3)]
    objects = _make_objects(len(keys))
    adapter = _make_raw_adapter()
    adapter.lookup_and_lock_sync.return_value = [True, True, True]
    adapter.load_sync.side_effect = RuntimeError("load failed")
    manager, l1_manager = _make_manager([adapter])
    l1_manager.reserve_write.return_value = _successful_reservations(keys, objects)

    with pytest.raises(RuntimeError, match="load failed"):
        manager.load_raw_block_prefix(keys, _make_layout())
    l1_manager.finish_write.assert_called_once_with(keys)
    l1_manager.delete.assert_called_once_with(keys, force=True)
    adapter.submit_unlock.assert_called_once_with(keys)


@pytest.mark.parametrize("failure_source", ["callback", "later_load"])
def test_pipelined_restore_error_preserves_handed_prefix_for_stream_cleanup(
    failure_source: str,
):
    keys = [_make_key(i) for i in range(3)]
    objects = _make_objects(len(keys))
    adapter = _make_raw_adapter()
    cast(MagicMock, adapter.lookup_and_lock_sync).return_value = [True, True, True]
    manager, l1_manager = _make_manager([adapter])
    l1_manager.reserve_write.return_value = _successful_reservations(keys, objects)
    handed_keys: list[ObjectKey] = []
    handed_objects: list[MemoryObj] = []

    def take_loaded_batch(
        start: int,
        end: int,
        batch_keys: list[ObjectKey],
        batch_objects: list[MemoryObj],
    ) -> None:
        assert (start, end) == (0, 1)
        handed_keys.extend(batch_keys)
        handed_objects.extend(batch_objects)
        if failure_source == "callback":
            raise RuntimeError("CUDA enqueue failed")

    def load_sync(load_keys, load_objects, **kwargs):
        assert load_keys == keys
        assert load_objects == objects
        assert kwargs["completion_batch_size"] == 1
        kwargs["on_batch_loaded"](0, 1)
        raise RuntimeError("later read failed")

    cast(MagicMock, adapter.load_sync).side_effect = load_sync
    expected_error = (
        "CUDA enqueue failed" if failure_source == "callback" else "later read failed"
    )
    with pytest.raises(RuntimeError, match=expected_error):
        manager.load_raw_block_prefix(
            keys,
            _make_layout(),
            completion_batch_size=1,
            on_batch_loaded=take_loaded_batch,
        )

    assert handed_keys == keys[:1]
    assert handed_objects == objects[:1]
    l1_manager.finish_write.assert_called_once_with(keys[1:])
    l1_manager.delete.assert_called_once_with(keys[1:], force=True)
    cast(MagicMock, adapter.submit_unlock).assert_called_once_with(keys)
    assert manager._adapter_lease_counts == {}

    manager.finish_raw_block_restore(handed_keys)
    assert l1_manager.finish_write.call_args_list == [
        call(keys[1:]),
        call(keys[:1]),
    ]
    assert l1_manager.delete.call_args_list == [
        call(keys[1:], force=True),
        call(keys[:1], force=True),
    ]


def test_fused_restore_cleans_successful_reservations_on_unlock_error():
    keys = [_make_key(i) for i in range(2)]
    objects = _make_objects(len(keys))
    adapter = _make_raw_adapter()
    adapter.lookup_and_lock_sync.return_value = [True, True]
    adapter.load_sync.return_value = [True, True]
    adapter.submit_unlock.side_effect = RuntimeError("unlock failed")
    manager, l1_manager = _make_manager([adapter])
    l1_manager.reserve_write.return_value = _successful_reservations(keys, objects)

    with pytest.raises(RuntimeError, match="unlock failed"):
        manager.load_raw_block_prefix(keys, _make_layout())

    l1_manager.finish_write.assert_called_once_with(keys)
    l1_manager.delete.assert_called_once_with(keys, force=True)


def test_fused_restore_leases_allow_concurrent_raw_block_loads():
    keys = [_make_key(0)]
    objects = _make_objects(2)
    adapter = _make_raw_adapter()
    load_barrier = threading.Barrier(2)
    results: list[tuple[list[ObjectKey], list[MemoryObj]] | None] = []

    def load_sync(_keys, _objects):
        load_barrier.wait(timeout=5)
        return [True]

    adapter.lookup_and_lock_sync.return_value = [True]
    adapter.load_sync.side_effect = load_sync
    manager, l1_manager = _make_manager([adapter])
    l1_manager.reserve_write.side_effect = [
        _successful_reservations(keys, [objects[0]]),
        _successful_reservations(keys, [objects[1]]),
    ]

    def load() -> None:
        results.append(manager.load_raw_block_prefix(keys, _make_layout()))

    load_threads = [
        threading.Thread(target=load),
        threading.Thread(target=load),
    ]
    for thread in load_threads:
        thread.start()
    for thread in load_threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in load_threads)
    assert len(results) == 2
    assert adapter.load_sync.call_count == 2
    assert adapter.submit_unlock.call_count == 2
    assert manager._adapter_lease_counts == {}


def test_delete_raw_adapter_waits_for_restore_lease_through_unlock():
    keys = [_make_key(0)]
    objects = _make_objects(1)
    adapter = _make_raw_adapter()
    unlock_entered = threading.Event()
    allow_unlock = threading.Event()
    adapter_closed = threading.Event()

    def submit_unlock(_keys):
        unlock_entered.set()
        assert allow_unlock.wait(timeout=5)

    adapter.lookup_and_lock_sync.return_value = [True]
    adapter.load_sync.return_value = [True]
    adapter.submit_unlock.side_effect = submit_unlock
    adapter.close.side_effect = adapter_closed.set
    manager, l1_manager = _make_manager([adapter])
    l1_manager.reserve_write.return_value = _successful_reservations(keys, objects)
    drained = threading.Event()
    drained.set()
    deletion_started = threading.Event()
    manager._store_controller = MagicMock()
    manager._store_controller.request_remove_adapter.side_effect = lambda _adapter_id: (
        deletion_started.set() or drained
    )
    manager._prefetch_controller = MagicMock()
    manager._prefetch_controller.request_remove_adapter.return_value = drained
    manager._l2_eviction_controller = MagicMock()

    load_thread = threading.Thread(
        target=manager.load_raw_block_prefix,
        args=(keys, _make_layout()),
    )
    delete_thread = threading.Thread(target=manager.delete_l2_adapter, args=(0,))
    load_thread.start()
    assert unlock_entered.wait(timeout=5)
    delete_thread.start()
    assert deletion_started.wait(timeout=5)

    # Deletion marks the adapter draining but cannot close it while submit_unlock
    # is still covered by the restore operation's lease.
    assert not adapter_closed.wait(timeout=0.05)
    allow_unlock.set()
    load_thread.join(timeout=5)
    delete_thread.join(timeout=5)

    assert not load_thread.is_alive()
    assert not delete_thread.is_alive()
    assert adapter_closed.is_set()
    adapter.close.assert_called_once_with()
    assert manager._l2_adapters == {}


def test_finish_raw_block_restore_attempts_force_delete_when_finish_fails():
    keys = [_make_key(0)]
    manager, l1_manager = _make_manager([])
    l1_manager.finish_write.side_effect = RuntimeError("finish failed")

    with pytest.raises(RuntimeError, match="finish failed"):
        manager.finish_raw_block_restore(keys)

    l1_manager.delete.assert_called_once_with(keys, force=True)


def test_storage_manager_closes_raw_adapter_before_l1_fixed_buffers():
    manager = StorageManager.__new__(StorageManager)
    manager._prefetch_controller = MagicMock()
    manager._store_controller = MagicMock()
    manager._eviction_controller = MagicMock()
    manager._l2_eviction_controller = MagicMock()
    adapter = MagicMock()
    manager._l2_adapters = {0: adapter}
    manager._lifecycle_lock = threading.Lock()
    manager._adapter_lease_condition = threading.Condition()
    manager._adapter_lease_counts = {}
    manager._draining_adapter_ids = set()
    manager._l1_manager = MagicMock()
    close_order: list[str] = []
    adapter.close.side_effect = lambda: close_order.append("adapter")
    manager._l1_manager.close.side_effect = lambda: close_order.append("l1")

    with patch("lmcache.v1.distributed.storage_manager.PeriodicEventNotifier.shutdown"):
        manager.close()

    assert close_order == ["adapter", "l1"]
