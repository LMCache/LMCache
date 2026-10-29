# SPDX-License-Identifier: Apache-2.0
"""Focused tests for node-local L1 inspection and byte snapshots."""

# Standard
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    L1BackendType,
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    GdsL1Config,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.error import InspectionReadError, L1Error
from lmcache.v1.distributed.eviction import L1EvictionPolicy
from lmcache.v1.distributed.eviction_policy import LRUEvictionPolicy
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_controllers import L1EvictionController
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
import lmcache.v1.distributed.l1_manager as l1_module
import lmcache.v1.distributed.storage_manager as storage_module


def _key(value: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(value),
        model_name="test-model",
        kv_rank=0,
    )


def _after_copy(monkeypatch: pytest.MonkeyPatch, action: Callable[[], None]) -> None:
    """Run a deterministic interleaving after bytes are copied, before validation."""

    def view_with_hook(buffer: Any) -> MagicMock:
        view = memoryview(buffer).cast("B")
        wrapped = MagicMock()
        wrapped.cast.return_value = wrapped
        wrapped.nbytes = view.nbytes

        def copy() -> bytes:
            data = view.tobytes()
            action()
            return data

        wrapped.tobytes.side_effect = copy
        return wrapped

    monkeypatch.setattr(storage_module, "memoryview", view_with_hook, raising=False)


def _memory_config() -> L1MemoryManagerConfig:
    return L1MemoryManagerConfig(
        size_in_bytes=8 << 20,
        use_lazy=False,
        init_size_in_bytes=8 << 20,
        align_bytes=4096,
        shm_name="",
    )


def _l1_config(gds: GdsL1Config | None = None) -> L1ManagerConfig:
    return L1ManagerConfig(memory_config=_memory_config(), gds_l1_config=gds)


def _storage_config(gds: GdsL1Config | None = None) -> StorageManagerConfig:
    return StorageManagerConfig(
        l1_manager_config=_l1_config(gds),
        eviction_config=EvictionConfig(eviction_policy="noop"),
    )


def _layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc(
        shapes=[torch.Size([4]), torch.Size([2])],
        dtypes=[torch.uint8, torch.int16],
    )


def _store_ready(
    manager: L1Manager,
    key: ObjectKey,
    layout: MemoryLayoutDesc,
    temporary: bool = False,
):
    result = manager.reserve_write([key], [temporary], layout)
    assert result[key][0] == L1Error.SUCCESS
    assert manager.finish_write([key])[key] == L1Error.SUCCESS
    return result[key][1]


class TestReserveReadForInspection:
    def test_expired_normal_lock_does_not_steal_new_reader(self) -> None:
        config = _l1_config()
        config.read_ttl_seconds = 1
        manager = L1Manager(config)
        key = _key(1)
        _store_ready(manager, key, _layout())
        try:
            assert manager.reserve_read([key])[key][0] == L1Error.SUCCESS
            with manager.reserve_read_for_inspection(key) as (error, _):
                assert error == L1Error.SUCCESS
                time.sleep(1.05)
                assert manager.reserve_read([key])[key][0] == L1Error.SUCCESS
            assert manager.delete([key])[key] == L1Error.KEY_IS_LOCKED
            assert manager.finish_read([key])[key] == L1Error.SUCCESS
            assert manager.delete([key])[key] == L1Error.SUCCESS
        finally:
            manager.close()

    def test_replacement_reader_is_not_released(self) -> None:
        manager = L1Manager(_l1_config())
        key = _key(1)
        _store_ready(manager, key, _layout())
        try:
            with pytest.raises(InspectionReadError):
                with manager.reserve_read_for_inspection(key):
                    assert manager.delete([key], force=True)[key] == L1Error.SUCCESS
                    _store_ready(manager, key, _layout())
                    assert manager.reserve_read([key])[key][0] == L1Error.SUCCESS
            assert manager.delete([key])[key] == L1Error.KEY_IS_LOCKED
            assert manager.finish_read([key])[key] == L1Error.SUCCESS
            assert manager.delete([key])[key] == L1Error.SUCCESS
        finally:
            manager.close()

    def test_missing_and_write_locked_results(self):
        manager = L1Manager(_l1_config())
        write_locked = _key(2)
        manager.reserve_write([write_locked], [False], _layout())

        with manager.reserve_read_for_inspection(_key(1)) as (error, obj):
            assert (error, obj) == (L1Error.KEY_NOT_EXIST, None)
        with manager.reserve_read_for_inspection(write_locked) as (error, obj):
            assert (error, obj) == (L1Error.KEY_NOT_READABLE, None)
        manager.close()

    def test_blocks_delete_and_releases_on_success_and_exception(self):
        manager = L1Manager(_l1_config())
        key = _key(1)
        _store_ready(manager, key, _layout())

        with manager.reserve_read_for_inspection(key) as (error, obj):
            assert error == L1Error.SUCCESS
            assert obj is not None
            assert manager.delete([key]) == {key: L1Error.KEY_IS_LOCKED}
        assert manager.delete([key]) == {key: L1Error.SUCCESS}

        _store_ready(manager, key, _layout())
        with pytest.raises(RuntimeError, match="copy failed"):
            with manager.reserve_read_for_inspection(key):
                raise RuntimeError("copy failed")
        assert manager.delete([key]) == {key: L1Error.SUCCESS}
        manager.close()

    def test_temporary_object_cleanup_is_preserved(self):
        manager = L1Manager(_l1_config())
        key = _key(1)
        _store_ready(manager, key, _layout(), temporary=True)

        with manager.reserve_read_for_inspection(key) as (error, obj):
            assert error == L1Error.SUCCESS
            assert obj is not None
            assert manager.get_object_state(key) is not None
        assert manager.get_object_state(key) is None
        manager.close()

    def test_coexists_with_normal_reader(self):
        manager = L1Manager(_l1_config())
        key = _key(1)
        _store_ready(manager, key, _layout())
        assert manager.reserve_read([key])[key][0] == L1Error.SUCCESS

        with manager.reserve_read_for_inspection(key) as (error, obj):
            assert error == L1Error.SUCCESS
            assert obj is not None
            assert manager.finish_read([key])[key] == L1Error.SUCCESS
            assert manager.delete([key]) == {key: L1Error.KEY_IS_LOCKED}

        assert manager.delete([key]) == {key: L1Error.SUCCESS}
        manager.close()

    def test_read_finish_and_temporary_deletion_keep_l1_tag(self, monkeypatch):
        manager = L1Manager(_l1_config())
        key = _key(1)
        _store_ready(manager, key, _layout(), temporary=True)
        publish = MagicMock(wraps=get_event_bus().publish)
        monkeypatch.setattr(get_event_bus(), "publish", publish)
        try:
            assert manager.reserve_read([key])[key][0] == L1Error.SUCCESS
            assert manager.finish_read([key])[key] == L1Error.SUCCESS
            events = [call.args[0] for call in publish.call_args_list]
            for event_type in (EventType.L1_READ_FINISHED, EventType.L1_KEYS_EVICTED):
                event = next(
                    event for event in events if event.event_type == event_type
                )
                assert event.metadata["l1_tag"] == manager.config.tag
        finally:
            manager.close()

    def test_does_not_touch_lru_or_emit_normal_read_events(self, monkeypatch):
        bus = get_event_bus()
        publish = MagicMock(wraps=bus.publish)
        monkeypatch.setattr(bus, "publish", publish)
        manager = L1Manager(_l1_config())
        policy = LRUEvictionPolicy()
        manager.register_listener(L1EvictionPolicy(policy))
        keys = [_key(1), _key(2), _key(3)]
        for key in keys:
            _store_ready(manager, key, _layout())
        listener = MagicMock()
        manager.register_listener(listener)
        before = policy.get_eviction_candidates(len(keys))
        publish.reset_mock()

        with manager.reserve_read_for_inspection(keys[1]) as (error, obj):
            assert error == L1Error.SUCCESS
            assert obj is not None

        assert policy.get_eviction_candidates(len(keys)) == before
        listener.on_l1_keys_reserved_read.assert_not_called()
        listener.on_l1_keys_read_finished.assert_not_called()
        publish.assert_not_called()
        manager.close()


class TestStorageManagerL1Snapshot:
    def test_read_ttl_expires_during_copy_without_releasing_buffer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _storage_config()
        config.l1_manager_config.read_ttl_seconds = 1
        storage = StorageManager(config)
        key = _key(1)
        source = storage.reserve_write([key], _layout())[key]
        memoryview(source.byte_array).cast("B")[:] = b"original"
        storage.finish_write([key])

        def after_copy() -> None:
            time.sleep(1.05)
            assert storage.delete_l1_keys([key]) == (0, 1)

        _after_copy(monkeypatch, after_copy)
        try:
            error, snapshot = storage.snapshot_l1_object(key)
            assert error == L1Error.SUCCESS
            assert snapshot is not None and snapshot.data == b"original"
            assert storage.delete_l1_keys([key]) == (1, 0)
        finally:
            storage.close()

    def test_force_delete_during_copy_defers_reuse(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        storage = StorageManager(_storage_config())
        key = _key(1)
        old = storage.reserve_write([key], _layout())[key]
        memoryview(old.byte_array).cast("B")[:] = b"original"
        storage.finish_write([key])
        new_objects = []

        def copy_after_delete(buffer: Any) -> MagicMock:
            view = memoryview(buffer).cast("B")
            wrapped = MagicMock()
            wrapped.cast.return_value = wrapped

            def copy() -> bytes:
                assert storage.delete_l1_keys([key], force=True) == (1, 0)
                replacement = storage.reserve_write([key], _layout())[key]
                new_objects.append(replacement)
                assert replacement.data_ptr != old.data_ptr
                memoryview(replacement.byte_array).cast("B")[:] = b"replaced"
                storage.finish_write([key])
                return view.tobytes()

            wrapped.tobytes.side_effect = copy
            return wrapped

        monkeypatch.setattr(
            storage_module, "memoryview", copy_after_delete, raising=False
        )
        try:
            assert storage.snapshot_l1_object(key) == (L1Error.KEY_NOT_READABLE, None)
            assert new_objects and bytes(new_objects[0].byte_array) == b"replaced"
            assert storage.delete_l1_keys([key]) == (1, 0)
        finally:
            storage.close()

    def test_gds_copy_falls_back_to_dram(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(l1_module, "initialize_l1_gds_context", lambda *_: None)
        gds = GdsL1Config(file_location="/unused", size_in_bytes=1 << 20)
        first = L1Manager(_l1_config(gds))
        second = L1Manager(L1ManagerConfig(_memory_config(), tag="secondary"))
        storage = StorageManager(_storage_config(gds), _l1_managers=(first, second))
        key = _key(1)
        try:
            gds_layout = MemoryLayoutDesc(
                shapes=[torch.Size([4096])], dtypes=[torch.uint8]
            )
            _store_ready(first, key, gds_layout)
            obj = _store_ready(second, key, _layout())
            memoryview(obj.byte_array).cast("B")[:] = b"secondar"
            error, snapshot = storage.snapshot_l1_object(key)
            assert error == L1Error.SUCCESS
            assert snapshot is not None and snapshot.data == b"secondar"
        finally:
            storage.close()

    def test_unreadable_first_l1_uses_readable_second_l1(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = L1Manager(_l1_config())
        second = L1Manager(L1ManagerConfig(_memory_config(), tag="secondary"))
        storage = StorageManager(_storage_config(), _l1_managers=(first, second))
        key = _key(1)
        try:
            assert (
                first.reserve_write([key], [False], _layout())[key][0]
                == L1Error.SUCCESS
            )
            obj = _store_ready(second, key, _layout())
            memoryview(obj.byte_array).cast("B")[:] = b"secondar"

            def check_copy_lock() -> None:
                assert second.delete([key])[key] == L1Error.KEY_IS_LOCKED

            _after_copy(monkeypatch, check_copy_lock)
            error, snapshot = storage.snapshot_l1_object(key)
            assert error == L1Error.SUCCESS
            assert snapshot is not None and snapshot.data == b"secondar"
            assert second.delete([key])[key] == L1Error.SUCCESS
        finally:
            storage.close()

    def test_all_l1_copies_unreadable(self) -> None:
        first = L1Manager(_l1_config())
        second = L1Manager(L1ManagerConfig(_memory_config(), tag="secondary"))
        storage = StorageManager(_storage_config(), _l1_managers=(first, second))
        key = _key(1)
        try:
            for manager in (first, second):
                assert (
                    manager.reserve_write([key], [False], _layout())[key][0]
                    == L1Error.SUCCESS
                )
            assert storage.snapshot_l1_object(key) == (
                L1Error.KEY_NOT_READABLE,
                None,
            )
        finally:
            storage.close()

    def test_snapshot_from_secondary_l1(self):
        first = L1Manager(_l1_config())
        second_config = L1ManagerConfig(memory_config=_memory_config(), tag="secondary")
        second = L1Manager(second_config)
        storage = StorageManager(_storage_config(), _l1_managers=(first, second))
        key = _key(1)
        try:
            obj = _store_ready(second, key, _layout())
            memoryview(obj.byte_array).cast("B")[:] = b"secondary"[:8]
            error, snapshot = storage.snapshot_l1_object(key)
            assert error == L1Error.SUCCESS
            assert snapshot is not None and snapshot.data == b"secondar"
            assert second.delete([key])[key] == L1Error.SUCCESS
        finally:
            storage.close()

    def test_temporary_snapshot_validated_before_cleanup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = L1Manager(_l1_config())
        monkeypatch.setattr(storage_module, "L1Manager", lambda config: manager)
        storage = StorageManager(_storage_config())
        key = _key(1)
        try:
            source = _store_ready(manager, key, _layout(), temporary=True)
            memoryview(source.byte_array).cast("B")[:] = b"temporary"[:8]
            error, snapshot = storage.snapshot_l1_object(key)
            assert error == L1Error.SUCCESS
            assert snapshot is not None and snapshot.data == b"temporar"
            assert manager.get_object_state(key) is None
        finally:
            storage.close()

    def test_zero_read_ttl_does_not_expire_inspection(self) -> None:
        config = _storage_config()
        config.l1_manager_config.read_ttl_seconds = 0
        storage = StorageManager(config)
        key = _key(1)
        try:
            storage.reserve_write([key], _layout())
            storage.finish_write([key])
            error, snapshot = storage.snapshot_l1_object(key)
            assert error == L1Error.SUCCESS
            assert snapshot is not None and snapshot.size_bytes == 8
        finally:
            storage.close()

    @pytest.mark.parametrize("replace", [False, True])
    def test_force_delete_or_replace_after_copy(
        self, monkeypatch: pytest.MonkeyPatch, replace: bool
    ) -> None:
        storage = StorageManager(_storage_config())
        key = _key(1)
        try:
            source = storage.reserve_write([key], _layout())[key]
            storage.finish_write([key])

            def delete_or_replace() -> None:
                assert storage.delete_l1_keys([key], force=True) == (1, 0)
                if replace:
                    replacement = storage.reserve_write([key], _layout())[key]
                    assert replacement is not source
                    storage.finish_write([key])

            _after_copy(monkeypatch, delete_or_replace)
            assert storage.snapshot_l1_object(key) == (L1Error.KEY_NOT_READABLE, None)
        finally:
            storage.close()

    def test_size_limit_precedes_byte_access(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        storage = StorageManager(_storage_config())
        key = _key(1)
        try:
            source = storage.reserve_write([key], _layout())[key]
            storage.finish_write([key])

            def unexpected_byte_access(obj: Any) -> Any:
                pytest.fail("oversized object must be rejected before byte access")

            monkeypatch.setattr(
                type(source), "byte_array", property(unexpected_byte_access)
            )
            assert storage.snapshot_l1_object(key, max_size_bytes=7) == (
                L1Error.OBJECT_TOO_LARGE,
                None,
            )
            assert storage.delete_l1_keys([key]) == (1, 0)
        finally:
            storage.close()

    def test_lru_eviction_during_snapshot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = L1Manager(_l1_config())
        policy = LRUEvictionPolicy()
        manager.register_listener(L1EvictionPolicy(policy))
        controller = L1EvictionController(
            manager, EvictionConfig(eviction_policy="LRU")
        )
        monkeypatch.setattr(storage_module, "L1Manager", lambda config: manager)
        storage = StorageManager(_storage_config())
        key, victim = _key(1), _key(2)
        try:
            for item in (key, victim):
                storage.reserve_write([item], _layout())
                storage.finish_write([item])

            def evict() -> None:
                actions = policy.get_eviction_actions(
                    1.0, key_eligible_filter=manager.is_key_evictable
                )
                assert [item for action in actions for item in action.keys] == [victim]
                for action in actions:
                    controller.execute_eviction_action(action)
                assert manager.get_object_state(victim) is None
                assert manager.get_object_state(key) is not None

            _after_copy(monkeypatch, evict)
            error, snapshot = storage.snapshot_l1_object(key, max_size_bytes=8)
            assert error == L1Error.SUCCESS
            assert snapshot is not None and snapshot.size_bytes == 8
            assert manager.is_key_evictable(key)
        finally:
            storage.close()

    def test_exact_independent_bytes_and_heterogeneous_metadata(self):
        storage_manager = StorageManager(_storage_config())
        key = _key(1)
        layout = _layout()
        source = storage_manager.reserve_write([key], layout)[key]
        expected = bytes(range(8))
        memoryview(source.byte_array).cast("B")[:] = expected
        storage_manager.finish_write([key])

        error, snapshot = storage_manager.snapshot_l1_object(key)
        assert error == L1Error.SUCCESS
        assert snapshot is not None
        assert snapshot.data == expected
        assert snapshot.size_bytes == len(expected)
        assert snapshot.backend == L1BackendType.DRAM
        assert snapshot.memory_format == "kv_2ltd"
        assert snapshot.shapes == ((4,), (2,))
        assert snapshot.dtypes == ("torch.uint8", "torch.int16")

        assert storage_manager.delete_l1_keys([key]) == (1, 0)
        updated = storage_manager.reserve_write([key], layout)[key]
        memoryview(updated.byte_array).cast("B")[:] = b"\xff" * len(expected)
        storage_manager.finish_write([key])
        assert snapshot.data == expected
        storage_manager.close()

    def test_missing_and_write_locked_results(self):
        storage_manager = StorageManager(_storage_config())
        write_locked = _key(2)
        storage_manager.reserve_write([write_locked], _layout())

        assert storage_manager.snapshot_l1_object(_key(1)) == (
            L1Error.KEY_NOT_EXIST,
            None,
        )
        assert storage_manager.snapshot_l1_object(write_locked) == (
            L1Error.KEY_NOT_READABLE,
            None,
        )
        storage_manager.close()

    def test_copy_holds_protection_against_non_force_delete(self, monkeypatch):
        storage_manager = StorageManager(_storage_config())
        key = _key(1)
        source = storage_manager.reserve_write([key], _layout())[key]
        storage_manager.finish_write([key])
        original_byte_array = type(source).byte_array
        delete_results = []

        def byte_array_while_deleting(memory_obj):
            delete_results.append(storage_manager.delete_l1_keys([key]))
            return original_byte_array.__get__(memory_obj, type(memory_obj))

        monkeypatch.setattr(
            type(source), "byte_array", property(byte_array_while_deleting)
        )
        error, snapshot = storage_manager.snapshot_l1_object(key)

        assert error == L1Error.SUCCESS
        assert snapshot is not None
        assert delete_results == [(0, 1)]
        assert storage_manager.delete_l1_keys([key]) == (1, 0)
        storage_manager.close()

    def test_copy_exception_releases_protection(self, monkeypatch):
        storage_manager = StorageManager(_storage_config())
        key = _key(1)
        source = storage_manager.reserve_write([key], _layout())[key]
        storage_manager.finish_write([key])

        def failing_byte_array(_memory_obj):
            raise RuntimeError("copy failed")

        monkeypatch.setattr(type(source), "byte_array", property(failing_byte_array))
        with pytest.raises(RuntimeError, match="copy failed"):
            storage_manager.snapshot_l1_object(key)
        assert storage_manager.delete_l1_keys([key]) == (1, 0)
        storage_manager.close()

    def test_gds_backend_is_explicitly_unsupported(self, monkeypatch):
        monkeypatch.setattr(l1_module, "initialize_l1_gds_context", lambda *_: None)
        gds = GdsL1Config(file_location="/unused", size_in_bytes=1 << 20)
        storage_manager = StorageManager(_storage_config(gds))
        key = _key(1)
        layout = MemoryLayoutDesc(shapes=[torch.Size([4096])], dtypes=[torch.uint8])
        storage_manager.reserve_write([key], layout)
        storage_manager.finish_write([key])

        assert storage_manager.snapshot_l1_object(key) == (
            L1Error.UNSUPPORTED_BACKEND,
            None,
        )
        storage_manager.close()
