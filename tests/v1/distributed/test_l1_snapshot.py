# SPDX-License-Identifier: Apache-2.0
"""Focused tests for node-local L1 inspection and byte snapshots."""

# Standard
from unittest.mock import MagicMock

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
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.eviction import L1EvictionPolicy
from lmcache.v1.distributed.eviction_policy import LRUEvictionPolicy
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.mp_observability.event_bus import get_event_bus


def _key(value: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(value),
        model_name="test-model",
        kv_rank=0,
    )


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


class TestProtectedReadForInspection:
    def test_missing_and_write_locked_results(self):
        manager = L1Manager(_l1_config())
        write_locked = _key(2)
        manager.reserve_write([write_locked], [False], _layout())

        with manager.protected_read_for_inspection(_key(1)) as (error, obj):
            assert (error, obj) == (L1Error.KEY_NOT_EXIST, None)
        with manager.protected_read_for_inspection(write_locked) as (error, obj):
            assert (error, obj) == (L1Error.KEY_NOT_READABLE, None)
        manager.close()

    def test_blocks_delete_and_releases_on_success_and_exception(self):
        manager = L1Manager(_l1_config())
        key = _key(1)
        _store_ready(manager, key, _layout())

        with manager.protected_read_for_inspection(key) as (error, obj):
            assert error == L1Error.SUCCESS
            assert obj is not None
            assert manager.delete([key]) == {key: L1Error.KEY_IS_LOCKED}
        assert manager.delete([key]) == {key: L1Error.SUCCESS}

        _store_ready(manager, key, _layout())
        with pytest.raises(RuntimeError, match="copy failed"):
            with manager.protected_read_for_inspection(key):
                raise RuntimeError("copy failed")
        assert manager.delete([key]) == {key: L1Error.SUCCESS}
        manager.close()

    def test_temporary_object_cleanup_is_preserved(self):
        manager = L1Manager(_l1_config())
        key = _key(1)
        _store_ready(manager, key, _layout(), temporary=True)

        with manager.protected_read_for_inspection(key) as (error, obj):
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

        with manager.protected_read_for_inspection(key) as (error, obj):
            assert error == L1Error.SUCCESS
            assert obj is not None
            assert manager.finish_read([key])[key] == L1Error.SUCCESS
            assert manager.delete([key]) == {key: L1Error.KEY_IS_LOCKED}

        assert manager.delete([key]) == {key: L1Error.SUCCESS}
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

        with manager.protected_read_for_inspection(keys[1]) as (error, obj):
            assert error == L1Error.SUCCESS
            assert obj is not None

        assert policy.get_eviction_candidates(len(keys)) == before
        listener.on_l1_keys_reserved_read.assert_not_called()
        listener.on_l1_keys_read_finished.assert_not_called()
        publish.assert_not_called()
        manager.close()


class TestStorageManagerL1Snapshot:
    def test_exact_independent_bytes_and_heterogeneous_metadata(self):
        storage_manager = StorageManager(_storage_config())
        key = _key(1)
        layout = _layout()
        source = storage_manager.reserve_write([key], layout, mode="new")[key]
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

        updated = storage_manager.reserve_write([key], layout, mode="update")[key]
        memoryview(updated.byte_array).cast("B")[:] = b"\xff" * len(expected)
        storage_manager.finish_write([key])
        assert snapshot.data == expected
        storage_manager.close()

    def test_missing_and_write_locked_results(self):
        storage_manager = StorageManager(_storage_config())
        write_locked = _key(2)
        storage_manager.reserve_write([write_locked], _layout(), mode="new")

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
        source = storage_manager.reserve_write([key], _layout(), mode="new")[key]
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
        source = storage_manager.reserve_write([key], _layout(), mode="new")[key]
        storage_manager.finish_write([key])

        def failing_byte_array(_memory_obj):
            raise RuntimeError("copy failed")

        monkeypatch.setattr(type(source), "byte_array", property(failing_byte_array))
        with pytest.raises(RuntimeError, match="copy failed"):
            storage_manager.snapshot_l1_object(key)
        assert storage_manager.delete_l1_keys([key]) == (1, 0)
        storage_manager.close()

    def test_gds_backend_is_explicitly_unsupported(self):
        gds = GdsL1Config(file_location="/unused", size_in_bytes=1 << 20)
        storage_manager = StorageManager(_storage_config(gds))
        key = _key(1)
        layout = MemoryLayoutDesc(
            shapes=[torch.Size([4096])], dtypes=[torch.uint8]
        )
        storage_manager.reserve_write([key], layout, mode="new")
        storage_manager.finish_write([key])

        assert storage_manager.snapshot_l1_object(key) == (
            L1Error.UNSUPPORTED_BACKEND,
            None,
        )
        storage_manager.close()
