# SPDX-License-Identifier: Apache-2.0
"""L1Manager integration for batched coordinator-owned shared L1."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchRequestSpec,
)
from lmcache.v1.distributed.config import (
    L1ManagerConfig,
    L1MemoryManagerConfig,
    SharedL1Config,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_manager import StorageManager
import lmcache.v1.distributed.l1_manager as l1_manager_module


def _key(seed: int) -> ObjectKey:
    return ObjectKey(seed.to_bytes(4, "big"), "model", 0)


def _layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc([torch.Size([4, 4])], [torch.float16])


def _config() -> L1ManagerConfig:
    return L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=4096,
            use_lazy=False,
            align_bytes=64,
            shm_name="",
            devdax_path="/dev/dax-test",
        ),
        shared_l1_config=SharedL1Config(
            coordinator_host="127.0.0.1",
            coordinator_port=9400,
            authkey_file="/unused/authkey",
            region_id="region",
            layout_id="layout",
            mapping_offset=0,
            visibility_library_path="/unused/visibility.so",
        ),
    )


def _manager(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[L1Manager, MagicMock]:
    client = MagicMock()
    monkeypatch.setattr(
        l1_manager_module,
        "SharedL1Client",
        lambda *_args, **_kwargs: client,
    )
    return L1Manager(_config()), client


def test_shared_manager_write_read_and_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, client = _manager(monkeypatch)
    layout = _layout()
    key = _key(1)
    write_obj = MagicMock()
    read_obj = MagicMock()
    client.reserve_writes.return_value = [write_obj]
    client.reserve_reads.return_value = [read_obj]
    try:
        assert manager.reserve_write([key], [False], layout)[key] == (
            L1Error.SUCCESS,
            write_obj,
        )
        assert manager.finish_write([key])[key] == L1Error.SUCCESS
        assert manager.reserve_read([key])[key] == (L1Error.SUCCESS, read_obj)
        assert manager.finish_read([key])[key] == L1Error.SUCCESS
        client.reserve_writes.assert_called_once_with([key], layout)
        client.reserve_reads.assert_called_once_with([key])
        client.finish_writes.assert_called_once_with([key])
        client.finish_reads.assert_called_once_with([key])

        failed_key = _key(2)
        failed_obj = MagicMock()
        client.reserve_writes.return_value = [failed_obj]
        manager.reserve_write([failed_key], [False], layout)
        assert manager.abort_write([failed_key])[failed_key] == L1Error.SUCCESS
        client.abort_writes.assert_called_once_with([failed_key])
        assert manager.get_object_state(failed_key) is None
    finally:
        manager.close()


def test_shared_manager_preserves_partial_batch_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, client = _manager(monkeypatch)
    keys = [_key(1), _key(2)]
    memory_obj = MagicMock()
    client.reserve_writes.return_value = [memory_obj, None]
    try:
        result = manager.reserve_write(keys, [False, False], _layout())
        assert result[keys[0]] == (L1Error.SUCCESS, memory_obj)
        assert result[keys[1]] == (L1Error.KEY_NOT_WRITABLE, None)
        client.reserve_writes.assert_called_once()
    finally:
        manager.close()


def test_shared_manager_rejects_multi_reader_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = _manager(monkeypatch)
    try:
        with pytest.raises(ValueError, match="TP=1"):
            manager.reserve_read([_key(1)], extra_count=1)
    finally:
        manager.close()


def test_shared_read_batch_rolls_back_after_local_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, client = _manager(monkeypatch)
    keys = [_key(1), _key(2)]
    client.reserve_reads.return_value = [MagicMock(), MagicMock()]
    broken_entry = MagicMock()
    broken_entry.available_for_read.side_effect = RuntimeError("injected failure")
    manager._objects[keys[1]] = broken_entry
    try:
        with pytest.raises(RuntimeError, match="injected failure"):
            manager.reserve_read(keys)
        client.abort_reads.assert_called_once_with(keys)
        assert not manager._objects[keys[0]].read_lock.is_locked()
    finally:
        manager.close()


def test_shared_write_batch_rolls_back_after_local_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, client = _manager(monkeypatch)
    keys = [_key(1), _key(2)]
    client.reserve_writes.return_value = [MagicMock(), MagicMock()]
    l1_object_state = l1_manager_module.L1ObjectState
    call_count = 0

    def fail_second_state(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("injected failure")
        return l1_object_state(*args, **kwargs)

    monkeypatch.setattr(l1_manager_module, "L1ObjectState", fail_second_state)
    try:
        with pytest.raises(RuntimeError, match="injected failure"):
            manager.reserve_write(keys, [False, False], _layout())
        client.abort_writes.assert_called_once_with(keys)
        assert not (manager._objects.keys() & keys)
    finally:
        manager.close()


def test_shared_reader_rejects_producer_layout_mismatch() -> None:
    manager = StorageManager.__new__(StorageManager)
    l1_manager = MagicMock(uses_shared_l1=True)
    manager._l1_manager = l1_manager
    key = _key(1)
    memory_obj = MagicMock()
    memory_obj.get_shapes.return_value = [torch.Size([8, 2])]
    memory_obj.get_dtypes.return_value = [torch.float16]
    l1_manager.reserve_read.return_value = {
        key: (L1Error.SUCCESS, memory_obj),
    }

    with pytest.raises(RuntimeError, match="layout does not match"):
        manager.submit_prefetch_task(PrefetchRequestSpec([key], _layout()))

    l1_manager.finish_read.assert_called_once_with([key], extra_count=0)
