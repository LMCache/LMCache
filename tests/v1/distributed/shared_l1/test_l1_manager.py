# SPDX-License-Identifier: Apache-2.0
"""L1Manager integration for batched coordinator-owned shared L1."""

# Standard
from collections.abc import Iterator
from contextlib import closing
from typing import Any
from unittest.mock import MagicMock
import threading

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
            coordinator_endpoint="http://127.0.0.1:9400",
            coordinator_token_file="/unused/token",
            region_id="region",
            layout_id="layout",
            mapping_offset_bytes=0,
            visibility_library_path="/unused/visibility.so",
        ),
    )


@pytest.fixture
def manager_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[L1Manager, MagicMock]]:
    backend = MagicMock()
    monkeypatch.setattr(
        l1_manager_module,
        "SharedDevDaxL1Backend",
        lambda *_args, **_kwargs: backend,
    )
    with closing(L1Manager(_config())) as manager:
        yield manager, backend


def test_shared_manager_write_read_and_abort(
    manager_backend: tuple[L1Manager, MagicMock],
) -> None:
    manager, backend = manager_backend
    layout = _layout()
    key = _key(1)
    write_obj = MagicMock()
    read_obj = MagicMock()
    backend.reserve_write.return_value = [write_obj]
    backend.reserve_read.return_value = [read_obj]
    listener = MagicMock()
    manager.register_listener(listener)

    def commit(_keys: list[ObjectKey]) -> None:
        listener.on_l1_keys_write_finished.assert_not_called()
        assert entry.write_lock.is_locked()

    backend.finish_write.side_effect = commit
    assert manager.uses_shared_l1
    assert manager.reserve_write([key], [False], layout)[key] == (
        L1Error.SUCCESS,
        write_obj,
    )
    entry = manager.get_object_state(key)
    assert entry is not None
    assert manager.finish_write([key])[key] == L1Error.SUCCESS
    listener.on_l1_keys_write_finished.assert_called_once_with([key])
    assert manager.reserve_read([key])[key] == (L1Error.SUCCESS, read_obj)
    # Duplicate finishes retain existing batched-read semantics and never evict.
    assert manager.finish_read([key, key])[key] == L1Error.SUCCESS
    assert manager.get_object_state(key) is not None
    listener.on_l1_keys_deleted_by_manager.assert_not_called()
    backend.reserve_write.assert_called_once_with([key], layout)
    backend.reserve_read.assert_called_once_with([key])
    backend.finish_write.assert_called_once_with([key])

    failed_key = _key(2)
    backend.reserve_write.return_value = [MagicMock()]
    manager.reserve_write([failed_key], [False], layout)
    assert manager.abort_write([failed_key])[failed_key] == L1Error.SUCCESS
    backend.abort_write.assert_called_once_with([failed_key])
    assert manager.get_object_state(failed_key) is None


def test_shared_manager_preserves_partial_batch_results(
    manager_backend: tuple[L1Manager, MagicMock],
) -> None:
    manager, backend = manager_backend
    keys = [_key(1), _key(2)]
    memory_obj = MagicMock()
    backend.reserve_write.return_value = [memory_obj, None]
    result = manager.reserve_write(keys, [False, False], _layout())
    assert result[keys[0]] == (L1Error.SUCCESS, memory_obj)
    assert result[keys[1]] == (L1Error.KEY_NOT_WRITABLE, None)
    backend.reserve_write.assert_called_once()


def test_shared_manager_rejects_multi_reader_count(
    manager_backend: tuple[L1Manager, MagicMock],
) -> None:
    manager, _ = manager_backend
    for operation in (manager.reserve_read, manager.finish_read):
        with pytest.raises(ValueError, match="TP=1"):
            operation([_key(1)], read_locks=2)


def test_shared_read_batch_rolls_back_after_local_failure(
    monkeypatch: pytest.MonkeyPatch,
    manager_backend: tuple[L1Manager, MagicMock],
) -> None:
    manager, backend = manager_backend
    keys = [_key(1), _key(2)]
    backend.reserve_write.return_value = [MagicMock()]
    backend.reserve_read.return_value = [MagicMock(), MagicMock()]
    assert (
        manager.reserve_write([keys[1]], [False], _layout())[keys[1]][0]
        == L1Error.SUCCESS
    )
    assert manager.finish_write([keys[1]])[keys[1]] == L1Error.SUCCESS
    entry = manager.get_object_state(keys[1])
    assert entry is not None
    monkeypatch.setattr(
        entry,
        "available_for_read",
        MagicMock(side_effect=RuntimeError("injected failure")),
    )
    with pytest.raises(RuntimeError, match="injected failure"):
        manager.reserve_read(keys)
    rolled_back = manager.get_object_state(keys[0])
    assert rolled_back is not None
    assert not rolled_back.read_lock.is_locked()


def test_shared_write_batch_rolls_back_after_local_failure(
    monkeypatch: pytest.MonkeyPatch,
    manager_backend: tuple[L1Manager, MagicMock],
) -> None:
    manager, backend = manager_backend
    keys = [_key(1), _key(2)]
    backend.reserve_write.return_value = [MagicMock(), MagicMock()]
    l1_object_state = l1_manager_module.L1ObjectState
    call_count = 0

    def fail_second_state(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("injected failure")
        return l1_object_state(*args, **kwargs)

    monkeypatch.setattr(l1_manager_module, "L1ObjectState", fail_second_state)
    with pytest.raises(RuntimeError, match="injected failure"):
        manager.reserve_write(keys, [False, False], _layout())
    backend.abort_write.assert_called_once_with(keys)
    assert all(manager.get_object_state(key) is None for key in keys)


def test_shared_manager_compatibility_surface(
    manager_backend: tuple[L1Manager, MagicMock],
) -> None:
    """Local-semantics APIs stay callable but never touch shared state."""
    manager, backend = manager_backend
    backend.get_memory_usage.return_value = (128, 4096)
    backend.memcheck.return_value = True
    assert manager.delete([_key(1)]) == {_key(1): L1Error.KEY_NOT_EXIST}
    manager.clear(force=True)
    with pytest.raises(RuntimeError, match="prefetch transition"):
        manager.finish_write_and_reserve_read([_key(1)])
    assert manager.get_memory_usage() == (128, 4096)
    status = manager.report_status()
    assert status["shared_l1"] is True
    assert status["is_healthy"] is True


@pytest.mark.parametrize("failure", ["missing", "duplicate", "backend"])
def test_shared_finish_failure_keeps_entire_batch_locked(
    manager_backend: tuple[L1Manager, MagicMock],
    failure: str,
) -> None:
    manager, backend = manager_backend
    keys = [_key(1), _key(2)]
    backend.reserve_write.return_value = [MagicMock(), MagicMock()]
    manager.reserve_write(keys, [False, False], _layout())
    listener = MagicMock()
    manager.register_listener(listener)
    attempted = keys + ([keys[0]] if failure == "duplicate" else [_key(3)])
    if failure == "backend":
        backend.finish_write.side_effect = RuntimeError("injected failure")
        with pytest.raises(RuntimeError, match="injected failure"):
            manager.finish_write(keys)
    elif failure == "duplicate":
        with pytest.raises(ValueError):
            manager.finish_write(attempted)
    else:
        assert manager.finish_write(attempted) == {
            keys[0]: L1Error.KEY_IN_WRONG_STATE,
            keys[1]: L1Error.KEY_IN_WRONG_STATE,
            _key(3): L1Error.KEY_NOT_EXIST,
        }
    if failure != "backend":
        backend.finish_write.assert_not_called()
    listener.on_l1_keys_write_finished.assert_not_called()
    for key in keys:
        entry = manager.get_object_state(key)
        assert entry is not None and entry.write_lock.is_locked()


@pytest.mark.parametrize("object_group_id", [0, 1])
def test_shared_reader_rejects_missing_or_mismatched_layout(
    object_group_id: int,
) -> None:
    manager = StorageManager.__new__(StorageManager)
    l1_manager = MagicMock(uses_shared_l1=True)
    manager._l1_manager = l1_manager
    key = ObjectKey(b"key", "model", 0, object_group_id=object_group_id)
    memory_obj = MagicMock()
    memory_obj.get_shapes.return_value = [torch.Size([8, 2])]
    memory_obj.get_dtypes.return_value = [torch.float16]
    l1_manager.reserve_read.return_value = {
        key: (L1Error.SUCCESS, memory_obj),
    }

    with pytest.raises(RuntimeError, match="layout does not match"):
        manager.submit_prefetch_task(PrefetchRequestSpec([key], {0: _layout()}))

    l1_manager.finish_read.assert_called_once_with([key], read_locks=1)


def test_shared_manager_rejects_runtime_l2_adapter() -> None:
    manager = StorageManager.__new__(StorageManager)
    manager._l1_manager = MagicMock(uses_shared_l1=True)
    manager._lifecycle_lock = threading.Lock()

    with pytest.raises(ValueError, match="cannot be combined with L2 adapters"):
        manager.add_l2_adapter(MagicMock())
