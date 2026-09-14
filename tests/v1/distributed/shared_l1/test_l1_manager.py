# SPDX-License-Identifier: Apache-2.0
"""L1Manager integration for batched coordinator-owned shared L1."""

# Standard
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


def _manager(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[L1Manager, MagicMock]:
    backend = MagicMock()
    monkeypatch.setattr(
        l1_manager_module,
        "SharedDevDaxL1Backend",
        lambda *_args, **_kwargs: backend,
    )
    return L1Manager(_config()), backend


def test_shared_manager_write_read_and_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, backend = _manager(monkeypatch)
    layout = _layout()
    key = _key(1)
    write_obj = MagicMock()
    read_obj = MagicMock()
    backend.reserve_write.return_value = [write_obj]
    backend.reserve_read.return_value = [read_obj]
    try:
        assert manager.uses_shared_l1
        assert manager.reserve_write([key], [False], layout)[key] == (
            L1Error.SUCCESS,
            write_obj,
        )
        assert manager.finish_write([key])[key] == L1Error.SUCCESS
        assert manager.reserve_read([key])[key] == (L1Error.SUCCESS, read_obj)
        assert manager.finish_read([key])[key] == L1Error.SUCCESS
        backend.reserve_write.assert_called_once_with([key], layout)
        backend.reserve_read.assert_called_once_with([key])
        backend.finish_write.assert_called_once_with([key])

        failed_key = _key(2)
        backend.reserve_write.return_value = [MagicMock()]
        manager.reserve_write([failed_key], [False], layout)
        assert manager.abort_write([failed_key])[failed_key] == L1Error.SUCCESS
        backend.abort_write.assert_called_once_with([failed_key])
        assert manager.get_object_state(failed_key) is None
    finally:
        manager.close()


def test_shared_manager_preserves_partial_batch_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, backend = _manager(monkeypatch)
    keys = [_key(1), _key(2)]
    memory_obj = MagicMock()
    backend.reserve_write.return_value = [memory_obj, None]
    try:
        result = manager.reserve_write(keys, [False, False], _layout())
        assert result[keys[0]] == (L1Error.SUCCESS, memory_obj)
        assert result[keys[1]] == (L1Error.KEY_NOT_WRITABLE, None)
        backend.reserve_write.assert_called_once()
    finally:
        manager.close()


def test_shared_manager_rejects_multi_reader_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = _manager(monkeypatch)
    try:
        with pytest.raises(ValueError, match="TP=1"):
            manager.reserve_read([_key(1)], read_locks=2)
        with pytest.raises(ValueError, match="TP=1"):
            manager.finish_read([_key(1)], read_locks=2)
    finally:
        manager.close()


def test_shared_read_batch_rolls_back_after_local_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, backend = _manager(monkeypatch)
    keys = [_key(1), _key(2)]
    backend.reserve_write.return_value = [MagicMock()]
    backend.reserve_read.return_value = [MagicMock(), MagicMock()]
    try:
        # Create a committed entry for keys[1] through the public write path,
        # then make its readability check fail mid-batch.
        assert manager.reserve_write([keys[1]], [False], _layout())[keys[1]][0] == (
            L1Error.SUCCESS
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
    finally:
        manager.close()


def test_shared_write_batch_rolls_back_after_local_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, backend = _manager(monkeypatch)
    keys = [_key(1), _key(2)]
    backend.reserve_write.return_value = [MagicMock(), MagicMock()]
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
        backend.abort_write.assert_called_once_with(keys)
        assert manager.get_object_state(keys[0]) is None
        assert manager.get_object_state(keys[1]) is None
    finally:
        manager.close()


def test_shared_manager_compatibility_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local-semantics APIs stay callable but never touch shared state."""
    manager, backend = _manager(monkeypatch)
    backend.get_memory_usage.return_value = (128, 4096)
    backend.memcheck.return_value = True
    try:
        # delete() refuses instead of pretending to reclaim a global object.
        assert manager.delete([_key(1)]) == {_key(1): L1Error.KEY_NOT_EXIST}
        # clear() is a logged no-op — M0 never reclaims.
        manager.clear(force=True)
        # The L2 prefetch transition is unsupported.
        with pytest.raises(RuntimeError, match="prefetch transition"):
            manager.finish_write_and_reserve_read([_key(1)])
        # Usage and status flow through the backend.
        assert manager.get_memory_usage() == (128, 4096)
        status = manager.report_status()
        assert status["shared_l1"] is True
        assert status["is_healthy"] is True
    finally:
        manager.close()


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
