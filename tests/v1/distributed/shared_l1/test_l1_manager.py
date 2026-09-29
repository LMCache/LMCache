# SPDX-License-Identifier: Apache-2.0
"""L1Manager integration for batched coordinator-owned shared L1."""

# Standard
from collections.abc import Iterator
from contextlib import closing
from typing import Any
from unittest.mock import MagicMock
import threading
import uuid

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
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    SharedL1Config,
    StorageManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
    PrefetchPlan,
    register_prefetch_policy,
)
from lmcache.v1.distributed.storage_controllers.utils import MapState
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj
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
def shared_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> MagicMock:
    backend = MagicMock()
    backend.get_memory_usage.return_value = (128, 4096)
    monkeypatch.setattr(
        l1_manager_module,
        "SharedDevDaxL1Backend",
        lambda *_args, **_kwargs: backend,
    )
    return backend


@pytest.fixture
def manager_backend(
    shared_backend: MagicMock,
) -> Iterator[tuple[L1Manager, MagicMock]]:
    with closing(L1Manager(_config())) as manager:
        yield manager, shared_backend


@pytest.fixture
def storage_manager(shared_backend: MagicMock) -> Iterator[StorageManager]:
    config = StorageManagerConfig(_config(), EvictionConfig(eviction_policy="noop"))
    with closing(StorageManager(config)) as manager:
        yield manager


def _rows(window: int = -1, count: int = 4) -> list[GroupedObjectKeys]:
    # Non-contiguous, reversed group IDs and different layouts catch stride
    # arithmetic or assumptions that all rows share a layout.
    return [
        GroupedObjectKeys(
            [ObjectKey(bytes([i]), "model", 0, gid) for i in range(count)],
            gid,
            layout,
            sliding_window_size=window if gid == 7 else -1,
        )
        for gid, layout in (
            (7, _layout()),
            (2, MemoryLayoutDesc([torch.Size([2, 4])], [torch.float32])),
        )
    ]


def _memory_object(layout: MemoryLayoutDesc) -> TensorMemoryObj:
    tensor = torch.empty(layout.shapes[0], dtype=layout.dtypes[0])
    return TensorMemoryObj(
        tensor,
        MemoryObjMetadata(
            shape=layout.shapes[0],
            dtype=layout.dtypes[0],
            shapes=layout.shapes,
            dtypes=layout.dtypes,
            address=0,
            phy_size=tensor.numel() * tensor.element_size(),
            ref_count=0,
        ),
        parent_allocator=None,
    )


@pytest.mark.parametrize("tag", ["", "writer"])
def test_shared_manager_write_read_and_abort(
    manager_backend: tuple[L1Manager, MagicMock],
    tag: str,
) -> None:
    manager, backend = manager_backend
    layout = _layout()
    key = _key(1)
    write_obj = MagicMock()
    write_obj.get_size.return_value = 32
    read_obj = MagicMock()
    backend.reserve_write.return_value = [write_obj]
    backend.reserve_read.return_value = [read_obj]
    listener = MagicMock()
    manager.register_listener(listener)

    def commit(_keys: list[ObjectKey]) -> None:
        listener.on_l1_keys_write_finished.assert_not_called()

    backend.finish_write.side_effect = commit
    assert manager.uses_shared_l1
    assert manager.reserve_write([key], [False], layout, tag=tag)[key] == (
        L1Error.SUCCESS,
        write_obj,
    )
    assert manager.get_object_state(key) is None
    assert manager.get_staging_memory_usage() == 32
    assert manager.delete([key], force=True) == {key: L1Error.KEY_IS_LOCKED}
    with pytest.raises(RuntimeError, match="write-and-delete"):
        manager.finish_write_and_delete([key], tag=tag)
    assert manager.get_object_state(key) is None
    assert manager.report_status()["write_locked_count"] == 1
    backend.finish_write.assert_not_called()
    backend.abort_write.assert_not_called()
    assert manager.finish_write([key], tag=tag)[key] == L1Error.SUCCESS
    assert manager.get_staging_memory_usage() == 0
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
    backend.reserve_write.return_value = [write_obj]
    manager.reserve_write([failed_key], [False], layout, tag=tag)
    assert manager.abort_write([failed_key], tag=tag)[failed_key] == L1Error.SUCCESS
    backend.abort_write.assert_called_once_with([failed_key])
    assert manager.get_object_state(failed_key) is None
    assert manager.get_staging_memory_usage() == 0


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


@pytest.mark.parametrize("operation", ["finish_write", "abort_write"])
def test_shared_write_tag_preserves_reservation_owner(
    manager_backend: tuple[L1Manager, MagicMock],
    operation: str,
) -> None:
    manager, backend = manager_backend
    key = _key(1)
    obj = MagicMock()
    obj.get_size.return_value = 32
    backend.reserve_write.return_value = [obj]
    assert manager.reserve_write([key], [False], _layout(), tag="owner")[key][0] == (
        L1Error.SUCCESS
    )
    backend.reserve_write.return_value = []
    assert manager.reserve_write([key], [False], _layout(), tag="other")[key][0] == (
        L1Error.KEY_NOT_WRITABLE
    )
    assert getattr(manager, operation)([key], tag="other") == {
        key: L1Error.KEY_NOT_EXIST
    }
    assert manager.get_object_state(key) is None
    assert manager.report_status()["write_locked_count"] == 1
    assert manager.get_staging_memory_usage() == 32
    assert getattr(manager, operation)([key], tag="owner") == {key: L1Error.SUCCESS}
    assert manager.get_staging_memory_usage() == 0


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
    read_lock = MagicMock()
    read_lock.lock.side_effect = RuntimeError("injected failure")
    monkeypatch.setattr(entry, "read_lock", read_lock)
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
        assert manager.get_object_state(key) is None
    assert manager.report_status()["write_locked_count"] == len(keys)


@pytest.mark.parametrize("mismatch", ["shape", "dtype"])
@pytest.mark.parametrize("lock_mode", list(PrefetchLockMode))
def test_shared_reader_rejects_mismatched_layout(
    storage_manager: StorageManager,
    shared_backend: MagicMock,
    mismatch: str,
    lock_mode: PrefetchLockMode,
) -> None:
    groups = _rows(count=2)
    objects = [_memory_object(row.layout_desc) for row in groups for _ in row.keys]
    expected = groups[1].layout_desc
    wrong = MemoryLayoutDesc(
        [torch.Size([4, 2])] if mismatch == "shape" else expected.shapes,
        [torch.float16] if mismatch == "dtype" else expected.dtypes,
    )
    objects[-1] = _memory_object(wrong)
    shared_backend.reserve_read.return_value = objects

    with pytest.raises(RuntimeError, match="layout does not match"):
        storage_manager.submit_prefetch_task(
            PrefetchTaskSpec(groups, lock_mode=lock_mode)
        )

    keys = [key for row in groups for key in row.keys]
    assert storage_manager.unsafe_read(keys) == ([], [])
    assert storage_manager.report_status()["l1_manager"]["read_locked_count"] == 0


@pytest.mark.parametrize("skip_l2", [False, True])
@pytest.mark.parametrize("lock_mode", list(PrefetchLockMode))
@pytest.mark.parametrize(
    "policy,window,present,retained",
    [
        ("prefix", -1, [[0, 1, 2, 3], [0, 1, 3]], [[0, 1], [0, 1]]),
        ("prefix", 2, [[0, 1, 2, 3], [0, 1, 2, 3]], [[2, 3], [0, 1, 2, 3]]),
        ("full", -1, [[0, 2], [1, 3]], [[0, 2], [1, 3]]),
        ("full", 2, [[0, 1, 2, 3], [0, 1, 2, 3]], [[], []]),
    ],
    ids=["prefix-gap", "prefix-window", "full-sparse", "full-window-rejected"],
)
def test_shared_prefetch_uses_grouped_policy_contract(
    storage_manager: StorageManager,
    shared_backend: MagicMock,
    policy: FetchingPolicy,
    window: int,
    present: list[list[int]],
    retained: list[list[int]],
    lock_mode: PrefetchLockMode,
    skip_l2: bool,
) -> None:
    groups = _rows(window)
    objects = {
        row.keys[i]: _memory_object(row.layout_desc)
        for row, indices in zip(groups, present, strict=True)
        for i in indices
    }
    shared_backend.reserve_read.side_effect = lambda keys: [
        objects.get(key) for key in keys
    ]
    handle = storage_manager.submit_prefetch_task(
        PrefetchTaskSpec(groups, fetching_policy=policy, lock_mode=lock_mode),
        skip_l2=skip_l2,
    )
    assert handle.total_requested_keys == 8
    assert handle.sliding_windows == (window, -1)
    assert storage_manager.wait_prefetch_status(handle, timeout=0)
    result = storage_manager.query_prefetch_status(handle)
    assert result is not None
    assert [row.get_indices_list() for row in result.hit_cells] == retained
    assert [row.get_indices_list() for row in result.l1_hit_cells] == retained
    assert [len(row) for row in result.l2_hit_cells] == [4, 4]
    assert result.l1_hit_count == sum(map(len, retained))
    assert result.l2_hit_count == 0
    assert storage_manager.query_prefetch_status(handle) is None

    retained_keys = [
        row.keys[i]
        for row, indices in zip(groups, retained, strict=True)
        for i in indices
    ]
    locked = retained_keys if lock_mode is PrefetchLockMode.LOCK else []
    assert storage_manager.unsafe_read(list(objects))[0] == locked
    storage_manager.finish_read_prefetched(locked)
    assert storage_manager.unsafe_read(list(objects)) == ([], [])


def test_shared_prefetch_lookup_hits_consumes_grouped_result(
    storage_manager: StorageManager, shared_backend: MagicMock
) -> None:
    groups = _rows(window=2)
    shared_backend.reserve_read.return_value = [
        _memory_object(row.layout_desc) for row in groups for _ in row.keys
    ]
    handle = storage_manager.submit_prefetch_task(PrefetchTaskSpec(groups))
    # Six retained objects serve four chunks, not six chunks.
    assert storage_manager.query_prefetch_lookup_hits(handle) == 4
    assert storage_manager.query_prefetch_status(handle) is None
    assert storage_manager.query_prefetch_lookup_hits(handle) is None
    storage_manager.finish_read_prefetched(groups[0].keys[2:] + groups[1].keys)


def test_shared_prefetch_empty_rows(
    storage_manager: StorageManager, shared_backend: MagicMock
) -> None:
    shared_backend.reserve_read.return_value = []
    handle = storage_manager.submit_prefetch_task(PrefetchTaskSpec(_rows(count=0)))
    assert handle.total_requested_keys == 0
    assert storage_manager.wait_prefetch_status(handle, timeout=0)
    result = storage_manager.query_prefetch_status(handle)
    assert result is not None
    assert [len(row) for row in result.hit_cells] == [0, 0]
    assert [len(row) for row in result.l1_hit_cells] == [0, 0]
    assert [len(row) for row in result.l2_hit_cells] == [0, 0]
    assert result.l1_hit_count == result.l2_hit_count == 0
    assert storage_manager.query_prefetch_status(handle) is None


def test_shared_prefetch_honors_configured_policy(shared_backend: MagicMock) -> None:
    calls: list[dict[str, Any]] = []

    class RejectPolicy(DefaultPrefetchPolicy):
        def plan_load(self, *args: Any, **kwargs: Any) -> PrefetchPlan:
            calls.append(kwargs)
            return PrefetchPlan(MapState(), MapState())

    policy_name = f"shared-test-{uuid.uuid4().hex}"
    register_prefetch_policy(policy_name, RejectPolicy)
    groups = _rows(count=1)
    shared_backend.reserve_read.return_value = [
        _memory_object(row.layout_desc) for row in groups
    ]
    config = StorageManagerConfig(
        _config(), EvictionConfig(eviction_policy="noop"), prefetch_policy=policy_name
    )
    with closing(StorageManager(config)) as manager:
        handle = manager.submit_prefetch_task(PrefetchTaskSpec(groups))
        result = manager.query_prefetch_status(handle)
        assert result is not None
        assert [row.popcount() for row in result.hit_cells] == [0, 0]
        assert len(calls) == 1
        assert calls[0]["key_groups"] == groups
        assert calls[0]["l1_locked_keys"].merge().popcount() == 2
        assert manager.unsafe_read([row.keys[0] for row in groups]) == ([], [])


@pytest.mark.parametrize("failure", ["backend", "reader-count"])
def test_shared_prefetch_propagates_safety_failures(
    storage_manager: StorageManager,
    shared_backend: MagicMock,
    failure: str,
) -> None:
    shared_backend.reserve_read.side_effect = RuntimeError("visibility failure")
    spec = PrefetchTaskSpec(
        _rows(count=1), num_kv_readers=2 if failure == "reader-count" else 1
    )
    error = ValueError if failure == "reader-count" else RuntimeError
    with pytest.raises(
        error, match="TP=1" if failure == "reader-count" else "visibility"
    ):
        storage_manager.submit_prefetch_task(spec)
    if failure == "reader-count":
        shared_backend.reserve_read.assert_not_called()
    assert storage_manager.report_status()["l1_manager"]["read_locked_count"] == 0


def test_shared_manager_rejects_runtime_l2_adapter() -> None:
    manager = StorageManager.__new__(StorageManager)
    manager._l1_manager = MagicMock(uses_shared_l1=True)
    manager._lifecycle_lock = threading.Lock()

    with pytest.raises(ValueError, match="cannot be combined with L2 adapters"):
        manager.add_l2_adapter(MagicMock())
