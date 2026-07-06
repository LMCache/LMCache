# SPDX-License-Identifier: Apache-2.0
"""
Integration smoke tests for the native filesystem L2 adapter.

These tests exercise the real C++ ``LMCacheFSClient`` through
``NativeConnectorL2Adapter``. They are skipped when the native extension is not
available, but do not fall back to a mock connector.
"""

# Standard
from pathlib import Path
import select

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    FSNativeL2AdapterConfig,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])


def _create_object_key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="fs_native_e2e_model",
        kv_rank=0,
    )


def _create_memory_obj(size: int, fill_value: float) -> TensorMemoryObj:
    tensor = torch.empty(size, dtype=torch.float32)
    tensor.fill_(fill_value)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.float32,
        address=0,
        phy_size=size * tensor.element_size(),
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(tensor, metadata, parent_allocator=None)


def _wait_for_event_fd(event_fd: int, timeout: float = 5.0) -> None:
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    events = poll.poll(int(timeout * 1000))
    assert events, f"timed out waiting for event fd {event_fd}"
    try:
        consume_fd(event_fd)
    except BlockingIOError:
        pass


def test_fs_native_batch_store_lookup_load_delete_e2e(tmp_path: Path) -> None:
    pytest.importorskip("lmcache.lmcache_fs")

    config = FSNativeL2AdapterConfig(
        base_path=str(tmp_path),
        num_workers=2,
        relative_tmp_dir="tmp",
        use_odirect=False,
    )
    adapter = create_l2_adapter(config)

    try:
        keys = [_create_object_key(i) for i in range(3)]
        missing_key = _create_object_key(99)
        store_objs = [
            _create_memory_obj(size=256, fill_value=float(i + 1)) for i in range(3)
        ]
        load_objs = [_create_memory_obj(size=256, fill_value=0.0) for _ in keys]

        store_task_id = adapter.submit_store_task(keys, store_objs)
        _wait_for_event_fd(adapter.get_store_event_fd())
        store_results = adapter.pop_completed_store_tasks()
        assert store_results[store_task_id].is_successful()

        lookup_task_id = adapter.submit_lookup_and_lock_task(
            keys + [missing_key],
            _EMPTY_LAYOUT,
        )
        _wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        lookup_bitmap = adapter.query_lookup_and_lock_result(lookup_task_id)
        assert lookup_bitmap is not None
        assert lookup_bitmap.get_indices_list() == [0, 1, 2]

        load_task_id = adapter.submit_load_task(keys, load_objs)
        _wait_for_event_fd(adapter.get_load_event_fd())
        load_bitmap = adapter.query_load_result(load_task_id)
        assert load_bitmap is not None
        assert load_bitmap.get_indices_list() == [0, 1, 2]
        for load_obj, store_obj in zip(load_objs, store_objs, strict=True):
            assert torch.equal(load_obj.tensor, store_obj.tensor)

        adapter.submit_unlock(keys)
        adapter.delete(keys[:2])

        deleted_lookup_task_id = adapter.submit_lookup_and_lock_task(
            keys,
            _EMPTY_LAYOUT,
        )
        _wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        deleted_lookup_bitmap = adapter.query_lookup_and_lock_result(
            deleted_lookup_task_id
        )
        assert deleted_lookup_bitmap is not None
        assert deleted_lookup_bitmap.get_indices_list() == [2]
    finally:
        adapter.close()
