# SPDX-License-Identifier: Apache-2.0
"""Streaming I/O ownership when an adapter does not complete on time."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.streaming import FSStreamingReader, StreamingLoadTimeout
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator


@pytest.fixture
def make_reader(monkeypatch):
    # First Party
    from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
    from lmcache.v1.distributed.l1_manager.dram import DramL1Manager

    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *a, **kw: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **kw: None)
    owners = []

    def create(adapter, capacity, **kwargs):
        owner = DramL1Manager(
            L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=capacity + 4096,
                    use_lazy=False,
                    align_bytes=4096,
                    shm_name="",
                )
            )
        )
        owners.append(owner)
        return FSStreamingReader(adapter, capacity, shared_owner=owner, **kwargs)

    yield create
    for owner in owners:
        owner.close()


def test_close_waits_for_active_window(make_reader):
    reader = make_reader(MagicMock(), 4096)
    layout = MemoryLayoutDesc(shapes=[torch.Size([4096])], dtypes=[torch.uint8])
    window = reader.pool.acquire(layout)
    with ThreadPoolExecutor(max_workers=1) as executor:
        closed = executor.submit(reader.close, 2)
        time.sleep(0.05)
        assert not closed.done()
        reader.pool.release(window)
        closed.result(timeout=2)


def test_timed_out_load_keeps_window_until_io_really_completes(make_reader):
    adapter = MagicMock()
    adapter.query_load_result.return_value = None
    reader = make_reader(adapter, 4096, operation_timeout=0.02)
    layout = MemoryLayoutDesc(shapes=[torch.Size([4096])], dtypes=[torch.uint8])
    window = reader.pool.acquire(layout)
    with pytest.raises(StreamingLoadTimeout):
        reader.load_many([MagicMock()], [window])
    with pytest.raises(LMCacheTimeoutError):
        reader.pool.acquire(layout, timeout=0)
    with pytest.raises(RuntimeError, match="outstanding"):
        reader.close(timeout=0)
    bitmap = Bitmap(1)
    bitmap.set(0)
    adapter.query_load_result.return_value = bitmap
    deadline = time.monotonic() + 2
    while reader.pool.get_memory_usage()[0] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert reader.pool.get_memory_usage()[0] == 0
    reader.close(timeout=2)


def test_batch_submits_all_reads_before_waiting(make_reader):
    adapter = MagicMock()
    adapter.submit_load_task.side_effect = [1, 2, 3]
    bitmap = Bitmap(1)
    bitmap.set(0)

    def result(task):
        assert adapter.submit_load_task.call_count == 3
        return bitmap

    adapter.query_load_result.side_effect = result
    reader = make_reader(adapter, 12288, operation_timeout=2)
    layout = MemoryLayoutDesc(shapes=[torch.Size([4096])], dtypes=[torch.uint8])
    windows = [reader.pool.acquire(layout) for _ in range(3)]
    assert reader.load_many([MagicMock() for _ in windows], windows)
    for window in windows:
        reader.pool.release(window)
    reader.close()


def test_partially_submitted_batch_holds_all_windows_until_reads_finish(make_reader):
    adapter = MagicMock()
    adapter.submit_load_task.side_effect = [1, RuntimeError("submit failed")]
    adapter.query_load_result.return_value = None
    reader = make_reader(adapter, 8192, operation_timeout=0.05)
    layout = MemoryLayoutDesc(shapes=[torch.Size([4096])], dtypes=[torch.uint8])
    windows = [reader.pool.acquire(layout) for _ in range(2)]
    with pytest.raises(StreamingLoadTimeout):
        reader.load_many([MagicMock(), MagicMock()], windows)
    assert reader.pool.get_memory_usage()[0] == 8192
    completed = Bitmap(1)
    completed.set(0)
    adapter.query_load_result.return_value = completed
    deadline = time.monotonic() + 2
    while reader.pool.get_memory_usage()[0] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert reader.pool.get_memory_usage()[0] == 0
    reader.close(timeout=2)


def test_deleted_fs_object_after_lookup_reports_failed_load(tmp_path, make_reader):
    # First Party
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.distributed.key_codec import encode_fs_filename
    from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
        FSL2Adapter,
        FSL2AdapterConfig,
    )

    adapter = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    reader = make_reader(adapter, 4096, operation_timeout=2)
    layout = MemoryLayoutDesc(shapes=[torch.Size([4096])], dtypes=[torch.uint8])
    key = ObjectKey(chunk_hash=b"cold", model_name="model", kv_rank=0)
    path = tmp_path / encode_fs_filename(key)
    path.write_bytes(bytes(4096))
    try:
        lookup = adapter.submit_lookup_and_lock_task([key], {0: layout})
        deadline = time.monotonic() + 2
        result = None
        while result is None and time.monotonic() < deadline:
            result = adapter.query_lookup_and_lock_result(lookup)
            if result is None:
                time.sleep(0.01)
        assert result is not None and result.popcount() == 1
        path.unlink()
        window = reader.pool.acquire(layout)
        assert not reader.load_many([key], [window])
        reader.pool.release(window)
        assert reader.pool.get_memory_usage()[0] == 0
    finally:
        reader.close(timeout=2)
        adapter.close()


def test_fs_batch_reads_distinct_objects_into_reusable_windows(tmp_path, make_reader):
    # First Party
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.distributed.key_codec import encode_fs_filename
    from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
        FSL2Adapter,
        FSL2AdapterConfig,
    )

    adapter = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    reader = make_reader(adapter, 8192, operation_timeout=2)
    layout = MemoryLayoutDesc(shapes=[torch.Size([4096])], dtypes=[torch.uint8])
    keys = [
        ObjectKey(chunk_hash=bytes([i]), model_name="model", kv_rank=0)
        for i in range(2)
    ]
    payloads = [bytes([i + 1]) * 4096 for i in range(2)]
    for key, payload in zip(keys, payloads, strict=True):
        (tmp_path / encode_fs_filename(key)).write_bytes(payload)
    try:
        windows = [reader.pool.acquire(layout) for _ in keys]
        assert reader.load_many(keys, windows)
        assert [bytes(window.memory_obj.byte_array) for window in windows] == payloads
        for window in windows:
            reader.pool.release(window)
        assert reader.pool.get_memory_usage()[0] == 0
    finally:
        reader.close(timeout=2)
        adapter.close()


@pytest.mark.parametrize("size", [-1, 1, True, 4097, "4096"])
def test_streaming_pool_config_rejects_invalid_size(tmp_path, size):
    # First Party
    from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import FSL2AdapterConfig

    with pytest.raises(ValueError, match="streaming_buffer_size_bytes"):
        FSL2AdapterConfig.from_dict(
            {"base_path": str(tmp_path), "streaming_buffer_size_bytes": size}
        )


def test_shutdown_interrupts_wait_without_reusing_inflight_memory(make_reader):
    adapter = MagicMock()
    adapter.query_load_result.return_value = None
    reader = make_reader(adapter, 4096, operation_timeout=5)
    layout = MemoryLayoutDesc(shapes=[torch.Size([4096])], dtypes=[torch.uint8])
    window = reader.pool.acquire(layout)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(reader.load_many, [MagicMock()], [window])
        deadline = time.monotonic() + 2
        while not adapter.submit_load_task.called and time.monotonic() < deadline:
            time.sleep(0.001)
        assert adapter.submit_load_task.called
        with pytest.raises(RuntimeError, match="outstanding"):
            reader.close(timeout=0.02)
        with pytest.raises(StreamingLoadTimeout):
            future.result(timeout=2)
        assert reader.pool.get_memory_usage()[0] == 4096
        completed = Bitmap(1)
        completed.set(0)
        adapter.query_load_result.return_value = completed
        reader.close(timeout=2)
