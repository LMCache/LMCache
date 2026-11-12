# SPDX-License-Identifier: Apache-2.0
"""Streaming I/O ownership when an adapter does not complete on time."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from threading import Event
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


def test_close_waits_for_active_window(monkeypatch):
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *a, **kw: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **kw: None)
    reader = FSStreamingReader(MagicMock(), 4096)
    layout = MemoryLayoutDesc(shapes=[torch.Size([4096])], dtypes=[torch.uint8])
    window = reader.pool.acquire(layout)
    with ThreadPoolExecutor(max_workers=1) as executor:
        closed = executor.submit(reader.close, 2)
        time.sleep(0.05)
        assert not closed.done()
        reader.pool.release(window)
        closed.result(timeout=2)


def test_device_fence_retains_windows_without_blocking_caller(monkeypatch):
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *a, **kw: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **kw: None)
    reader = FSStreamingReader(MagicMock(), 4096)
    layout = MemoryLayoutDesc(shapes=[torch.Size([4096])], dtypes=[torch.uint8])
    window = reader.pool.acquire(layout)
    event = Event()
    fence = MagicMock()
    fence.is_complete.side_effect = event.is_set
    reader.release_after_transfer([window], fence)
    with pytest.raises(LMCacheTimeoutError):
        reader.pool.acquire(layout, timeout=0)
    with ThreadPoolExecutor(max_workers=1) as executor:
        closed = executor.submit(reader.close, 2)
        time.sleep(0.05)
        assert not closed.done()
        event.set()
        closed.result(timeout=2)


def test_fence_query_error_keeps_windows_until_a_later_success(monkeypatch):
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *a, **kw: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **kw: None)
    reader = FSStreamingReader(MagicMock(), 4096)
    layout = MemoryLayoutDesc(shapes=[torch.Size([4096])], dtypes=[torch.uint8])
    window = reader.pool.acquire(layout)
    fence = MagicMock()
    fence.is_complete.side_effect = [RuntimeError("query failed"), False, True]

    reader.release_after_transfer([window], fence)
    reader.close(timeout=2)

    assert fence.is_complete.call_count == 3


def test_timed_out_load_keeps_window_until_io_really_completes(monkeypatch):
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *a, **kw: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **kw: None)
    adapter = MagicMock()
    adapter.query_load_result.return_value = None
    reader = FSStreamingReader(adapter, 4096, operation_timeout=0.02)
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


def test_lookup_timeout_returns_control(monkeypatch):
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *a, **kw: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **kw: None)
    adapter = MagicMock()
    adapter.query_lookup_and_lock_result.return_value = None
    reader = FSStreamingReader(adapter, 4096, operation_timeout=0.02)
    with pytest.raises(LMCacheTimeoutError):
        reader.lookup([MagicMock()], {})
    bitmap = Bitmap(1)
    adapter.query_lookup_and_lock_result.return_value = bitmap
    reader.close(timeout=2)


def test_batch_submits_all_reads_before_waiting(monkeypatch):
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *a, **kw: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **kw: None)
    adapter = MagicMock()
    adapter.submit_load_task.side_effect = [1, 2, 3]
    bitmap = Bitmap(1)
    bitmap.set(0)

    def result(task):
        assert adapter.submit_load_task.call_count == 3
        return bitmap

    adapter.query_load_result.side_effect = result
    reader = FSStreamingReader(adapter, 12288, operation_timeout=2)
    layout = MemoryLayoutDesc(shapes=[torch.Size([4096])], dtypes=[torch.uint8])
    windows = [reader.pool.acquire(layout) for _ in range(3)]
    assert reader.load_many([MagicMock() for _ in windows], windows)
    for window in windows:
        reader.pool.release(window)
    reader.close()


def test_fs_unexpected_lookup_error_completes_as_miss(tmp_path, monkeypatch):
    # First Party
    from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
        FSL2Adapter,
        FSL2AdapterConfig,
    )

    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *a, **kw: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **kw: None)
    adapter = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))

    async def failed_lookup(_key):
        raise OSError("injected filesystem failure")

    monkeypatch.setattr(adapter, "_key_exists_on_disk", failed_lookup)
    reader = FSStreamingReader(adapter, 4096, operation_timeout=0.2)
    try:
        assert reader.lookup([MagicMock()], {}).popcount() == 0
    finally:
        reader.close(timeout=1)
        adapter.close()


def test_partially_submitted_batch_holds_all_windows_until_reads_finish(monkeypatch):
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *a, **kw: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **kw: None)
    adapter = MagicMock()
    adapter.submit_load_task.side_effect = [1, RuntimeError("submit failed")]
    adapter.query_load_result.return_value = None
    reader = FSStreamingReader(adapter, 8192, operation_timeout=0.05)
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


def test_deleted_fs_object_after_lookup_reports_failed_load(tmp_path, monkeypatch):
    # First Party
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.distributed.key_codec import encode_fs_filename
    from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
        FSL2Adapter,
        FSL2AdapterConfig,
    )

    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *a, **kw: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **kw: None)
    adapter = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    reader = FSStreamingReader(adapter, 4096, operation_timeout=2)
    layout = MemoryLayoutDesc(shapes=[torch.Size([4096])], dtypes=[torch.uint8])
    key = ObjectKey(chunk_hash=b"cold", model_name="model", kv_rank=0)
    path = tmp_path / encode_fs_filename(key)
    path.write_bytes(bytes(4096))
    try:
        assert reader.lookup([key], {0: layout}).popcount() == 1
        path.unlink()
        window = reader.pool.acquire(layout)
        assert not reader.load_many([key], [window])
        assert reader.pool.release(window)
        assert reader.pool.get_memory_usage()[0] == 0
    finally:
        reader.close(timeout=2)
        adapter.close()


def test_fs_batch_reads_distinct_objects_into_reusable_windows(tmp_path, monkeypatch):
    # First Party
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.distributed.key_codec import encode_fs_filename
    from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
        FSL2Adapter,
        FSL2AdapterConfig,
    )

    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *a, **kw: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **kw: None)
    adapter = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    reader = FSStreamingReader(adapter, 8192, operation_timeout=2)
    layout = MemoryLayoutDesc(shapes=[torch.Size([4096])], dtypes=[torch.uint8])
    keys = [
        ObjectKey(chunk_hash=bytes([i]), model_name="model", kv_rank=0)
        for i in range(2)
    ]
    payloads = [bytes([i + 1]) * 4096 for i in range(2)]
    for key, payload in zip(keys, payloads, strict=True):
        (tmp_path / encode_fs_filename(key)).write_bytes(payload)
    try:
        assert reader.lookup(keys, {0: layout}).popcount() == len(keys)
        windows = [reader.pool.acquire(layout) for _ in keys]
        assert reader.load_many(keys, windows)
        assert [bytes(window.memory_obj.byte_array) for window in windows] == payloads
        for window in windows:
            assert reader.pool.release(window)
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


def test_shutdown_interrupts_wait_without_reusing_inflight_memory(monkeypatch):
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *a, **kw: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **kw: None)
    adapter = MagicMock()
    adapter.query_load_result.return_value = None
    reader = FSStreamingReader(adapter, 4096, operation_timeout=5)
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
