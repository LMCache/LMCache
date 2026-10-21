# SPDX-License-Identifier: Apache-2.0
"""Atomic publication tests for filesystem L2 cache objects."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import multiprocessing
import select
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
    _object_key_to_filename,
    _publish_temp_file,
    _write_all,
)


class _BufferObj:
    def __init__(self, payload: bytes) -> None:
        self._buffer = bytearray(payload)

    @property
    def byte_array(self) -> memoryview:
        return memoryview(self._buffer)

    def get_size(self) -> int:
        return len(self._buffer)


def _key() -> ObjectKey:
    return ObjectKey(
        chunk_hash=bytes.fromhex("aabbccdd"),
        model_name="atomic/model",
        kv_rank=0,
        object_group_id=3,
    )


def _wait_for_adapter_stores(
    adapter: FSL2Adapter, expected: int, timeout: float = 10.0
) -> dict:
    completed: dict[int, Any] = {}
    deadline = time.monotonic() + timeout
    while len(completed) < expected and time.monotonic() < deadline:
        completed.update(adapter.pop_completed_store_tasks())
        if len(completed) < expected:
            time.sleep(0.01)
    assert len(completed) == expected
    return completed


def _wait_for_native_completion(client, timeout: float = 20.0):
    poller = select.poll()
    poller.register(client.event_fd(), select.POLLIN)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if poller.poll(50):
            completions = client.drain_completions()
            if completions:
                return completions[0]
    raise AssertionError("native connector completion timed out")


def _native_store_process(
    base_path: str,
    value: int,
    payload_size: int,
    ready_queue: Any,
    start_event: Any,
    result_queue: Any,
) -> None:
    try:
        # First Party
        from lmcache.lmcache_fs import LMCacheFSClient

        client = LMCacheFSClient(base_path, 1, "pending", False, 0)
        try:
            ready_queue.put(True)
            if not start_event.wait(timeout=20.0):
                raise RuntimeError("cross-process start barrier timed out")
            payload = memoryview(bytearray(bytes([value]) * payload_size))
            future_id = client.submit_batch_set(
                ["atomic/model@00000000@3@aabbccdd"], [payload]
            )
            completed_id, ok, error, _results = _wait_for_native_completion(client)
            result_queue.put((completed_id == future_id, ok, error))
        finally:
            client.close()
    except BaseException as exc:
        result_queue.put((False, False, repr(exc), None))


def test_publish_keeps_existing_complete_inode(tmp_path) -> None:
    final_path = tmp_path / "shared.data"
    temp_path = tmp_path / "shared.data.tmp.writer"
    final_path.write_bytes(b"established")
    temp_path.write_bytes(b"replacement")

    assert _publish_temp_file(temp_path, final_path) is False
    assert final_path.read_bytes() == b"established"
    assert not temp_path.exists()


def test_concurrent_publish_selects_one_complete_inode(tmp_path) -> None:
    final_path = tmp_path / "shared.data"
    payloads = [bytes([value]) * (1024 * 1024) for value in range(8)]
    temp_paths = []
    for index, payload in enumerate(payloads):
        temp_path = tmp_path / f"shared.data.tmp.{index}"
        temp_path.write_bytes(payload)
        temp_paths.append(temp_path)

    with ThreadPoolExecutor(max_workers=len(temp_paths)) as executor:
        published = list(
            executor.map(lambda path: _publish_temp_file(path, final_path), temp_paths)
        )

    assert published.count(True) == 1
    assert final_path.read_bytes() in payloads
    assert all(not path.exists() for path in temp_paths)


def test_write_all_retries_short_writes(monkeypatch) -> None:
    written = bytearray()

    def short_write(_fd: int, data: memoryview) -> int:
        count = min(3, len(data))
        written.extend(data[:count])
        return count

    monkeypatch.setattr("os.write", short_write)
    _write_all(7, memoryview(b"complete-payload"))
    assert written == b"complete-payload"


def test_python_adapter_duplicate_stores_publish_one_value(tmp_path) -> None:
    pending = tmp_path / "pending"
    adapter = FSL2Adapter(
        FSL2AdapterConfig(base_path=str(tmp_path), relative_tmp_dir="pending")
    )
    payloads = [bytes([value]) * (4 * 1024 * 1024) for value in range(4)]
    try:
        task_ids = []
        for payload in payloads:
            objects: Any = [_BufferObj(payload)]
            task_ids.append(adapter.submit_store_task([_key()], objects))
        completed = _wait_for_adapter_stores(adapter, len(task_ids))

        assert all(completed[task_id].is_successful() for task_id in task_ids)
        assert sum(
            completed[task_id].bytes_transferred() for task_id in task_ids
        ) == len(payloads[0])
        assert (tmp_path / _object_key_to_filename(_key())).read_bytes() in payloads
        assert list(pending.iterdir()) == []
    finally:
        adapter.close()


def test_native_duplicate_workers_publish_one_value(tmp_path) -> None:
    lmcache_fs = pytest.importorskip("lmcache.lmcache_fs")
    pending = tmp_path / "pending"
    client = lmcache_fs.LMCacheFSClient(str(tmp_path), 4, "pending", False, 0)
    payloads = [bytearray(bytes([value]) * (4 * 1024 * 1024)) for value in range(4)]
    try:
        future_id = client.submit_batch_set(
            ["atomic/model@00000000@3@aabbccdd"] * len(payloads),
            [memoryview(payload) for payload in payloads],
        )
        completed_id, ok, error, _results = _wait_for_native_completion(client)

        assert completed_id == future_id
        assert ok, error
        assert (tmp_path / _object_key_to_filename(_key())).read_bytes() in payloads
        assert list(pending.iterdir()) == []
    finally:
        client.close()


def test_native_cross_process_writers_publish_one_value(tmp_path) -> None:
    pytest.importorskip("lmcache.lmcache_fs")
    ctx = multiprocessing.get_context("spawn")
    ready_queue = ctx.Queue()
    result_queue = ctx.Queue()
    start_event = ctx.Event()
    payload_size = 16 * 1024 * 1024
    values = [ord("a"), ord("b")]
    processes = [
        ctx.Process(
            target=_native_store_process,
            args=(
                str(tmp_path),
                value,
                payload_size,
                ready_queue,
                start_event,
                result_queue,
            ),
        )
        for value in values
    ]
    for process in processes:
        process.start()
    try:
        for _ in processes:
            assert ready_queue.get(timeout=20.0) is True
        start_event.set()
        results = [result_queue.get(timeout=30.0) for _ in processes]
        assert all(submitted and ok for submitted, ok, _error in results), results
    finally:
        start_event.set()
        for process in processes:
            process.join(timeout=30.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        ready_queue.close()
        result_queue.close()

    published = (tmp_path / _object_key_to_filename(_key())).read_bytes()
    assert published in [bytes([value]) * payload_size for value in values]
    assert list((tmp_path / "pending").iterdir()) == []
