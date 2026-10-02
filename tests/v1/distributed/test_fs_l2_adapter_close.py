# SPDX-License-Identifier: Apache-2.0
"""Verify that filesystem shutdown retains buffers until I/O completes."""

# Standard
from pathlib import Path
from typing import cast
import io
import os
import threading
import time

# Third Party
from typing_extensions import Buffer
import aiofiles.threadpool
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.memory_management import MemoryObj

pytestmark = pytest.mark.no_shared_allocator


class _Buffer:
    """Supply the byte view consumed by the adapter's public I/O methods."""

    def __init__(self, data: bytes) -> None:
        self.byte_array = bytearray(data)


def _store(adapter: FSL2Adapter, key: ObjectKey, payload: bytes) -> None:
    task_id = adapter.submit_store_task([key], [cast(MemoryObj, _Buffer(payload))])
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        results = adapter.pop_completed_store_tasks()
        if task_id in results:
            assert results[task_id].is_successful()
            return
        time.sleep(0.01)
    pytest.fail("filesystem store did not complete")


def test_close_waits_for_executor_read_before_returning_buffer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancelled asyncio awaiter must not release a still-active read buffer."""
    adapter = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    key = ObjectKey(chunk_hash=b"x" * 32, model_name="m", kv_rank=0)
    destination = _Buffer(b"?" * 8)
    read_started = threading.Event()
    release_read = threading.Event()
    read_finished = threading.Event()
    close_started = threading.Event()
    close_finished = threading.Event()
    errors: list[BaseException] = []
    closer: threading.Thread | None = None

    class DelayedReader(io.BufferedReader):
        def readinto(self, target: Buffer) -> int:
            # Perform real file I/O, then hold its executor before the final
            # write to caller-owned bytes. Closing the asyncio task cannot
            # cancel this already-running synchronous operation.
            view = memoryview(target)
            data = os.read(self.fileno(), len(view))
            read_started.set()
            if not release_read.wait(5):
                raise TimeoutError("test did not release the read")
            view[: len(data)] = data
            read_finished.set()
            return len(data)

    def delayed_open(file: str | Path, **_kwargs: object) -> DelayedReader:
        return DelayedReader(io.FileIO(file, "r"))

    def close_adapter() -> None:
        close_started.set()
        try:
            adapter.close()
        except BaseException as error:
            errors.append(error)
        finally:
            close_finished.set()

    try:
        _store(adapter, key, b"OLD-DATA")
        monkeypatch.setattr(aiofiles.threadpool, "sync_open", delayed_open)
        adapter.submit_load_task([key], [cast(MemoryObj, destination)])
        assert read_started.wait(5)
        closer = threading.Thread(target=close_adapter)
        closer.start()
        assert close_started.wait(5)
        assert not close_finished.wait(0.1), "close returned while read owned buffer"
        assert not read_finished.is_set()
        release_read.set()
        assert close_finished.wait(5)
        assert read_finished.is_set()
        assert errors == []
        assert destination.byte_array == b"OLD-DATA"
        # Reuse is safe once close has established physical completion.
        destination.byte_array[:] = b"NEW-DATA"
        assert destination.byte_array == b"NEW-DATA"
    finally:
        release_read.set()
        if closer is not None:
            closer.join(timeout=5)
        else:
            adapter.close()


@pytest.mark.parametrize("operation", ["store", "load"])
def test_close_rejects_new_buffer_io(tmp_path: Path, operation: str) -> None:
    """A closed adapter cannot accept ownership of another source or destination."""
    adapter = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    key = ObjectKey(chunk_hash=b"x" * 32, model_name="m", kv_rank=0)
    objects = [cast(MemoryObj, _Buffer(b"new-data"))]
    adapter.close()
    with pytest.raises(RuntimeError, match="closed"):
        if operation == "store":
            adapter.submit_store_task([key], objects)
        else:
            adapter.submit_load_task([key], objects)
