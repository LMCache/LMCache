# SPDX-License-Identifier: Apache-2.0
"""Filesystem lookup behavior and event-loop responsiveness."""

# Standard
from collections.abc import Iterator
from pathlib import Path
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any, cast
import os
import select
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.key_codec import encode_fs_filename
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.platform import consume_fd

if TYPE_CHECKING:
    # First Party
    from lmcache.lmcache_native import Bitmap
    from lmcache.v1.memory_management import MemoryObj


def make_key(index: int) -> ObjectKey:
    """Return a distinct key with the same model and rank."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(index), model_name="lookup-model", kv_rank=0
    )


def wait_for_lookup(adapter: FSL2Adapter, task_id: int) -> "Bitmap":
    """Wait for a public lookup result using its completion fd."""
    poller = select.poll()
    fd = adapter.get_lookup_and_lock_event_fd()
    poller.register(fd, select.POLLIN)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = adapter.query_lookup_and_lock_result(task_id)
        if result is not None:
            return result
        if poller.poll(100):
            consume_fd(fd)
    raise AssertionError("lookup did not complete")


@pytest.fixture
def adapter(tmp_path: Path) -> Iterator[FSL2Adapter]:
    """Create an adapter observing the test directory."""
    instance = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    try:
        yield instance
    finally:
        instance.close()


@pytest.mark.parametrize("size", [0, 1, 25, 64, 257])
def test_lookup_preserves_positions_and_is_consumed_once(
    adapter: FSL2Adapter, tmp_path: Path, size: int
) -> None:
    """Every hit bit describes its original input position, including holes."""
    keys = [make_key(i) for i in range(size)]
    for i, key in enumerate(keys):
        if i % 3 == 0:
            (tmp_path / encode_fs_filename(key)).write_bytes(b"data")
    task_id = adapter.submit_lookup_and_lock_task(keys, {})
    result = wait_for_lookup(adapter, task_id)
    assert len(result) == size
    assert result.get_indices_list() == list(range(0, size, 3))
    assert adapter.query_lookup_and_lock_result(task_id) is None


def test_lookup_observes_external_creates_and_deletes(
    adapter: FSL2Adapter, tmp_path: Path
) -> None:
    """Shared-filesystem updates must be visible on the next lookup."""
    keys = [make_key(i) for i in range(3)]
    paths = [tmp_path / encode_fs_filename(key) for key in keys]
    paths[0].write_bytes(b"first")
    first = adapter.submit_lookup_and_lock_task(keys, {})
    assert wait_for_lookup(adapter, first).get_indices_list() == [0]
    paths[0].unlink()
    paths[2].write_bytes(b"external writer")
    second = adapter.submit_lookup_and_lock_task(keys, {})
    assert wait_for_lookup(adapter, second).get_indices_list() == [2]


def test_duplicate_keys_keep_separate_result_positions(
    adapter: FSL2Adapter, tmp_path: Path
) -> None:
    """Repeated keys still have one result bit per submitted position."""
    present, missing = make_key(0), make_key(1)
    (tmp_path / encode_fs_filename(present)).write_bytes(b"present")
    task_id = adapter.submit_lookup_and_lock_task([present, missing, present], {})
    assert wait_for_lookup(adapter, task_id).get_indices_list() == [0, 2]


def test_slow_filesystem_lookup_does_not_block_other_tasks(
    adapter: FSL2Adapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blocked metadata read must leave the adapter's event loop responsive."""
    keys = [make_key(i) for i in range(4)]
    paths = [tmp_path / encode_fs_filename(key) for key in keys]
    for path in paths:
        path.write_bytes(b"data")
    entered, release = Event(), Event()
    original_stat = os.stat

    def stat_with_slow_path(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if path == paths[0] or path == str(paths[0]):
            entered.set()
            if not release.wait(timeout=10):
                raise RuntimeError("test did not release the slow filesystem read")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat_with_slow_path)
    slow = adapter.submit_lookup_and_lock_task(keys[:1], {})
    try:
        assert entered.wait(timeout=5)
        fast = adapter.submit_lookup_and_lock_task(keys[1:], {})
        assert wait_for_lookup(adapter, fast).get_indices_list() == [0, 1, 2]
        assert adapter.query_lookup_and_lock_result(slow) is None
    finally:
        release.set()
    assert wait_for_lookup(adapter, slow).get_indices_list() == [0]


def test_four_slow_batches_leave_capacity_for_another_lookup(
    adapter: FSL2Adapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lookup pool retains at least Python's minimum default concurrency."""
    slow_keys = [make_key(i) for i in range(4)]
    slow_paths = {str(tmp_path / encode_fs_filename(key)) for key in slow_keys}
    fast = make_key(4)
    (tmp_path / encode_fs_filename(fast)).write_bytes(b"data")
    entered, release = Event(), Event()
    lock = Lock()
    started: set[str] = set()
    original_stat = os.stat

    def stat_with_slow_paths(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if isinstance(path, (str, Path)) and str(path) in slow_paths:
            with lock:
                started.add(str(path))
                if started == slow_paths:
                    entered.set()
            if not release.wait(timeout=10):
                raise RuntimeError("test did not release the slow filesystem reads")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat_with_slow_paths)
    tasks = [adapter.submit_lookup_and_lock_task([key], {}) for key in slow_keys]
    try:
        assert entered.wait(timeout=5)
        task = adapter.submit_lookup_and_lock_task([fast], {})
        assert wait_for_lookup(adapter, task).get_indices_list() == [0]
    finally:
        release.set()

    for task in tasks:
        assert wait_for_lookup(adapter, task).get_indices_list() == []


def test_close_discards_an_inflight_lookup_result(
    adapter: FSL2Adapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancelled lookup stops before starting another metadata syscall."""
    keys = [make_key(i) for i in range(3)]
    paths = [tmp_path / encode_fs_filename(key) for key in keys]
    for path in paths:
        path.write_bytes(b"data")
    entered, release, closed, extra_read = Event(), Event(), Event(), Event()
    original_stat = os.stat

    def stat_with_slow_path(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if path == paths[0] or path == str(paths[0]):
            entered.set()
            if not release.wait(timeout=10):
                raise RuntimeError("test did not release the metadata read")
        elif path in paths[1:] or path in [str(p) for p in paths[1:]]:
            extra_read.set()
        return original_stat(path, *args, **kwargs)

    def close_adapter() -> None:
        adapter.close()
        closed.set()

    monkeypatch.setattr(os, "stat", stat_with_slow_path)
    closer = Thread(target=close_adapter)
    try:
        task = adapter.submit_lookup_and_lock_task(keys, {})
        assert entered.wait(timeout=5)
        closer.start()
        assert closed.wait(timeout=5)
        release.set()
        assert not extra_read.wait(timeout=0.2)
        assert adapter.query_lookup_and_lock_result(task) is None
    finally:
        release.set()
        if closer.ident is not None:
            closer.join(timeout=5)
            assert not closer.is_alive()


def test_lookup_backlog_leaves_workers_for_buffered_load(
    adapter: FSL2Adapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A saturated metadata queue must leave the data I/O executor available."""
    key = make_key(0)
    file_path = tmp_path / encode_fs_filename(key)
    file_path.write_bytes(b"data")
    blocked_path = tmp_path / encode_fs_filename(make_key(1))
    release, workers_busy = Event(), Event()
    lock = Lock()
    started = 0
    original_stat = os.stat

    def slow_stat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        nonlocal started
        if path in (blocked_path, str(blocked_path)):
            with lock:
                started += 1
                if started == 4:
                    workers_busy.set()
            assert release.wait(timeout=10)
        return original_stat(path, *args, **kwargs)

    class Buffer:
        byte_array = memoryview(bytearray(4))

    monkeypatch.setattr(os, "stat", slow_stat)
    for _ in range(40):
        adapter.submit_lookup_and_lock_task([make_key(1)], {})
    try:
        assert workers_busy.wait(timeout=5)
        buffer = Buffer()
        task_id = adapter.submit_load_task([key], [cast("MemoryObj", buffer)])
        fd = adapter.get_load_event_fd()
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        assert poller.poll(2000), "metadata queue delayed buffered load"
        result = adapter.query_load_result(task_id)
        assert result is not None
        assert result.get_indices_list() == [0]
        assert bytes(buffer.byte_array) == b"data"
    finally:
        release.set()
