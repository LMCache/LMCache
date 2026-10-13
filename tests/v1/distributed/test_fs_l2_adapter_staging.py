# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for owned FS L2 direct-I/O buffers.

Except for the explicit Linux round trip, tests remove O_DIRECT at the OS
boundary so failures depend on buffer ownership and publication semantics,
not on the filesystem backing pytest's temporary directory.
"""

# Standard
from collections.abc import Callable, Iterator
from io import FileIO
from pathlib import Path
from typing import cast
import ctypes
import errno
import math
import mmap
import os
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.memory_management import MemoryObj

pytestmark = pytest.mark.skipif(
    not hasattr(os, "statvfs"), reason="FS O_DIRECT configuration requires statvfs"
)


class Buffer:
    """Expose an unaligned ctypes byte view, as MemoryObj.byte_array can do."""

    def __init__(self, data: bytes) -> None:
        self.storage = bytearray(len(data) + 1)
        self.storage[1:] = data
        self.array = (ctypes.c_ubyte * len(data)).from_buffer(self.storage, 1)

    @property
    def byte_array(self) -> memoryview:
        """Return the non-native '<B' view used for both store and load."""
        return memoryview(self.array)


def key(index: int = 0) -> ObjectKey:
    return ObjectKey(chunk_hash=index.to_bytes(4, "big"), model_name="test", kv_rank=0)


def wait_until(predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("adapter operation did not complete within 5s")


def store(
    adapter: FSL2Adapter, keys: list[ObjectKey], bufs: list[Buffer]
) -> L2StoreResult:
    task = adapter.submit_store_task(keys, cast(list[MemoryObj], bufs))
    completed: dict[int, L2StoreResult] = {}

    def done() -> bool:
        completed.update(adapter.pop_completed_store_tasks())
        return task in completed

    wait_until(done)
    return completed[task]


def load(adapter: FSL2Adapter, keys: list[ObjectKey], bufs: list[Buffer]) -> list[int]:
    task = adapter.submit_load_task(keys, cast(list[MemoryObj], bufs))
    indices: list[int] = []

    def done() -> bool:
        result = adapter.query_load_result(task)
        if result is None:
            return False
        indices.extend(result.get_indices_list())
        return True

    wait_until(done)
    return indices


@pytest.fixture
def buffered_syscalls(monkeypatch: pytest.MonkeyPatch) -> None:
    real_open = os.open

    def open_without_direct(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        return real_open(path, flags & ~getattr(os, "O_DIRECT", 0), mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", open_without_direct)


@pytest.fixture
def adapter(tmp_path: Path, buffered_syscalls: None) -> Iterator[FSL2Adapter]:
    instance = FSL2Adapter(FSL2AdapterConfig(str(tmp_path), use_odirect=True))
    try:
        yield instance
    finally:
        instance.close()


def test_store_uses_aligned_ordinary_memory(
    adapter: FSL2Adapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = bytes(range(251)) * mmap.PAGESIZE
    source = Buffer(payload)
    source_address = ctypes.addressof(source.array)
    alignment = math.lcm(mmap.PAGESIZE, os.statvfs(tmp_path).f_bsize)
    real_write = os.write
    writes: list[tuple[int, bytes]] = []

    def inspect_write(fd: int, buf: bytes | memoryview) -> int:
        address = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        writes.append((address, bytes(buf)))
        if address == source_address or address % alignment:
            raise OSError(errno.EFAULT, "buffer is borrowed or unaligned")
        return real_write(fd, buf)

    monkeypatch.setattr(os, "write", inspect_write)
    result = store(adapter, [key()], [source])
    assert result.is_successful()
    assert result.bytes_transferred() == len(payload)
    assert len(writes) == 1
    assert writes[0][1] == payload
    assert next(tmp_path.glob("*.data")).read_bytes() == payload


def test_load_copies_back_to_ctypes_view(adapter: FSL2Adapter) -> None:
    payload = bytes(range(256)) * 32
    assert store(adapter, [key()], [Buffer(payload)]).is_successful()
    target = Buffer(b"?" * len(payload))
    assert load(adapter, [key()], [target]) == [0]
    assert bytes(target.byte_array) == payload


def test_short_write_is_not_published(
    adapter: FSL2Adapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write = os.write

    def short_write(fd: int, buf: bytes | memoryview) -> int:
        return real_write(fd, buf[:-1])

    monkeypatch.setattr(os, "write", short_write)
    result = store(adapter, [key()], [Buffer(b"x" * mmap.PAGESIZE)])
    assert not result.is_successful()
    assert result.bytes_transferred() == 0
    assert list(tmp_path.glob("*.data")) == []
    assert list(tmp_path.glob("*.tmp")) == []


def test_reused_temp_file_is_truncated(adapter: FSL2Adapter, tmp_path: Path) -> None:
    payload = b"x" * mmap.PAGESIZE
    assert store(adapter, [key()], [Buffer(payload)]).is_successful()
    final = next(tmp_path.glob("*.data"))
    adapter.delete([key()])
    final.with_suffix(".tmp").write_bytes(b"stale" * len(payload))
    assert store(adapter, [key()], [Buffer(payload)]).is_successful()
    assert final.read_bytes() == payload


def test_short_read_leaves_destination_unchanged_and_continues(
    adapter: FSL2Adapter, tmp_path: Path
) -> None:
    size = mmap.PAGESIZE * 2
    keys = [key(0), key(1)]
    assert store(
        adapter, keys, [Buffer(b"a" * size), Buffer(b"b" * size)]
    ).is_successful()
    next(tmp_path.glob(f"*{keys[0].chunk_hash.hex()}.data")).write_bytes(
        b"a" * mmap.PAGESIZE
    )
    targets = [Buffer(b"?" * size), Buffer(b"?" * size)]
    assert load(adapter, keys, targets) == [1]
    assert bytes(targets[0].byte_array) == b"?" * size
    assert bytes(targets[1].byte_array) == b"b" * size


def test_close_during_read_does_not_write_released_destination(
    tmp_path: Path, buffered_syscalls: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = FSL2Adapter(FSL2AdapterConfig(str(tmp_path), use_odirect=True))
    size = mmap.PAGESIZE
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    owners: list[mmap.mmap] = []
    errors: list[Exception] = []
    real_fdopen = os.fdopen
    closed = False

    class BlockingReader:
        def __init__(self, file: FileIO) -> None:
            self.file = file

        def __enter__(self) -> "BlockingReader":
            return self

        def __exit__(self, *args: object) -> None:
            self.file.close()

        def readinto(self, buf: memoryview) -> int | None:
            owners.append(cast(mmap.mmap, buf.obj))
            started.set()
            try:
                if not release.wait(5):
                    raise AssertionError("test did not release the blocked read")
                return self.file.readinto(buf)
            except Exception as exc:
                errors.append(exc)
                raise
            finally:
                # Drop this derived view before the worker closes its mmap.
                buf.release()
                finished.set()

    def blocking_fdopen(fd: int, mode: str, buffering: int) -> BlockingReader:
        return BlockingReader(cast(FileIO, real_fdopen(fd, mode, buffering)))

    try:
        assert store(adapter, [key()], [Buffer(b"a" * size)]).is_successful()
        monkeypatch.setattr(os, "fdopen", blocking_fdopen)
        target = Buffer(b"?" * size)
        adapter.submit_load_task([key()], cast(list[MemoryObj], [target]))
        assert started.wait(5)
        adapter.close()
        closed = True
        with target.byte_array.cast("B") as reused:
            reused[:] = b"z" * size
        release.set()
        assert finished.wait(5)
        assert errors == []
        wait_until(lambda: all(owner.closed for owner in owners))
        assert bytes(target.byte_array) == b"z" * size
        assert owners
    finally:
        release.set()
        if not closed:
            adapter.close()


def test_close_during_store_preserves_source_snapshot(
    tmp_path: Path, buffered_syscalls: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = FSL2Adapter(FSL2AdapterConfig(str(tmp_path), use_odirect=True))
    payload = b"a" * mmap.PAGESIZE
    source = Buffer(payload)
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    owners: list[mmap.mmap] = []
    writes: list[bytes] = []
    real_mmap = mmap.mmap
    real_write = os.write
    closed = False

    def blocking_mmap(fileno: int, length: int) -> mmap.mmap:
        started.set()
        if not release.wait(5):
            raise AssertionError("test did not release the blocked store")
        owner = real_mmap(fileno, length)
        owners.append(owner)
        return owner

    def record_write(fd: int, buf: bytes | memoryview) -> int:
        writes.append(bytes(buf))
        try:
            return real_write(fd, buf)
        finally:
            finished.set()

    monkeypatch.setattr(mmap, "mmap", blocking_mmap)
    monkeypatch.setattr(os, "write", record_write)
    try:
        adapter.submit_store_task([key()], cast(list[MemoryObj], [source]))
        assert started.wait(5)
        adapter.close()
        closed = True
        with source.byte_array.cast("B") as reused:
            reused[:] = b"z" * len(payload)
        release.set()
        assert finished.wait(5)
        wait_until(lambda: bool(owners) and all(owner.closed for owner in owners))
        assert writes == [payload]
        assert list(tmp_path.glob("*.data")) == []
    finally:
        release.set()
        if not closed:
            adapter.close()


def test_buffered_store_and_load(tmp_path: Path) -> None:
    adapter = FSL2Adapter(FSL2AdapterConfig(str(tmp_path), use_odirect=False))
    try:
        payload = b"buffered payload"
        assert store(adapter, [key()], [Buffer(payload)]).is_successful()
        target = Buffer(b"?" * len(payload))
        assert load(adapter, [key()], [target]) == [0]
        assert bytes(target.byte_array) == payload
    finally:
        adapter.close()


@pytest.mark.skipif(not hasattr(os, "O_DIRECT"), reason="requires Linux O_DIRECT")
def test_linux_odirect_round_trip(tmp_path: Path) -> None:
    # Establish filesystem support with an independently aligned buffer.
    probe = tmp_path / "probe"
    alignment = math.lcm(mmap.PAGESIZE, os.statvfs(tmp_path).f_bsize)
    fd = os.open(probe, os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        with mmap.mmap(-1, alignment * 2) as memory:
            start = (-ctypes.addressof(ctypes.c_char.from_buffer(memory))) % alignment
            with memoryview(memory)[start : start + alignment] as view:
                fd = os.open(probe, os.O_WRONLY | os.O_DIRECT)
                try:
                    assert os.write(fd, view) == alignment
                finally:
                    os.close(fd)
    except OSError as exc:
        if exc.errno in (errno.EINVAL, errno.EOPNOTSUPP):
            pytest.skip(f"temporary filesystem rejects O_DIRECT: {exc}")
        raise
    finally:
        probe.unlink()

    adapter = FSL2Adapter(FSL2AdapterConfig(str(tmp_path), use_odirect=True))
    try:
        payload = bytes(range(256)) * (9 * 1024 * 1024 // 256)
        assert store(adapter, [key()], [Buffer(payload)]).is_successful()
        target = Buffer(b"?" * len(payload))
        assert load(adapter, [key()], [target]) == [0]
        assert bytes(target.byte_array) == payload
    finally:
        adapter.close()
