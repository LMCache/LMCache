# SPDX-License-Identifier: Apache-2.0
"""Public filesystem load behavior, including short reads and worker lifetime."""

# Standard
from collections.abc import Callable
from pathlib import Path
from typing import cast
import builtins
import io
import os
import select
import threading

# Third Party
from typing_extensions import Buffer
import aiofiles.threadpool
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.key_codec import encode_fs_filename
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.memory_management import MemoryObj


class _Buffer:
    def __init__(self, size: int) -> None:
        self.storage = bytearray(b"!" * (size + 2))

    @property
    def byte_array(self) -> memoryview:
        return memoryview(self.storage)[1:-1]


def _key(index: int) -> ObjectKey:
    """Return a distinct object identity for a file."""
    return ObjectKey(chunk_hash=bytes([index]), model_name="load", kv_rank=0)


def _wait_result(adapter: FSL2Adapter, task_id: int) -> Bitmap:
    """Wait for a public completion notification and consume its result."""
    poll = select.poll()
    fd = adapter.get_load_event_fd()
    poll.register(fd, select.POLLIN)
    assert poll.poll(5000), "load completion timed out"
    os.read(fd, 8)
    result = adapter.query_load_result(task_id)
    assert result is not None
    assert adapter.query_load_result(task_id) is None
    return result


def _patch_reader(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[io.BufferedReader], io.BufferedReader],
) -> None:
    """Substitute real file readers for both synchronous and aiofiles opens."""
    real_open = builtins.open

    def open_file(path: Path, mode: str = "rb", **kwargs: object) -> io.BufferedReader:
        assert mode == "rb"
        return factory(real_open(path, "rb"))

    monkeypatch.setattr(builtins, "open", open_file)
    monkeypatch.setattr(aiofiles.threadpool, "sync_open", open_file)


@pytest.mark.parametrize("read_ahead", [None, 1, 19, 20, 21])
@pytest.mark.parametrize("size", [0, 1, 19, 20, 21, 65537])
def test_load_windows_and_eof(
    tmp_path: Path, read_ahead: int | None, size: int
) -> None:
    """Only full files hit; offset destination guards and key order survive."""
    payload = bytes(i % 251 for i in range(size))
    keys = [_key(i) for i in range(4)]
    (tmp_path / encode_fs_filename(keys[0])).write_bytes(payload)
    (tmp_path / encode_fs_filename(keys[2])).write_bytes(payload[:-1])
    (tmp_path / encode_fs_filename(keys[3])).write_bytes(payload)
    buffers = [_Buffer(size) for _ in keys]
    adapter = FSL2Adapter(
        FSL2AdapterConfig(base_path=str(tmp_path), read_ahead_size=read_ahead)
    )
    try:
        task = adapter.submit_load_task(keys, cast("list[MemoryObj]", buffers))
        result = _wait_result(adapter, task)
        assert result.get_indices_list() == ([0, 2, 3] if size == 0 else [0, 3])
        for index in [0, 3]:
            assert bytes(buffers[index].byte_array) == payload
        for buf in buffers:
            assert buf.storage[:1] == buf.storage[-1:] == b"!"
    finally:
        adapter.close()


@pytest.mark.parametrize("read_ahead", [None, 7, 100])
def test_short_reads_fill_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, read_ahead: int | None
) -> None:
    """A read that returns fewer bytes than requested is retried until EOF."""

    class ShortReader(io.BufferedReader):
        def readinto(self, buf: Buffer) -> int:
            return super().readinto(memoryview(buf)[:3])

    payload = bytes(range(40))
    key = _key(0)
    (tmp_path / encode_fs_filename(key)).write_bytes(payload)
    _patch_reader(monkeypatch, lambda f: ShortReader(f.detach()))
    adapter = FSL2Adapter(
        FSL2AdapterConfig(base_path=str(tmp_path), read_ahead_size=read_ahead)
    )
    dst = _Buffer(len(payload))
    try:
        task = adapter.submit_load_task([key], cast("list[MemoryObj]", [dst]))
        assert _wait_result(adapter, task).get_indices_list() == [0]
        assert bytes(dst.byte_array) == payload
    finally:
        adapter.close()


def test_read_failure_does_not_abort_other_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An I/O failure is a miss, while later keys still load and files close."""
    closed: list[str] = []

    class FailingReader(io.BufferedReader):
        def readinto(self, buf: Buffer) -> int:
            if str(self.name).endswith("00.data"):
                raise OSError("injected read failure")
            return super().readinto(buf)

        def close(self) -> None:
            closed.append(str(self.name))
            super().close()

    keys = [_key(0), _key(1)]
    for key in keys:
        (tmp_path / encode_fs_filename(key)).write_bytes(b"abc")
    _patch_reader(monkeypatch, lambda f: FailingReader(f.detach()))
    adapter = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    try:
        dst = [_Buffer(3), _Buffer(3)]
        task = adapter.submit_load_task(keys, cast("list[MemoryObj]", dst))
        assert _wait_result(adapter, task).get_indices_list() == [1]
        assert bytes(dst[1].byte_array) == b"abc"
        assert len(closed) == 2
    finally:
        adapter.close()


def test_close_waits_for_inflight_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returning from close guarantees no worker can still write to a buffer."""
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    closed = threading.Event()

    class BlockingReader(io.BufferedReader):
        def readinto(self, buf: Buffer) -> int:
            entered.set()
            assert release.wait(5)
            n = super().readinto(buf)
            finished.set()
            return n

    key = _key(0)
    (tmp_path / encode_fs_filename(key)).write_bytes(b"abc")
    _patch_reader(monkeypatch, lambda f: BlockingReader(f.detach()))
    adapter = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    dst = _Buffer(3)
    adapter.submit_load_task([key], cast("list[MemoryObj]", [dst]))
    closer: threading.Thread | None = None

    def close() -> None:
        adapter.close()
        closed.set()

    try:
        assert entered.wait(5)
        closer = threading.Thread(target=close)
        closer.start()
        assert not closed.wait(0.1)
        release.set()
        assert closed.wait(5)
        assert finished.is_set()
        assert bytes(dst.byte_array) == b"abc"
    finally:
        release.set()
        if closer is not None:
            closer.join(timeout=5)
        else:
            adapter.close()
