# SPDX-License-Identifier: Apache-2.0
"""Tests for FSL2Adapter's O_DIRECT path.

O_DIRECT needs the length, the offset and the buffer address aligned to the
device's logical block size. When a buffer is not aligned, or the filesystem
refuses the flag, the adapter must fall back to buffered I/O rather than fail
the store or turn the load into a miss. See
https://github.com/LMCache/LMCache/issues/4364.
"""

# Standard
from collections.abc import Iterator
from pathlib import Path
from typing import cast
import ctypes
import errno
import fcntl
import os
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.l2_adapters import fs_l2_adapter
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
    _buffer_is_aligned,
)
from lmcache.v1.memory_management import MemoryObj

pytestmark = pytest.mark.skipif(
    not hasattr(os, "O_DIRECT"), reason="O_DIRECT is not available"
)

_ALIGN = 4096
_SIZE = 2 * _ALIGN


class _Buf:
    """MemoryObj stand-in whose ``byte_array`` starts at a chosen offset
    from a 4096-aligned address, so a test controls the buffer address."""

    def __init__(self, data: bytes, misalign: int = 0) -> None:
        self._raw = bytearray(len(data) + 2 * _ALIGN)
        base = ctypes.addressof((ctypes.c_byte * 1).from_buffer(self._raw))
        start = (-base) % _ALIGN + misalign
        self._view = memoryview(self._raw)[start : start + len(data)]
        self._view[:] = data

    @property
    def byte_array(self) -> memoryview:
        return self._view


def _key(h: bytes = b"\xde\xad\xbe\xef") -> ObjectKey:
    return ObjectKey(chunk_hash=h, model_name="llama", kv_rank=0)


@pytest.fixture
def adapter(tmp_path: Path) -> Iterator[FSL2Adapter]:
    adp = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path), use_odirect=True))
    try:
        yield adp
    finally:
        adp.close()


def _is_odirect_fd(fd: int) -> bool:
    return bool(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_DIRECT)


@pytest.fixture
def real_odirect(tmp_path: Path) -> None:
    """Skip unless this filesystem accepts an aligned O_DIRECT write."""
    probe = tmp_path / "probe"
    buf = _Buf(b"p" * _ALIGN).byte_array
    try:
        fd = os.open(probe, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
        try:
            os.write(fd, buf)
        finally:
            os.close(fd)
    except OSError as e:
        pytest.skip(f"filesystem refuses O_DIRECT: {e}")
    finally:
        probe.unlink(missing_ok=True)


@pytest.fixture
def odirect_opens(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the paths opened with O_DIRECT."""
    opened: list[str] = []
    real_open = os.open

    def spy(path, flags, *args, **kwargs):
        if flags & os.O_DIRECT:
            opened.append(os.path.basename(path))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", spy)
    return opened


def _wait(poll, what: str):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        result = poll()
        if result is not None:
            return result
        time.sleep(0.01)
    pytest.fail(f"{what} did not complete within 5s")


def _store(adp: FSL2Adapter, key: ObjectKey, buf: _Buf) -> int:
    task_id = adp.submit_store_task([key], cast("list[MemoryObj]", [buf]))
    result = _wait(lambda: adp.pop_completed_store_tasks().get(task_id), "store")
    return int(result)


def _load(adp: FSL2Adapter, key: ObjectKey, buf: _Buf) -> bool:
    task_id = adp.submit_load_task([key], cast("list[MemoryObj]", [buf]))
    bitmap = _wait(lambda: adp.query_load_result(task_id), "load")
    return bitmap.test(0)


def _payload() -> bytes:
    return bytes(range(256)) * (_SIZE // 256)


class TestBufferAlignment:
    def test_aligned_and_misaligned_addresses(self) -> None:
        assert _buffer_is_aligned(_Buf(b"x" * _ALIGN).byte_array, _ALIGN)
        # 64 mod 512 is what an unaligned torch.empty() base produces.
        assert not _buffer_is_aligned(_Buf(b"x" * _ALIGN, 64).byte_array, 512)

    def test_read_only_buffer_counts_as_unaligned(self) -> None:
        assert not _buffer_is_aligned(b"x" * _ALIGN, 1)


class TestBlockSize:
    def test_uses_device_logical_block_size(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fs_l2_adapter, "_device_logical_block_size", lambda p: 512)
        adp = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path), use_odirect=True))
        try:
            assert adp._os_disk_bs == 512
        finally:
            adp.close()

    def test_falls_back_to_statvfs_without_a_block_device(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fs_l2_adapter, "_device_logical_block_size", lambda p: 0)
        adp = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path), use_odirect=True))
        try:
            assert adp._os_disk_bs == os.statvfs(tmp_path).f_bsize
        finally:
            adp.close()


class TestStoreLoad:
    def test_aligned_buffer_uses_odirect(
        self, adapter: FSL2Adapter, odirect_opens: list[str], real_odirect: None
    ) -> None:
        if _ALIGN % adapter._os_disk_bs:
            pytest.skip("block size larger than the test buffers")
        data = _payload()
        assert _store(adapter, _key(), _Buf(data)) >= 0
        out = _Buf(b"\x00" * _SIZE)
        assert _load(adapter, _key(), out)
        assert bytes(out.byte_array) == data
        # the store's temp file and the load (the probe fixture opens one too)
        assert [n for n in odirect_opens if n != "probe"] == [
            "llama@0x00000000@0@deadbeef.tmp",
            "llama@0x00000000@0@deadbeef.data",
        ]

    def test_misaligned_address_falls_back_to_buffered(
        self, adapter: FSL2Adapter, odirect_opens: list[str]
    ) -> None:
        """The length is aligned but the address is not, as with a lazy L1
        allocator whose base is only 64-byte aligned."""
        data = _payload()
        assert _store(adapter, _key(), _Buf(data, misalign=64)) >= 0
        out = _Buf(b"\x00" * _SIZE, misalign=64)
        assert _load(adapter, _key(), out)
        assert bytes(out.byte_array) == data
        assert odirect_opens == []

    @pytest.mark.parametrize("err", [errno.EINVAL, errno.EOPNOTSUPP])
    def test_refused_at_open_falls_back_to_buffered(
        self, adapter: FSL2Adapter, monkeypatch: pytest.MonkeyPatch, err: int
    ) -> None:
        """A filesystem that refuses O_DIRECT must not lose the store or turn
        the load into a miss."""
        real_open = os.open

        def refuse(path, flags, *args, **kwargs):
            if flags & os.O_DIRECT:
                raise OSError(err, os.strerror(err))
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", refuse)
        data = _payload()
        assert _store(adapter, _key(), _Buf(data)) >= 0
        out = _Buf(b"\x00" * _SIZE)
        assert _load(adapter, _key(), out)
        assert bytes(out.byte_array) == data

    @pytest.mark.parametrize("partial", [False, True])
    def test_refused_at_write_falls_back_to_buffered(
        self,
        adapter: FSL2Adapter,
        monkeypatch: pytest.MonkeyPatch,
        real_odirect: None,
        partial: bool,
    ) -> None:
        """open() accepts O_DIRECT and write() refuses it, the failure seen on
        XFS in #4364, optionally after part of the object was written."""
        if _ALIGN % adapter._os_disk_bs:
            pytest.skip("block size larger than the test buffers")
        real_write = os.write

        def refuse(fd, buf):
            if not _is_odirect_fd(fd):
                return real_write(fd, buf)
            if partial and len(buf) == _SIZE:
                return real_write(fd, memoryview(buf)[:_ALIGN])
            raise OSError(errno.EINVAL, "Invalid argument")

        monkeypatch.setattr(os, "write", refuse)
        data = _payload()
        assert _store(adapter, _key(), _Buf(data)) >= 0
        assert adapter._key_to_path(_key()).read_bytes() == data

    def test_short_write_is_completed(
        self,
        adapter: FSL2Adapter,
        monkeypatch: pytest.MonkeyPatch,
        real_odirect: None,
    ) -> None:
        """A short O_DIRECT write must not publish a truncated file."""
        if _ALIGN % adapter._os_disk_bs:
            pytest.skip("block size larger than the test buffers")
        real_write = os.write
        calls = []

        def short_first(fd, buf):
            if not _is_odirect_fd(fd):
                return real_write(fd, buf)
            calls.append(len(buf))
            if len(calls) == 1:
                return real_write(fd, memoryview(buf)[:_ALIGN])
            return real_write(fd, buf)

        monkeypatch.setattr(os, "write", short_first)
        data = _payload()
        assert _store(adapter, _key(), _Buf(data)) >= 0
        assert calls[:2] == [_SIZE, _SIZE - _ALIGN]
        path = adapter._key_to_path(_key())
        assert path.read_bytes() == data

    def test_unaligned_buffers_warn_once(
        self, adapter: FSL2Adapter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        warnings: list[str] = []
        monkeypatch.setattr(
            fs_l2_adapter.logger, "warning", lambda msg, *a: warnings.append(msg)
        )
        for i in range(3):
            assert _store(adapter, _key(bytes([i])), _Buf(_payload(), 64)) >= 0
        assert len(warnings) == 1

    def test_missing_file_is_a_miss(self, adapter: FSL2Adapter) -> None:
        assert not _load(adapter, _key(b"\x01"), _Buf(b"\x00" * _SIZE))

    def test_lookup_after_odirect_store(self, adapter: FSL2Adapter) -> None:
        assert _store(adapter, _key(), _Buf(_payload())) >= 0
        task_id = adapter.submit_lookup_and_lock_task(
            [_key()], {0: MemoryLayoutDesc(shapes=[], dtypes=[])}
        )
        bitmap = _wait(lambda: adapter.query_lookup_and_lock_result(task_id), "lookup")
        assert bitmap.test(0)
