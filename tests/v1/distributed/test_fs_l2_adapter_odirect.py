# SPDX-License-Identifier: Apache-2.0
"""Tests for FSL2Adapter's O_DIRECT fallback.

Whether ``O_DIRECT`` is usable depends on the file system, the device
stack and the alignment of the buffers handed to the adapter, none of
which can be checked up front.  When a direct write is rejected the
adapter must degrade to buffered I/O instead of failing that store and
every store after it, which would leave the L2 tier permanently dead
while the server keeps reporting no fatal error.
"""

# Standard
from collections.abc import Iterator
from pathlib import Path
from typing import cast
import errno
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.memory_management import MemoryObj


class _Buf:
    """Minimal MemoryObj stand-in: just the ``byte_array`` the FS
    adapter's store path reads."""

    def __init__(self, data: bytes) -> None:
        self._data = bytearray(data)

    @property
    def byte_array(self) -> memoryview:
        return memoryview(self._data)


def _key(h: bytes) -> ObjectKey:
    return ObjectKey(
        chunk_hash=h,
        model_name="llama",
        kv_rank=42,
        cache_salt="alice",
    )


@pytest.fixture
def odirect_adapter(tmp_path: Path) -> Iterator[FSL2Adapter]:
    adp = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path), use_odirect=True))
    try:
        yield adp
    finally:
        adp.close()


def _store_and_wait(
    adp: FSL2Adapter, keys: list[ObjectKey], payloads: list[bytes]
) -> None:
    """Submit a store and poll until its result is available."""
    objs = cast("list[MemoryObj]", [_Buf(p) for p in payloads])
    task_id = adp.submit_store_task(keys, objs)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completed = adp.pop_completed_store_tasks()
        if task_id in completed:
            # L2StoreResult is an int: >= 0 success, -1 failure.
            assert int(completed[task_id]) >= 0
            return
        time.sleep(0.01)
    pytest.fail("store task did not complete within 5s")


class TestODirectFallback:
    def test_rejected_odirect_write_falls_back_to_buffered(
        self, odirect_adapter: FSL2Adapter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A kernel that refuses the direct write must not kill the tier.

        The chunk is retried buffered, O_DIRECT is dropped for the rest
        of the process, and later chunks never attempt it again.
        """
        adp = odirect_adapter
        # Keep the payload block-aligned so the size check does not
        # opt out of O_DIRECT before the write is even attempted.
        block = adp._os_disk_bs or 4096
        payload_a = b"a" * (block * 2)
        payload_b = b"b" * (block * 2)

        attempts: list[Path] = []

        def _reject(file_path: Path, buf: bytes) -> None:
            attempts.append(file_path)
            raise OSError(errno.EINVAL, "Invalid argument")

        monkeypatch.setattr(adp, "_write_with_odirect", _reject)

        k1 = _key(b"\x01")
        k2 = _key(b"\x02")
        _store_and_wait(adp, [k1], [payload_a])
        _store_and_wait(adp, [k2], [payload_b])

        assert adp._key_to_path(k1).read_bytes() == payload_a
        assert adp._key_to_path(k2).read_bytes() == payload_b
        # Only the first chunk pays for the failed attempt.
        assert len(attempts) == 1
        assert adp._use_odirect is False

    def test_successful_odirect_write_keeps_odirect_enabled(
        self, odirect_adapter: FSL2Adapter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fallback must not fire when direct I/O works."""
        adp = odirect_adapter
        block = adp._os_disk_bs or 4096
        payload = b"c" * (block * 2)

        calls: list[Path] = []
        original = adp._write_with_odirect

        def _record(file_path: Path, buf: bytes) -> None:
            calls.append(file_path)
            original(file_path, buf)

        monkeypatch.setattr(adp, "_write_with_odirect", _record)

        k = _key(b"\x03")
        _store_and_wait(adp, [k], [payload])

        assert adp._key_to_path(k).read_bytes() == payload
        assert len(calls) == 1
        assert adp._use_odirect is True
