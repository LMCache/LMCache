# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for the native libblkio storage connector.

Tests exercise the C++ BlkioConnector through the Python BlkioClient,
verifying write→read→verify data integrity via libblkio's io_uring
backend.

Device selection (checked in order):
  1. LMCACHE_BLKIO_TEST_DEVICE env var  — real block device path
  2. Auto-created loopback device        — requires root (losetup)
  3. Sparse temp file                    — always available (no direct I/O)

All tests are skipped if the C++ extension ``lmcache.lmcache_blkio``
cannot be imported.
"""

# Future
from __future__ import annotations

# Standard
import asyncio
import os
import subprocess
import tempfile
import threading

# Third Party
import pytest


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------


def _has_blkio_ext() -> bool:
    """Check if the C++ blkio extension is importable."""
    try:
        from lmcache.lmcache_blkio import LMCacheBlkioClient  # noqa: F401

        return True
    except ImportError:
        return False


requires_blkio = pytest.mark.skipif(
    not _has_blkio_ext(),
    reason="C++ libblkio extension (lmcache_blkio) not available",
)


# ---------------------------------------------------------------------------
# Device provisioning helpers
# ---------------------------------------------------------------------------

# 4 MB — small enough for fast CI, large enough for meaningful I/O
_DEVICE_SIZE_BYTES = 4 * 1024 * 1024
_BLOCK_SIZE = 4096


class _LoopDevice:
    """RAII wrapper around a losetup-backed loopback device."""

    def __init__(self, size_bytes: int):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".blkio_test", delete=False)
        self._tmp.truncate(size_bytes)
        self._tmp.close()
        self._loop_path: str | None = None

        try:
            result = subprocess.run(
                ["losetup", "-f", "--show", self._tmp.name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self._loop_path = result.stdout.strip()
        except Exception:
            pass

    @property
    def path(self) -> str | None:
        return self._loop_path

    def close(self) -> None:
        if self._loop_path:
            try:
                subprocess.run(
                    ["losetup", "-d", self._loop_path],
                    capture_output=True,
                    timeout=5,
                )
            except Exception:
                pass
            self._loop_path = None
        try:
            os.unlink(self._tmp.name)
        except FileNotFoundError:
            pass


class _TempFileDevice:
    """Fallback: a sparse file that libblkio can open via io_uring."""

    def __init__(self, size_bytes: int):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".blkio_test", delete=False)
        self._tmp.truncate(size_bytes)
        self._tmp.close()

    @property
    def path(self) -> str:
        return self._tmp.name

    def close(self) -> None:
        try:
            os.unlink(self._tmp.name)
        except FileNotFoundError:
            pass


def _resolve_device() -> tuple[str, callable, bool]:
    """Resolve a test device using the priority order in the docstring.

    Returns:
        (device_path, cleanup_callable, use_direct_io)
    """
    # 1. Explicit env var
    env_dev = os.environ.get("LMCACHE_BLKIO_TEST_DEVICE")
    if env_dev:
        return env_dev, lambda: None, True

    # 2. Loopback (requires root)
    if os.geteuid() == 0:
        loop = _LoopDevice(_DEVICE_SIZE_BYTES)
        if loop.path:
            return loop.path, loop.close, True

    # 3. Sparse temp file (no direct I/O — files may not support it)
    tmp = _TempFileDevice(_DEVICE_SIZE_BYTES)
    return tmp.path, tmp.close, False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def loop_in_thread():
    """Run an asyncio event loop in a background thread."""
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, name="blkio-test-loop", daemon=True)
    t.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=5)
        loop.close()


@pytest.fixture
def blkio_device():
    """Provide a block device (or file) for testing."""
    path, cleanup, direct_io = _resolve_device()
    try:
        yield path, direct_io
    finally:
        cleanup()


# ---------------------------------------------------------------------------
# Tests — C++ connector via Python client
# ---------------------------------------------------------------------------


@requires_blkio
class TestBlkioConnectorDirect:
    """Direct tests against ``LMCacheBlkioClient`` (raw pybind API)."""

    def test_construct_and_close(self, blkio_device):
        """Connector can be created and cleanly shut down."""
        from lmcache.lmcache_blkio import LMCacheBlkioClient

        device_path, direct_io = blkio_device
        client = LMCacheBlkioClient(device_path, 2, direct_io)
        assert client.event_fd() >= 0
        client.close()

    def test_event_fd_is_valid(self, blkio_device):
        """event_fd returns a non-negative file descriptor."""
        from lmcache.lmcache_blkio import LMCacheBlkioClient

        device_path, direct_io = blkio_device
        client = LMCacheBlkioClient(device_path, 1, direct_io)
        try:
            fd = client.event_fd()
            assert isinstance(fd, int)
            assert fd >= 0
        finally:
            client.close()

    def test_write_read_verify(self, blkio_device):
        """Write a buffer, read it back, verify contents match."""
        from lmcache.lmcache_blkio import LMCacheBlkioClient

        device_path, direct_io = blkio_device
        client = LMCacheBlkioClient(device_path, 2, direct_io)

        try:
            size = _BLOCK_SIZE
            offset_hex = format(0, "x")
            key = f"test_model@00000000@{offset_hex}"

            # Write buffer filled with 0xAB
            write_buf = bytearray(b"\xab" * size)
            write_mv = memoryview(write_buf)

            fid = client.submit_batch_set([key], [write_mv])
            assert isinstance(fid, int)

            # Drain completion
            completions = _drain_until(client, fid)
            assert len(completions) == 1
            fid_out, ok, error, _ = completions[0]
            assert fid_out == fid
            assert ok, f"SET failed: {error}"

            # Read into a zero-filled buffer
            read_buf = bytearray(size)
            read_mv = memoryview(read_buf)

            fid2 = client.submit_batch_get([key], [read_mv])
            completions2 = _drain_until(client, fid2)
            assert len(completions2) == 1
            fid_out2, ok2, error2, result_bools = completions2[0]
            assert fid_out2 == fid2
            assert ok2, f"GET failed: {error2}"

            # Verify
            assert read_buf == write_buf, "Read data does not match written data"
        finally:
            client.close()

    def test_write_read_distinct_patterns(self, blkio_device):
        """Write pattern1, overwrite write buffer with pattern2,
        read back, confirm device holds pattern1."""
        from lmcache.lmcache_blkio import LMCacheBlkioClient

        device_path, direct_io = blkio_device
        client = LMCacheBlkioClient(device_path, 1, direct_io)

        try:
            size = _BLOCK_SIZE
            offset_hex = format(_BLOCK_SIZE, "x")  # second block
            key = f"test_model@00000000@{offset_hex}"

            pattern1 = b"\x55" * size
            pattern2 = b"\xaa" * size

            write_buf = bytearray(pattern1)
            fid = client.submit_batch_set([key], [memoryview(write_buf)])
            comps = _drain_until(client, fid)
            assert comps[0][1], f"SET failed: {comps[0][2]}"

            # Overwrite the write buffer
            write_buf[:] = pattern2

            # Read back — should get pattern1
            read_buf = bytearray(size)
            fid2 = client.submit_batch_get([key], [memoryview(read_buf)])
            comps2 = _drain_until(client, fid2)
            assert comps2[0][1], f"GET failed: {comps2[0][2]}"

            assert read_buf == bytearray(pattern1), (
                "Read data should match original write, not the overwritten buffer"
            )
        finally:
            client.close()

    def test_batch_write_read(self, blkio_device):
        """Batch write multiple keys, read them all back."""
        from lmcache.lmcache_blkio import LMCacheBlkioClient

        device_path, direct_io = blkio_device
        client = LMCacheBlkioClient(device_path, 2, direct_io)

        try:
            n = 4
            size = _BLOCK_SIZE
            keys = []
            write_bufs = []
            for i in range(n):
                offset = i * size
                keys.append(f"test_model@00000000@{format(offset, 'x')}")
                buf = bytearray(bytes([i & 0xFF]) * size)
                write_bufs.append(buf)

            write_mvs = [memoryview(b) for b in write_bufs]
            fid = client.submit_batch_set(keys, write_mvs)
            comps = _drain_until(client, fid)
            assert comps[0][1], f"Batch SET failed: {comps[0][2]}"

            read_bufs = [bytearray(size) for _ in range(n)]
            read_mvs = [memoryview(b) for b in read_bufs]
            fid2 = client.submit_batch_get(keys, read_mvs)
            comps2 = _drain_until(client, fid2)
            assert comps2[0][1], f"Batch GET failed: {comps2[0][2]}"

            for i in range(n):
                assert read_bufs[i] == write_bufs[i], (
                    f"Data mismatch at batch index {i}"
                )
        finally:
            client.close()

    def test_multiple_workers(self, blkio_device):
        """Connector works with multiple worker threads."""
        from lmcache.lmcache_blkio import LMCacheBlkioClient

        device_path, direct_io = blkio_device
        for num_workers in [1, 2, 4]:
            client = LMCacheBlkioClient(device_path, num_workers, direct_io)
            try:
                size = _BLOCK_SIZE
                key = "test_model@00000000@0"
                buf = bytearray(b"\x42" * size)
                fid = client.submit_batch_set([key], [memoryview(buf)])
                comps = _drain_until(client, fid)
                assert comps[0][1], (
                    f"SET with {num_workers} workers failed: {comps[0][2]}"
                )
            finally:
                client.close()


# ---------------------------------------------------------------------------
# Tests — Python BlkioClient wrapper
# ---------------------------------------------------------------------------


@requires_blkio
class TestBlkioClientWrapper:
    """Tests for the Python ``BlkioClient`` async wrapper."""

    def test_construct_and_close(self, blkio_device, loop_in_thread):
        """Client can be created on a background loop and closed."""
        device_path, direct_io = blkio_device

        fut = asyncio.run_coroutine_threadsafe(
            _async_construct_client(device_path, direct_io, loop_in_thread),
            loop_in_thread,
        )
        client = fut.result(timeout=10)
        assert client is not None
        client.close()

    def test_sync_set_get_roundtrip(self, blkio_device, loop_in_thread):
        """Synchronous set/get roundtrip through BlkioClient."""
        device_path, direct_io = blkio_device

        fut = asyncio.run_coroutine_threadsafe(
            _async_construct_client(device_path, direct_io, loop_in_thread),
            loop_in_thread,
        )
        client = fut.result(timeout=10)

        try:
            size = _BLOCK_SIZE
            key = "test_model@00000000@0"

            write_buf = bytearray(b"\xcd" * size)
            client.set_sync(key, memoryview(write_buf))

            read_buf = bytearray(size)
            client.get_sync(key, memoryview(read_buf))

            assert read_buf == write_buf
        finally:
            client.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _async_construct_client(
    device_path: str, direct_io: bool, loop: asyncio.AbstractEventLoop
) -> object:
    """Construct a BlkioClient on the given event loop."""
    from lmcache.v1.storage_backend.native_clients.blkio_client import (
        BlkioClient,
    )

    return BlkioClient(
        device_path=device_path,
        num_workers=2,
        direct_io=direct_io,
        loop=loop,
    )


def _drain_until(client: object, target_fid: int, timeout_s: float = 10.0) -> list:
    """Poll drain_completions until we see target_fid or timeout."""
    import select
    import time

    fd = client.event_fd()
    deadline = time.monotonic() + timeout_s
    collected = []

    while time.monotonic() < deadline:
        remaining = max(0, deadline - time.monotonic())
        poll = select.poll()
        poll.register(fd, select.POLLIN)
        events = poll.poll(remaining * 1000)

        if events:
            items = client.drain_completions()
            for item in items:
                collected.append(item)
                if item[0] == target_fid:
                    return collected

    raise TimeoutError(f"Timed out waiting for completion of future {target_fid}")
