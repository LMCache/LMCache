# SPDX-License-Identifier: Apache-2.0
"""
Tests for the Rust ``RawBlockDevice`` with ``io_engine='libblkio'``.

Three levels of coverage:

1. **Unit / smoke tests** — verify open, size_bytes, pwrite/pread roundtrip,
   close, and error handling using a sparse temp file (always available).

2. **Full backend integration** — exercise the ``RustRawBlockBackend`` plugin
   with ``io_engine='libblkio'``, covering put/get roundtrip, eviction, and
   checkpoint recovery through the real KV-cache storage path.

3. **O_DIRECT on a loopback block device** — exercises alignment-sensitive
   paths (bounce buffers, O_DIRECT semantics).  Requires root or a
   pre-created device via ``LMCACHE_BLKIO_TEST_DEVICE``.

All tests are skipped when the Rust extension was built without the
``blkio`` cargo feature (``io_engine='libblkio'`` not accepted).
"""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import Future
import asyncio
import os
import subprocess
import tempfile
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import AdHocMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.plugins.rust_raw_block_backend import (
    RustRawBlockBackend,
)


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------


def _has_rust_ext() -> bool:
    """Check if the base Rust raw-block extension is importable."""
    try:
        import lmcache_rust_raw_block_io  # noqa: F401

        return True
    except ImportError:
        return False


def _has_blkio_feature() -> bool:
    """Check if the Rust extension was built with the ``blkio`` feature.

    Creates a tiny temp file and attempts to open it with
    ``io_engine='libblkio'``.  Returns ``True`` when the engine is
    accepted, ``False`` if the extension rejects the engine name or is
    not installed.
    """
    try:
        from lmcache_rust_raw_block_io import RawBlockDevice  # noqa: F401

        with tempfile.NamedTemporaryFile(suffix=".blkio_probe") as f:
            f.truncate(4096)
            f.flush()
            dev = RawBlockDevice(
                f.name, writable=False, io_engine="libblkio"
            )
            dev.close()
        return True
    except (ImportError, ValueError, RuntimeError, OSError):
        return False


requires_blkio = pytest.mark.skipif(
    not _has_blkio_feature(),
    reason="Rust blkio feature not enabled (io_engine='libblkio' not accepted)",
)

requires_rust_ext = pytest.mark.skipif(
    not _has_rust_ext(),
    reason="lmcache_rust_raw_block_io extension not installed",
)

_BLOCK_SIZE = 4096
_DEVICE_SIZE_BYTES = 64 * 1024 * 1024  # 64 MB


# ---------------------------------------------------------------------------
# Device provisioning helpers
# ---------------------------------------------------------------------------


class _LoopDevice:
    """RAII wrapper around a losetup-backed loopback device."""

    def __init__(self, size_bytes: int):
        self._tmp = tempfile.NamedTemporaryFile(
            suffix=".blkio_blkdev_test", delete=False
        )
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


def _resolve_odirect_device() -> tuple[str, callable] | None:
    """Try to get a device that supports O_DIRECT.

    Returns:
        ``(device_path, cleanup_callable)`` or ``None`` if unavailable.
    """
    # 1. Explicit env var
    env_dev = os.environ.get("LMCACHE_BLKIO_TEST_DEVICE")
    if env_dev:
        return env_dev, lambda: None

    # 2. Loopback (requires root)
    if os.geteuid() == 0:
        loop = _LoopDevice(_DEVICE_SIZE_BYTES)
        if loop.path:
            return loop.path, loop.close

    return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def loop_in_thread():
    """Run an asyncio event loop in a background thread."""
    loop = asyncio.new_event_loop()
    t = threading.Thread(
        target=loop.run_forever, name="blkio-blkdev-test-loop", daemon=True
    )
    t.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=5)
        loop.close()


@pytest.fixture
def temp_device_file():
    """Provide a sparse temp file usable as a device path (no O_DIRECT)."""
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(_DEVICE_SIZE_BYTES)
        yield dev_path


@pytest.fixture
def odirect_device():
    """Provide a block device that supports O_DIRECT, or skip the test."""
    result = _resolve_odirect_device()
    if result is None:
        pytest.skip(
            "No O_DIRECT-capable device available "
            "(set LMCACHE_BLKIO_TEST_DEVICE or run as root for loopback)"
        )
    path, cleanup = result
    try:
        yield path
    finally:
        cleanup()


# ---------------------------------------------------------------------------
# Level 1: RawBlockDevice(io_engine='libblkio') unit / smoke tests
# ---------------------------------------------------------------------------


@requires_blkio
class TestBlkioBlockDeviceSmoke:
    """Direct tests against ``RawBlockDevice`` with ``io_engine='libblkio'``."""

    def test_open_size_close(self, temp_device_file: str) -> None:
        """Device can be opened, reports correct size, and closes cleanly."""
        from lmcache_rust_raw_block_io import RawBlockDevice

        dev = RawBlockDevice(
            temp_device_file, writable=True, use_odirect=False,
            alignment=4096, io_engine="libblkio",
        )
        assert dev.size_bytes() == _DEVICE_SIZE_BYTES
        dev.close()

    def test_write_read_roundtrip(self, temp_device_file: str) -> None:
        """Write data, read it back, verify contents match."""
        from lmcache_rust_raw_block_io import RawBlockDevice

        dev = RawBlockDevice(
            temp_device_file, writable=True, use_odirect=False,
            alignment=4096, io_engine="libblkio",
        )
        try:
            data = bytearray(b"\xab" * _BLOCK_SIZE)
            dev.pwrite_from_buffer(0, data, payload_len=_BLOCK_SIZE, total_len=_BLOCK_SIZE)

            out = bytearray(_BLOCK_SIZE)
            dev.pread_into(0, out, payload_len=_BLOCK_SIZE, total_len=_BLOCK_SIZE)
            assert out == data
        finally:
            dev.close()

    def test_write_read_with_padding(self, temp_device_file: str) -> None:
        """Write with payload < total_len (zero-padded), read back payload portion."""
        from lmcache_rust_raw_block_io import RawBlockDevice

        dev = RawBlockDevice(
            temp_device_file, writable=True, use_odirect=False,
            alignment=4096, io_engine="libblkio",
        )
        try:
            payload = b"hello world"
            payload_len = len(payload)
            total_len = _BLOCK_SIZE
            data = bytearray(payload + b"\x00" * (total_len - payload_len))

            dev.pwrite_from_buffer(0, data, payload_len=payload_len, total_len=total_len)

            out = bytearray(total_len)
            dev.pread_into(0, out, payload_len=payload_len, total_len=total_len)
            assert out[:payload_len] == bytearray(payload)
        finally:
            dev.close()

    def test_distinct_offsets(self, temp_device_file: str) -> None:
        """Writes at different offsets don't interfere."""
        from lmcache_rust_raw_block_io import RawBlockDevice

        dev = RawBlockDevice(
            temp_device_file, writable=True, use_odirect=False,
            alignment=4096, io_engine="libblkio",
        )
        try:
            pattern_a = bytearray(b"\xaa" * _BLOCK_SIZE)
            pattern_b = bytearray(b"\xbb" * _BLOCK_SIZE)

            dev.pwrite_from_buffer(0, pattern_a, payload_len=_BLOCK_SIZE, total_len=_BLOCK_SIZE)
            dev.pwrite_from_buffer(
                _BLOCK_SIZE, pattern_b, payload_len=_BLOCK_SIZE, total_len=_BLOCK_SIZE
            )

            out_a = bytearray(_BLOCK_SIZE)
            out_b = bytearray(_BLOCK_SIZE)
            dev.pread_into(0, out_a, payload_len=_BLOCK_SIZE, total_len=_BLOCK_SIZE)
            dev.pread_into(
                _BLOCK_SIZE, out_b, payload_len=_BLOCK_SIZE, total_len=_BLOCK_SIZE
            )

            assert out_a == pattern_a
            assert out_b == pattern_b
        finally:
            dev.close()

    def test_close_is_idempotent(self, temp_device_file: str) -> None:
        """Calling close() twice should not raise."""
        from lmcache_rust_raw_block_io import RawBlockDevice

        dev = RawBlockDevice(
            temp_device_file, writable=True, use_odirect=False,
            alignment=4096, io_engine="libblkio",
        )
        dev.close()
        dev.close()  # Should not raise

    def test_operations_after_close_raise(self, temp_device_file: str) -> None:
        """I/O after close() should raise RuntimeError."""
        from lmcache_rust_raw_block_io import RawBlockDevice

        dev = RawBlockDevice(
            temp_device_file, writable=True, use_odirect=False,
            alignment=4096, io_engine="libblkio",
        )
        dev.close()

        with pytest.raises(RuntimeError):
            dev.size_bytes()

        with pytest.raises(RuntimeError):
            dev.pwrite_from_buffer(0, bytearray(4096))

        with pytest.raises(RuntimeError):
            dev.pread_into(0, bytearray(4096), payload_len=4096)

    def test_empty_path_raises(self) -> None:
        """Empty path should raise ValueError."""
        from lmcache_rust_raw_block_io import RawBlockDevice

        with pytest.raises((ValueError, RuntimeError)):
            RawBlockDevice("", writable=True, io_engine="libblkio")

    def test_invalid_alignment_raises(self, temp_device_file: str) -> None:
        """Non-power-of-two alignment should raise ValueError."""
        from lmcache_rust_raw_block_io import RawBlockDevice

        with pytest.raises((ValueError, RuntimeError)):
            RawBlockDevice(
                temp_device_file, writable=True, use_odirect=False,
                alignment=3000, io_engine="libblkio",
            )

    def test_large_roundtrip(self, temp_device_file: str) -> None:
        """Write and read back a larger buffer (multiple blocks)."""
        from lmcache_rust_raw_block_io import RawBlockDevice

        dev = RawBlockDevice(
            temp_device_file, writable=True, use_odirect=False,
            alignment=4096, io_engine="libblkio",
        )
        try:
            size = 16 * _BLOCK_SIZE  # 64 KB
            data = bytearray(os.urandom(size))
            dev.pwrite_from_buffer(0, data, payload_len=size, total_len=size)

            out = bytearray(size)
            dev.pread_into(0, out, payload_len=size, total_len=size)
            assert out == data
        finally:
            dev.close()


# ---------------------------------------------------------------------------
# Level 2: RustRawBlockBackend integration with io_engine='libblkio'
# ---------------------------------------------------------------------------


def _make_backend(
    dev_path: str,
    memory_allocator: object,
    loop: asyncio.AbstractEventLoop,
    *,
    instance_id: str = "test_blkio_rawblock",
    extra_config_overrides: dict | None = None,
) -> tuple[RustRawBlockBackend, LocalCPUBackend]:
    """Build a ``RustRawBlockBackend`` with ``io_engine='libblkio'``.

    Returns:
        ``(backend, local_cpu)`` — caller is responsible for calling
        ``backend.close()``.
    """
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=0.1,
        lmcache_instance_id=instance_id,
    )
    config.storage_plugins = []
    extra = {
        "rust_raw_block.device_path": dev_path,
        "rust_raw_block.block_align": 4096,
        "rust_raw_block.header_bytes": 4096,
        "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
        "rust_raw_block.meta_enable_periodic": False,
        "rust_raw_block.io_engine": "libblkio",
    }
    if extra_config_overrides:
        extra.update(extra_config_overrides)
    config.extra_config = extra

    metadata = LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 256, 8, 128),
    )

    local_cpu = LocalCPUBackend(
        config=config,
        metadata=metadata,
        dst_device="cpu",
        memory_allocator=memory_allocator,
    )
    backend = RustRawBlockBackend(
        config=config,
        metadata=metadata,
        local_cpu_backend=local_cpu,
        loop=loop,
        dst_device="cpu",
    )
    return backend, local_cpu


@requires_blkio
@requires_rust_ext
class TestBlkioRawBlockBackendIntegration:
    """Full backend integration tests using ``io_engine='libblkio'``."""

    def test_put_get_roundtrip(
        self,
        memory_allocator: object,
        loop_in_thread: asyncio.AbstractEventLoop,
        temp_device_file: str,
    ) -> None:
        """Basic put/get roundtrip through the libblkio I/O path."""
        backend, local_cpu = _make_backend(
            temp_device_file, memory_allocator, loop_in_thread
        )
        try:
            key = CacheEngineKey("test_model", 1, 0, 12345, torch.bfloat16)
            allocator = AdHocMemoryAllocator(device="cpu")
            obj = allocator.allocate(
                [torch.Size([2, 16, 8, 128])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj is not None and obj.tensor is not None
            obj.tensor.fill_(7)
            expected = bytes(obj.byte_array)

            futs = backend.batched_submit_put_task([key], [obj])
            assert futs is not None
            assert isinstance(futs[0], Future)
            futs[0].result(timeout=10)

            out = backend.get_blocking(key)
            assert out is not None
            assert bytes(out.byte_array) == expected
        finally:
            backend.close()

    def test_batched_get_blocking_prefix_stop(
        self,
        memory_allocator: object,
        loop_in_thread: asyncio.AbstractEventLoop,
        temp_device_file: str,
    ) -> None:
        """Batched blocking get stops at the first miss."""
        backend, local_cpu = _make_backend(
            temp_device_file, memory_allocator, loop_in_thread
        )
        try:
            allocator = AdHocMemoryAllocator(device="cpu")
            key1 = CacheEngineKey("test_model", 1, 0, 1001, torch.bfloat16)
            key_miss = CacheEngineKey("test_model", 1, 0, 1002, torch.bfloat16)
            key3 = CacheEngineKey("test_model", 1, 0, 1003, torch.bfloat16)

            obj1 = allocator.allocate(
                [torch.Size([2, 16, 8, 128])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            obj3 = allocator.allocate(
                [torch.Size([2, 16, 8, 128])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj1 is not None and obj1.tensor is not None
            assert obj3 is not None and obj3.tensor is not None
            obj1.tensor.fill_(1)
            obj3.tensor.fill_(3)
            expected1 = bytes(obj1.byte_array)
            expected3 = bytes(obj3.byte_array)

            for key, obj in ((key1, obj1), (key3, obj3)):
                futs = backend.batched_submit_put_task([key], [obj])
                assert futs is not None
                futs[0].result(timeout=10)
                obj.ref_count_down()

            blocking_results = backend.batched_get_blocking(
                [key1, key_miss, key3]
            )
            assert len(blocking_results) == 3
            assert blocking_results[0] is not None
            assert bytes(blocking_results[0].byte_array) == expected1
            assert blocking_results[1] is None
            assert blocking_results[2] is None
            blocking_results[0].ref_count_down()

            # key3 should still be retrievable individually
            out3 = backend.get_blocking(key3)
            assert out3 is not None
            assert bytes(out3.byte_array) == expected3
            out3.ref_count_down()
        finally:
            backend.close()

    def test_capacity_overflow_rejects(
        self,
        memory_allocator: object,
        loop_in_thread: asyncio.AbstractEventLoop,
        temp_device_file: str,
    ) -> None:
        """Write beyond capacity is rejected (no auto-eviction)."""
        capacity_bytes = 3 * 4 * 1024 * 1024
        slot_bytes = 4 * 1024 * 1024
        meta_total_bytes = 4 * 1024 * 1024  # matches _make_backend default

        backend, local_cpu = _make_backend(
            temp_device_file,
            memory_allocator,
            loop_in_thread,
            extra_config_overrides={
                "rust_raw_block.capacity_bytes": capacity_bytes,
                "rust_raw_block.slot_bytes": slot_bytes,
            },
        )
        try:
            allocator = AdHocMemoryAllocator(device="cpu")
            max_slots = (capacity_bytes - meta_total_bytes) // slot_bytes

            keys = []
            # Fill all slots
            for i in range(max_slots):
                key = CacheEngineKey("test_model", 1, 0, 5000 + i, torch.bfloat16)
                keys.append(key)
                obj = allocator.allocate(
                    [torch.Size([2, 16, 8, 128])],
                    [torch.bfloat16],
                    fmt=MemoryFormat.KV_T2D,
                )
                assert obj is not None and obj.tensor is not None
                obj.tensor.fill_(i)
                futs = backend.batched_submit_put_task([key], [obj])
                futs[0].result(timeout=10)
                obj.ref_count_down()

            # One more write should fail (no free slots, no auto-eviction)
            overflow_key = CacheEngineKey(
                "test_model", 1, 0, 5000 + max_slots, torch.bfloat16
            )
            obj = allocator.allocate(
                [torch.Size([2, 16, 8, 128])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj is not None and obj.tensor is not None
            obj.tensor.fill_(99)
            futs = backend.batched_submit_put_task([overflow_key], [obj])
            with pytest.raises(RuntimeError, match="Failed to persist"):
                futs[0].result(timeout=10)
            obj.ref_count_down()

            # Original keys should still be retrievable
            for key in keys:
                out = backend.get_blocking(key)
                assert out is not None
                out.ref_count_down()

            # Overflow key should not exist
            out_overflow = backend.get_blocking(overflow_key)
            assert out_overflow is None
        finally:
            backend.close()

    def test_checkpoint_roundtrip(
        self,
        memory_allocator: object,
        loop_in_thread: asyncio.AbstractEventLoop,
    ) -> None:
        """Metadata checkpoint survives close + reopen."""
        with tempfile.TemporaryDirectory() as td:
            dev_path = os.path.join(td, "dev.bin")
            with open(dev_path, "wb") as f:
                f.truncate(_DEVICE_SIZE_BYTES)

            key = CacheEngineKey("test_model", 1, 0, 7777, torch.bfloat16)
            allocator = AdHocMemoryAllocator(device="cpu")

            # --- write + close (close triggers checkpoint) ---
            backend, local_cpu = _make_backend(
                dev_path,
                memory_allocator,
                loop_in_thread,
                instance_id="test_blkio_ckpt",
            )
            try:
                obj = allocator.allocate(
                    [torch.Size([2, 16, 8, 128])],
                    [torch.bfloat16],
                    fmt=MemoryFormat.KV_T2D,
                )
                assert obj is not None and obj.tensor is not None
                obj.tensor.fill_(77)
                expected = bytes(obj.byte_array)

                futs = backend.batched_submit_put_task([key], [obj])
                futs[0].result(timeout=10)
            finally:
                backend.close()

            # --- reopen and verify ---
            backend2, local_cpu2 = _make_backend(
                dev_path,
                memory_allocator,
                loop_in_thread,
                instance_id="test_blkio_ckpt",
            )
            try:
                out = backend2.get_blocking(key)
                assert out is not None, "Key not found after checkpoint recovery"
                assert bytes(out.byte_array) == expected
                out.ref_count_down()
            finally:
                backend2.close()


# ---------------------------------------------------------------------------
# Level 3: O_DIRECT on a real block device / loopback
# ---------------------------------------------------------------------------


@requires_blkio
class TestBlkioBlockDeviceODirect:
    """O_DIRECT tests requiring a real block device or loopback."""

    def test_odirect_write_read_roundtrip(self, odirect_device: str) -> None:
        """O_DIRECT write/read roundtrip on a block device."""
        from lmcache_rust_raw_block_io import RawBlockDevice

        dev = RawBlockDevice(
            odirect_device, writable=True, use_odirect=True,
            alignment=4096, io_engine="libblkio",
        )
        try:
            data = bytearray(b"\xcd" * _BLOCK_SIZE)
            dev.pwrite_from_buffer(0, data, payload_len=_BLOCK_SIZE, total_len=_BLOCK_SIZE)

            out = bytearray(_BLOCK_SIZE)
            dev.pread_into(0, out, payload_len=_BLOCK_SIZE, total_len=_BLOCK_SIZE)
            assert out == data
        finally:
            dev.close()

    def test_odirect_large_roundtrip(self, odirect_device: str) -> None:
        """O_DIRECT with a multi-block buffer."""
        from lmcache_rust_raw_block_io import RawBlockDevice

        dev = RawBlockDevice(
            odirect_device, writable=True, use_odirect=True,
            alignment=4096, io_engine="libblkio",
        )
        try:
            size = 64 * _BLOCK_SIZE  # 256 KB
            data = bytearray(os.urandom(size))
            dev.pwrite_from_buffer(0, data, payload_len=size, total_len=size)

            out = bytearray(size)
            dev.pread_into(0, out, payload_len=size, total_len=size)
            assert out == data
        finally:
            dev.close()

    def test_odirect_with_padding(self, odirect_device: str) -> None:
        """O_DIRECT write with payload + zero-padding, read back payload."""
        from lmcache_rust_raw_block_io import RawBlockDevice

        dev = RawBlockDevice(
            odirect_device, writable=True, use_odirect=True,
            alignment=4096, io_engine="libblkio",
        )
        try:
            payload = os.urandom(2000)
            payload_len = len(payload)
            total_len = _BLOCK_SIZE  # aligned
            data = bytearray(payload + b"\x00" * (total_len - payload_len))

            dev.pwrite_from_buffer(0, data, payload_len=payload_len, total_len=total_len)

            out = bytearray(total_len)
            dev.pread_into(0, out, payload_len=payload_len, total_len=total_len)
            assert out[:payload_len] == bytearray(payload)
        finally:
            dev.close()

    def test_odirect_multiple_offsets(self, odirect_device: str) -> None:
        """O_DIRECT writes at aligned offsets are independent."""
        from lmcache_rust_raw_block_io import RawBlockDevice

        dev = RawBlockDevice(
            odirect_device, writable=True, use_odirect=True,
            alignment=4096, io_engine="libblkio",
        )
        try:
            n = 8
            patterns = [bytearray(bytes([i & 0xFF]) * _BLOCK_SIZE) for i in range(n)]

            for i, pat in enumerate(patterns):
                dev.pwrite_from_buffer(
                    i * _BLOCK_SIZE,
                    pat,
                    payload_len=_BLOCK_SIZE,
                    total_len=_BLOCK_SIZE,
                )

            for i, pat in enumerate(patterns):
                out = bytearray(_BLOCK_SIZE)
                dev.pread_into(
                    i * _BLOCK_SIZE,
                    out,
                    payload_len=_BLOCK_SIZE,
                    total_len=_BLOCK_SIZE,
                )
                assert out == pat, f"Mismatch at offset {i * _BLOCK_SIZE}"
        finally:
            dev.close()


@requires_blkio
@requires_rust_ext
class TestBlkioRawBlockBackendODirect:
    """Full backend integration with O_DIRECT on a real block device."""

    def test_odirect_put_get_roundtrip(
        self,
        memory_allocator: object,
        loop_in_thread: asyncio.AbstractEventLoop,
        odirect_device: str,
    ) -> None:
        """Put/get roundtrip through the full backend with O_DIRECT."""
        backend, local_cpu = _make_backend(
            odirect_device,
            memory_allocator,
            loop_in_thread,
            instance_id="test_blkio_odirect",
            extra_config_overrides={"rust_raw_block.use_odirect": True},
        )
        try:
            key = CacheEngineKey("test_model", 1, 0, 9999, torch.bfloat16)
            allocator = AdHocMemoryAllocator(device="cpu")
            obj = allocator.allocate(
                [torch.Size([2, 16, 8, 128])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj is not None and obj.tensor is not None
            obj.tensor.fill_(42)
            expected = bytes(obj.byte_array)

            futs = backend.batched_submit_put_task([key], [obj])
            assert futs is not None
            futs[0].result(timeout=10)

            out = backend.get_blocking(key)
            assert out is not None
            assert bytes(out.byte_array) == expected
        finally:
            backend.close()


# ---------------------------------------------------------------------------
# Level 4: Throughput benchmarks (pytest-benchmark)
# ---------------------------------------------------------------------------

_BENCH_SIZES_MB = [1, 4, 16]

# Module-level collector for throughput results.  Each benchmark test
# appends a dict here; the ``pytest_terminal_summary`` hook (below)
# renders them as a table with GB/s.
_throughput_results: list[dict] = []


def _record_throughput(benchmark, nbytes: int, label: str) -> None:
    """Record throughput result for the summary table.

    Args:
        benchmark: The pytest-benchmark fixture after ``benchmark()``
            has been called (stats are populated).
        nbytes: Number of bytes transferred per iteration.
        label: Human-readable label for the result row.
    """
    mean_s = benchmark.stats["mean"]
    min_s = benchmark.stats["min"]
    max_s = benchmark.stats["max"]
    gb = nbytes / (1024 * 1024 * 1024)
    _throughput_results.append({
        "label": label,
        "size_mb": nbytes / (1024 * 1024),
        "mean_us": mean_s * 1e6,
        "min_gbps": gb / max_s if max_s > 0 else 0.0,
        "mean_gbps": gb / mean_s if mean_s > 0 else 0.0,
        "max_gbps": gb / min_s if min_s > 0 else 0.0,
    })


@requires_blkio
class TestBlkioBenchmarkFile:
    """Throughput benchmarks on a temp file (no O_DIRECT)."""

    @pytest.mark.benchmark(group="blkio_write_file")
    @pytest.mark.parametrize("size_mb", _BENCH_SIZES_MB)
    def test_write_throughput(
        self, benchmark, temp_device_file: str, size_mb: int
    ) -> None:
        """Measure blkio write throughput to a temp file."""
        from lmcache_rust_raw_block_io import RawBlockDevice

        dev = RawBlockDevice(
            temp_device_file, writable=True, use_odirect=False,
            alignment=4096, io_engine="libblkio",
        )
        nbytes = size_mb * 1024 * 1024
        data = bytearray(os.urandom(nbytes))

        def do_write():
            dev.pwrite_from_buffer(
                0, data, payload_len=nbytes, total_len=nbytes
            )

        benchmark.extra_info["size_mb"] = size_mb
        benchmark(do_write)
        _record_throughput(benchmark, nbytes, f"file write {size_mb}M")
        dev.close()

    @pytest.mark.benchmark(group="blkio_read_file")
    @pytest.mark.parametrize("size_mb", _BENCH_SIZES_MB)
    def test_read_throughput(
        self, benchmark, temp_device_file: str, size_mb: int
    ) -> None:
        """Measure blkio read throughput from a temp file."""
        from lmcache_rust_raw_block_io import RawBlockDevice

        dev = RawBlockDevice(
            temp_device_file, writable=True, use_odirect=False,
            alignment=4096, io_engine="libblkio",
        )
        nbytes = size_mb * 1024 * 1024
        data = bytearray(os.urandom(nbytes))
        dev.pwrite_from_buffer(
            0, data, payload_len=nbytes, total_len=nbytes
        )
        out = bytearray(nbytes)

        def do_read():
            dev.pread_into(0, out, payload_len=nbytes, total_len=nbytes)

        benchmark.extra_info["size_mb"] = size_mb
        benchmark(do_read)
        _record_throughput(benchmark, nbytes, f"file read {size_mb}M")
        dev.close()


@requires_blkio
class TestBlkioBenchmarkODirect:
    """Throughput benchmarks with O_DIRECT on a block device."""

    @pytest.mark.benchmark(group="blkio_write_odirect")
    @pytest.mark.parametrize("size_mb", _BENCH_SIZES_MB)
    def test_write_throughput(
        self, benchmark, odirect_device: str, size_mb: int
    ) -> None:
        """Measure blkio O_DIRECT write throughput."""
        from lmcache_rust_raw_block_io import RawBlockDevice

        dev = RawBlockDevice(
            odirect_device, writable=True, use_odirect=True,
            alignment=4096, io_engine="libblkio",
        )
        nbytes = size_mb * 1024 * 1024
        data = bytearray(os.urandom(nbytes))

        def do_write():
            dev.pwrite_from_buffer(
                0, data, payload_len=nbytes, total_len=nbytes
            )

        benchmark.extra_info["size_mb"] = size_mb
        benchmark(do_write)
        _record_throughput(benchmark, nbytes, f"O_DIRECT write {size_mb}M")
        dev.close()

    @pytest.mark.benchmark(group="blkio_read_odirect")
    @pytest.mark.parametrize("size_mb", _BENCH_SIZES_MB)
    def test_read_throughput(
        self, benchmark, odirect_device: str, size_mb: int
    ) -> None:
        """Measure blkio O_DIRECT read throughput."""
        from lmcache_rust_raw_block_io import RawBlockDevice

        dev = RawBlockDevice(
            odirect_device, writable=True, use_odirect=True,
            alignment=4096, io_engine="libblkio",
        )
        nbytes = size_mb * 1024 * 1024
        data = bytearray(os.urandom(nbytes))
        dev.pwrite_from_buffer(
            0, data, payload_len=nbytes, total_len=nbytes
        )
        out = bytearray(nbytes)

        def do_read():
            dev.pread_into(0, out, payload_len=nbytes, total_len=nbytes)

        benchmark.extra_info["size_mb"] = size_mb
        benchmark(do_read)
        _record_throughput(benchmark, nbytes, f"O_DIRECT read {size_mb}M")
        dev.close()
