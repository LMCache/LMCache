# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SPDK-specific RawBlockCore functionality.

These tests cover:
- HeaderBufferPool DMA buffer management
- Buffer registration tracking via _is_buffer_spdk_registered()
- SPDK checkpoint read method routing
"""

# Standard
from pathlib import Path
from unittest.mock import MagicMock, patch
import ctypes
import importlib.util

# Third Party
import pytest

# First Party
# First Party & Test Utils
from lmcache.v1.storage_backend.raw_block.core import (
    HeaderBufferPool,
    RawBlockCore,
    RawBlockCoreConfig,
)
from lmcache.v1.storage_backend.raw_block.spdk_ffi import (
    IO_READ,
    IO_WRITE,
)
from tests.v1.storage_backend.raw_block_test_utils import (
    make_raw_block_file,
)

requires_rust_raw_block_io = pytest.mark.skipif(
    importlib.util.find_spec("lmcache_rust_raw_block_io") is None,
    reason="lmcache_rust_raw_block_io is not installed",
)

# CI-safe defaults
RAW_BLOCK_CI_CAPACITY_BYTES = 128 * 1024 * 1024
RAW_BLOCK_CI_BLOCK_ALIGN = 4096
RAW_BLOCK_CI_HEADER_BYTES = 4096
RAW_BLOCK_CI_SLOT_BYTES = 65536
RAW_BLOCK_CI_META_TOTAL_BYTES = 1 * 1024 * 1024


def _make_config(tmp_path: Path, capacity_bytes: int = RAW_BLOCK_CI_CAPACITY_BYTES):
    """Build a raw block config for tests."""
    return RawBlockCoreConfig(
        device_path=str(make_raw_block_file(tmp_path, capacity_bytes)),
        capacity_bytes=capacity_bytes,
        block_align=RAW_BLOCK_CI_BLOCK_ALIGN,
        header_bytes=RAW_BLOCK_CI_HEADER_BYTES,
        slot_bytes=RAW_BLOCK_CI_SLOT_BYTES,
        use_odirect=False,
        enable_zero_copy=False,
        meta_total_bytes=RAW_BLOCK_CI_META_TOTAL_BYTES,
        meta_magic=b"LMCIDX01",
        meta_version=1,
        meta_checkpoint_interval_sec=60,
        meta_idle_quiet_ms=0,
        meta_enable_periodic=False,
        meta_verify_on_load=True,
        io_engine="posix",
    )


class TestSPDKBufferPools:
    """Tests for SPDK DMA buffer pools."""

    def test_header_pool_allocation_and_stats(self, tmp_path):
        """Test HeaderBufferPool allocation and statistics tracking."""
        mock_spdk = MagicMock()
        allocated = []

        def mock_allocate(size, align, numa_id=-1):
            ptr = 0x100000000 + len(allocated) * size
            allocated.append(ptr)
            return ptr

        mock_spdk.allocate_spdk_memory.side_effect = mock_allocate
        mock_spdk.free_spdk_memory.return_value = None

        pool = HeaderBufferPool(
            buffer_size=4096,
            pool_size=4,
            spdk_engine=mock_spdk,
        )

        assert mock_spdk.allocate_spdk_memory.call_count == 4
        stats = pool.stats()
        assert stats["available"] == 4
        assert stats["in_use"] == 0
        assert stats["total"] == 4

        # Test acquire_many/release_many updates stats
        bufs = pool.acquire_many(1)
        buf = bufs[0]
        assert pool.stats()["available"] == 3
        assert pool.stats()["in_use"] == 1
        pool.release_many([buf])
        assert pool.stats()["available"] == 4

    def test_acquire_many_grows_pool_and_release_many(self, tmp_path):
        """Test acquire_many grows the pool on demand and release_many returns
        buffers to it."""
        mock_spdk = MagicMock()
        allocated = []

        def mock_allocate(size, align, numa_id=-1):
            ptr = 0x100000000 + len(allocated) * size
            allocated.append(ptr)
            return ptr

        mock_spdk.allocate_spdk_memory.side_effect = mock_allocate
        mock_spdk.free_spdk_memory.return_value = None

        pool = HeaderBufferPool(
            buffer_size=4096,
            pool_size=2,
            spdk_engine=mock_spdk,
        )

        # Initial pool holds two buffers.
        assert mock_spdk.allocate_spdk_memory.call_count == 2

        # Acquiring more than the initial size grows the pool via SPDK.
        bufs = pool.acquire_many(5)
        assert len(bufs) == 5
        # 2 initial + 3 newly allocated.
        assert mock_spdk.allocate_spdk_memory.call_count == 5
        stats = pool.stats()
        assert stats["available"] == 0
        assert stats["in_use"] == 5
        assert stats["total"] == 5

        # Release them all back.
        pool.release_many(bufs)
        stats = pool.stats()
        assert stats["available"] == 5
        assert stats["in_use"] == 0
        assert stats["total"] == 5

        # A subsequent acquire reuses the returned buffers (no new allocation).
        more = pool.acquire_many(3)
        assert len(more) == 3
        assert mock_spdk.allocate_spdk_memory.call_count == 5

    def test_acquire_many_rejects_non_positive_count(self, tmp_path):
        """Test acquire_many rejects a non-positive count."""
        mock_spdk = MagicMock()
        mock_spdk.allocate_spdk_memory.side_effect = lambda size, align, numa_id=-1: (
            0x100000000
        )
        mock_spdk.free_spdk_memory.return_value = None

        pool = HeaderBufferPool(
            buffer_size=4096,
            pool_size=1,
            spdk_engine=mock_spdk,
        )

        with pytest.raises(ValueError, match="count must be > 0"):
            pool.acquire_many(0)
        with pytest.raises(ValueError, match="count must be > 0"):
            pool.acquire_many(-1)

    def test_release_many_rejects_foreign_buffer(self, tmp_path):
        """Test release_many raises when given a buffer not from this pool."""
        mock_spdk = MagicMock()
        mock_spdk.allocate_spdk_memory.side_effect = lambda size, align, numa_id=-1: (
            0x100000000
        )
        mock_spdk.free_spdk_memory.return_value = None

        pool = HeaderBufferPool(
            buffer_size=4096,
            pool_size=1,
            spdk_engine=mock_spdk,
        )

        with pytest.raises(ValueError, match="Buffer not from this HeaderBufferPool"):
            pool.release_many([object()])

    def test_pool_cleanup_frees_memory(self, tmp_path):
        """Test that cleanup properly frees all SPDK-allocated memory."""
        mock_spdk = MagicMock()
        freed_ptrs = []

        def mock_allocate(size, align, numa_id=-1):
            return 0x100000000 + len(freed_ptrs) * size

        def mock_free(ptr):
            freed_ptrs.append(ptr)

        mock_spdk.allocate_spdk_memory.side_effect = mock_allocate
        mock_spdk.free_spdk_memory.side_effect = mock_free

        pool = HeaderBufferPool(
            buffer_size=4096,
            pool_size=3,
            spdk_engine=mock_spdk,
        )

        pool.acquire_many(1)
        pool.cleanup()

        assert len(freed_ptrs) == 3
        assert pool.stats()["available"] == 0


@requires_rust_raw_block_io
class TestBufferRegistration:
    """Tests for SPDK buffer registration tracking."""

    def test_buffer_registration_with_multiple_sources(self, tmp_path):
        """Test detection of buffers from external and header pool sources."""
        config = _make_config(tmp_path)
        core = RawBlockCore(config, key_namespace="object")
        try:
            # Set up multiple buffer sources
            core._registered_external_buffers = [(0x100000000, 0x1000000)]
            core._header_pool = MagicMock()
            core._header_pool._spdk_ptrs = [(0x200000000, 4096)]

            # Test all regions (buffer range is [ptr, ptr+size))
            # External: [0x100000000, 0x101000000), size=0x1000000 (16MB)
            assert core._is_buffer_spdk_registered(0x100000000) is True
            assert core._is_buffer_spdk_registered(0x100800000) is True  # Middle
            assert (
                core._is_buffer_spdk_registered(0x100FFFFFF) is True
            )  # Last byte (ptr+size-1)
            assert (
                core._is_buffer_spdk_registered(0x101000000) is False
            )  # Just outside (ptr+size)

            # Header: [0x200000000, 0x200001000), size=4096 (0x1000)
            assert core._is_buffer_spdk_registered(0x200000000) is True  # Start
            assert core._is_buffer_spdk_registered(0x200000800) is True  # Middle
            assert (
                core._is_buffer_spdk_registered(0x200000FFF) is True
            )  # Last byte (ptr+size-1)
            assert (
                core._is_buffer_spdk_registered(0x200001000) is False
            )  # Just outside (ptr+size)

            assert core._is_buffer_spdk_registered(0x500000000) is False  # None
        finally:
            core.close()


@requires_rust_raw_block_io
class TestCheckpointReadRouting:
    """Tests for checkpoint read method routing."""

    def test_read_meta_header_routes_to_spdk(self, tmp_path):
        """Test that _read_meta_header routes to SPDK when enabled."""
        config = _make_config(tmp_path)
        core = RawBlockCore(config, key_namespace="object")
        try:
            core._spdk_engine = MagicMock()
            core._read_meta_header_spdk = MagicMock(return_value=None)

            with patch.object(core, "io_engine", "spdk"):
                result = core._read_meta_header(0)

            core._read_meta_header_spdk.assert_called_once_with(0)
            assert result is None
        finally:
            core.close()

    def test_load_meta_payload_routes_to_spdk(self, tmp_path):
        """Test that _load_meta_payload routes to SPDK when enabled."""
        config = _make_config(tmp_path)
        core = RawBlockCore(config, key_namespace="object")
        try:
            core._spdk_engine = MagicMock()
            core._load_meta_payload_spdk = MagicMock(return_value=None)

            header = {
                "seq": 1,
                "payload_len": 100,
                "crc": 0,
                "container_offset": 0,
            }

            with patch.object(core, "io_engine", "spdk"):
                result = core._load_meta_payload(header)

            core._load_meta_payload_spdk.assert_called_once_with(header)
            assert result is None
        finally:
            core.close()

    def test_checkpoint_fallback_when_no_spdk(self, tmp_path):
        """Test checkpoint methods work when SPDK not available."""
        config = _make_config(tmp_path)
        core = RawBlockCore(config, key_namespace="object")
        try:
            # No SPDK engine - should not crash
            assert core._read_meta_header(0) is None

            core._spdk_engine = None
            assert (
                core._load_meta_payload(
                    {"seq": 1, "payload_len": 100, "crc": 0, "container_offset": 0}
                )
                is None
            )
        finally:
            core.close()


@requires_rust_raw_block_io
class TestHeaderEncoding:
    """Tests for header encoding functionality."""

    def test_encode_header_requires_pool(self, tmp_path):
        """Test that _encode_header_using_pool raises when no pool."""
        config = _make_config(tmp_path)
        core = RawBlockCore(config, key_namespace="object")
        try:
            core._header_pool = None

            with pytest.raises(RuntimeError, match="requires SPDK header pool"):
                core._encode_header_using_pool(
                    slot_identity=12345,
                    payload_len=512,
                )
        finally:
            core.close()


class _FakeSpdkIoEngine:
    """In-memory stand-in for ``SpdkIoEngineFFI`` used to exercise the SPDK
    read/write buffer paths without loading the native library.

    It records the arguments passed to ``batch_io_submit`` and returns
    configurable submit/wait outcomes so the core's error handling can be
    verified in isolation.
    """

    def __init__(self, submit_rc: int = 0, wait_status: int = 0) -> None:
        self.submit_rc = submit_rc
        self.wait_status = wait_status
        self.batch_io_submit_calls: list[tuple] = []
        self.wait_batch_calls: list[int] = []

    def batch_io_submit(
        self, offsets, total_lens, buf_ptrs, count, op
    ) -> tuple[int, int]:
        self.batch_io_submit_calls.append(
            (list(offsets), list(total_lens), list(buf_ptrs), count, op)
        )
        return self.submit_rc, 0xABCDEF

    def wait_batch(self, batch_id: int) -> int:
        self.wait_batch_calls.append(batch_id)
        return self.wait_status


@requires_rust_raw_block_io
class TestSPDKReadWritePath:
    """Tests for the SPDK batched read/write buffer paths.

    These exercise ``_write_spdk_buffers`` and ``_read_spdk_buffers`` directly
    with a fake FFI engine, covering buffer registration, batch submission, and
    the submit/wait error paths.
    """

    def _make_spdk_core(self, tmp_path):
        """Build a core whose SPDK engine can be swapped for a fake."""
        config = _make_config(tmp_path)
        return RawBlockCore(config, key_namespace="object")

    @staticmethod
    def _register_buffer(core: RawBlockCore, buf: bytearray) -> int:
        """Register ``buf``'s address so the zero-copy registration check passes."""
        ptr = ctypes.addressof((ctypes.c_ubyte * len(buf)).from_buffer(buf))
        core._registered_external_buffers = [(ptr, len(buf))]
        return ptr

    def test_write_spdk_buffers_submits_registered_buffers(self, tmp_path):
        """A successful write submits offsets/total_lens and waits for the batch."""
        core = self._make_spdk_core(tmp_path)
        try:
            fake = _FakeSpdkIoEngine(submit_rc=0, wait_status=0)
            core._spdk_engine = fake
            buf = bytearray(bytes([1]) * 4096)
            self._register_buffer(core, buf)

            core._write_spdk_buffers([0x1000], [buf], [4096], [4096])

            assert len(fake.batch_io_submit_calls) == 1
            offsets, total_lens, buf_ptrs, count, op = fake.batch_io_submit_calls[0]
            assert offsets == [0x1000]
            assert total_lens == [4096]
            assert buf_ptrs == [
                ctypes.addressof((ctypes.c_ubyte * 4096).from_buffer(buf))
            ]
            assert count == 1
            assert op == IO_WRITE
            assert fake.wait_batch_calls == [0xABCDEF]
        finally:
            core.close()

    def test_write_spdk_buffers_requires_engine(self, tmp_path):
        """Writing without an initialized SPDK engine raises."""
        core = self._make_spdk_core(tmp_path)
        try:
            core._spdk_engine = None

            with pytest.raises(RuntimeError, match="SPDK engine not initialized"):
                core._write_spdk_buffers([0x1000], [bytearray(4096)], [4096], [4096])
        finally:
            core.close()

    def test_write_spdk_buffers_rejects_unregistered_buffer(self, tmp_path):
        """A buffer that is not registered with SPDK is rejected."""
        core = self._make_spdk_core(tmp_path)
        try:
            core._spdk_engine = _FakeSpdkIoEngine()
            # Deliberately do not register the buffer.
            unregistered = bytearray(bytes([2]) * 4096)

            with pytest.raises(RuntimeError, match="not registered with SPDK"):
                core._write_spdk_buffers([0x1000], [unregistered], [4096], [4096])
        finally:
            core.close()

    def test_write_spdk_buffers_submission_failure(self, tmp_path):
        """A non-zero submit return code is surfaced as an error."""
        core = self._make_spdk_core(tmp_path)
        try:
            fake = _FakeSpdkIoEngine(submit_rc=1, wait_status=0)
            core._spdk_engine = fake
            buf = bytearray(bytes([1]) * 4096)
            self._register_buffer(core, buf)

            with pytest.raises(
                RuntimeError, match="SPDK batched write submission failed"
            ):
                core._write_spdk_buffers([0x1000], [buf], [4096], [4096])
            # wait_batch must not be reached after a failed submission.
            assert fake.wait_batch_calls == []
        finally:
            core.close()

    def test_write_spdk_buffers_wait_failure(self, tmp_path):
        """A non-zero wait status is surfaced as an error."""
        core = self._make_spdk_core(tmp_path)
        try:
            fake = _FakeSpdkIoEngine(submit_rc=0, wait_status=-1)
            core._spdk_engine = fake
            buf = bytearray(bytes([1]) * 4096)
            self._register_buffer(core, buf)

            with pytest.raises(
                RuntimeError, match="SPDK write failed for batch_id=11259375"
            ):
                core._write_spdk_buffers([0x1000], [buf], [4096], [4096])
        finally:
            core.close()

    def test_read_spdk_buffers_submits_registered_buffers(self, tmp_path):
        """A successful read submits offsets/total_lens and waits for the batch."""
        core = self._make_spdk_core(tmp_path)
        try:
            fake = _FakeSpdkIoEngine(submit_rc=0, wait_status=0)
            core._spdk_engine = fake
            buf = bytearray(bytes([0]) * 4096)
            self._register_buffer(core, buf)

            core._read_spdk_buffers([0x1000], [buf], [4096], [4096])

            assert len(fake.batch_io_submit_calls) == 1
            offsets, total_lens, buf_ptrs, count, op = fake.batch_io_submit_calls[0]
            assert offsets == [0x1000]
            assert total_lens == [4096]
            assert count == 1
            assert op == IO_READ
            assert fake.wait_batch_calls == [0xABCDEF]
        finally:
            core.close()

    def test_read_spdk_buffers_requires_engine(self, tmp_path):
        """Reading without an initialized SPDK engine raises."""
        core = self._make_spdk_core(tmp_path)
        try:
            core._spdk_engine = None

            with pytest.raises(RuntimeError, match="SPDK engine not initialized"):
                core._read_spdk_buffers([0x1000], [bytearray(4096)], [4096], [4096])
        finally:
            core.close()

    def test_read_spdk_buffers_rejects_unregistered_buffer(self, tmp_path):
        """A buffer that is not registered with SPDK is rejected."""
        core = self._make_spdk_core(tmp_path)
        try:
            core._spdk_engine = _FakeSpdkIoEngine()
            unregistered = bytearray(bytes([0]) * 4096)

            with pytest.raises(RuntimeError, match="not registered with SPDK"):
                core._read_spdk_buffers([0x1000], [unregistered], [4096], [4096])
        finally:
            core.close()

    def test_read_spdk_buffers_rejects_short_buffer(self, tmp_path):
        """An output buffer shorter than the payload is rejected."""
        core = self._make_spdk_core(tmp_path)
        try:
            fake = _FakeSpdkIoEngine()
            core._spdk_engine = fake
            buf = bytearray(bytes([0]) * 1024)
            self._register_buffer(core, buf)

            with pytest.raises(
                ValueError, match="output buffer shorter than payload_len"
            ):
                core._read_spdk_buffers([0x1000], [buf], [4096], [4096])
            assert fake.batch_io_submit_calls == []
        finally:
            core.close()

    def test_read_spdk_buffers_submission_failure(self, tmp_path):
        """A non-zero submit return code is surfaced as an error."""
        core = self._make_spdk_core(tmp_path)
        try:
            fake = _FakeSpdkIoEngine(submit_rc=1, wait_status=0)
            core._spdk_engine = fake
            buf = bytearray(bytes([0]) * 4096)
            self._register_buffer(core, buf)

            with pytest.raises(
                RuntimeError, match="SPDK batched read submission failed"
            ):
                core._read_spdk_buffers([0x1000], [buf], [4096], [4096])
            assert fake.wait_batch_calls == []
        finally:
            core.close()

    def test_read_spdk_buffers_wait_failure(self, tmp_path):
        """A non-zero wait status is surfaced as an error."""
        core = self._make_spdk_core(tmp_path)
        try:
            fake = _FakeSpdkIoEngine(submit_rc=0, wait_status=-1)
            core._spdk_engine = fake
            buf = bytearray(bytes([0]) * 4096)
            self._register_buffer(core, buf)

            with pytest.raises(
                RuntimeError, match="SPDK read failed for batch_id=11259375"
            ):
                core._read_spdk_buffers([0x1000], [buf], [4096], [4096])
        finally:
            core.close()
