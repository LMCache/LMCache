# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SPDK-specific RawBlockCore functionality.

These tests cover:
- HeaderBufferPool and CheckPointPayloadBufferPool DMA buffer management
- Buffer registration tracking via _is_buffer_spdk_registered()
- SPDK checkpoint read method routing
- _spdk_call_with_gil_released() helper function
"""

# Standard
from pathlib import Path
from unittest.mock import MagicMock, patch

# Third Party
import pytest

# First Party
# First Party & Test Utils
from lmcache.v1.storage_backend.raw_block.core import (
    CheckPointPayloadBufferPool,
    HeaderBufferPool,
    RawBlockCore,
    RawBlockCoreConfig,
    _spdk_call_with_gil_released,
)
from tests.v1.storage_backend.raw_block_test_utils import (
    make_raw_block_file,
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

        # Test acquire/release updates stats
        buf = pool.acquire()
        assert pool.stats()["available"] == 3
        assert pool.stats()["in_use"] == 1
        pool.release(buf)
        assert pool.stats()["available"] == 4

    def test_checkpoint_pool_allocation(self, tmp_path):
        """Test CheckPointPayloadBufferPool allocates buffers correctly."""
        mock_spdk = MagicMock()
        ptr_counter = [0x200000000]

        def mock_allocate(size, align, numa_id=-1):
            ptr = ptr_counter[0]
            ptr_counter[0] += size
            return ptr

        mock_spdk.allocate_spdk_memory.side_effect = mock_allocate
        mock_spdk.free_spdk_memory.return_value = None

        pool = CheckPointPayloadBufferPool(
            buffer_size=8192,
            pool_size=2,
            spdk_engine=mock_spdk,
        )

        assert mock_spdk.allocate_spdk_memory.call_count == 2
        assert pool.stats()["available"] == 2

        buf = pool.acquire()
        assert pool.stats()["in_use"] == 1
        pool.release(buf)

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
            buffer_size=128,
            pool_size=3,
            spdk_engine=mock_spdk,
        )

        pool.acquire()
        pool.cleanup()

        assert len(freed_ptrs) == 3
        assert pool.stats()["available"] == 0


class TestBufferRegistration:
    """Tests for SPDK buffer registration tracking."""

    def test_buffer_registration_with_multiple_sources(self, tmp_path):
        """Test detection of buffers from external, header pool, and checkpoint pool."""
        config = _make_config(tmp_path)
        core = RawBlockCore(config, key_namespace="object")
        try:
            # Set up multiple buffer sources
            core._registered_external_buffers = [(0x100000000, 0x1000000)]
            core._header_pool = MagicMock()
            core._header_pool._spdk_ptrs = [(0x200000000, 4096)]
            core._checkpoint_pool = MagicMock()
            core._checkpoint_pool._spdk_ptrs = [(0x300000000, 8192)]

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

            # Checkpoint: [0x300000000, 0x300002000), size=8192 (0x2000)
            assert core._is_buffer_spdk_registered(0x300000000) is True  # Start
            assert core._is_buffer_spdk_registered(0x300001000) is True  # Middle
            assert (
                core._is_buffer_spdk_registered(0x300001FFF) is True
            )  # Last byte (ptr+size-1)
            assert (
                core._is_buffer_spdk_registered(0x300002000) is False
            )  # Just outside (ptr+size)

            assert core._is_buffer_spdk_registered(0x500000000) is False  # None
        finally:
            core.close()


class TestSPDKCallWithGILReleased:
    """Tests for the _spdk_call_with_gil_released helper function."""

    def test_spdk_call_propagates_result_and_exception(self):
        """Test result propagation and exception handling."""
        # Test successful call
        mock_func = MagicMock(return_value=0)
        result = _spdk_call_with_gil_released(mock_func, 100, 200, 0x12345)
        assert result == 0
        mock_func.assert_called_once_with(100, 200, 0x12345)

        # Test exception propagation
        mock_func = MagicMock(side_effect=RuntimeError("SPDK error"))
        with pytest.raises(RuntimeError, match="SPDK error"):
            _spdk_call_with_gil_released(mock_func)


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
