# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SPDK raw-block I/O engine and its header buffer pool.

These tests cover:
- :class:`HeaderBufferPool` DMA buffer management
- :class:`SpdkBlockIO` buffer-registration tracking via ``is_buffer_registered``
- :class:`SpdkBlockIO` batched write/read and ``read_batch_keys`` paths
- :class:`RawBlockCore` slot-header recovery dispatch to the SPDK batched path

The tests inject an in-memory fake :class:`SpdkIoEngineFFI` so they exercise the
real :class:`SpdkBlockIO` code paths without loading the native SPDK library or
hardware.
"""

# Standard
from pathlib import Path
from unittest.mock import MagicMock, patch
import ctypes
import importlib.util

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.raw_block.core import RawBlockCore
from lmcache.v1.distributed.l2_adapters.raw_block.spdk_ffi import (
    IO_READ,
    IO_WRITE,
)
from lmcache.v1.distributed.l2_adapters.raw_block.spdk_io_engine import (
    _DEFAULT_SPDK_READ_BATCH_KEYS,
    _MAX_SPDK_READ_BATCH_KEYS_CAP,
    HeaderBufferPool,
    SpdkBlockIO,
)
from tests.v1.distributed.raw_block.test_utils import (
    RAW_BLOCK_CI_BLOCK_ALIGN,
    RAW_BLOCK_CI_CAPACITY_BYTES,
    RAW_BLOCK_CI_HEADER_BYTES,
    make_raw_block_core_config,
    make_raw_block_file,
)

requires_rust_raw_block_io = pytest.mark.skipif(
    importlib.util.find_spec("lmcache_rust_raw_block_io") is None,
    reason="lmcache_rust_raw_block_io is not installed",
)


def _make_posix_config(
    tmp_path: Path, capacity_bytes: int = RAW_BLOCK_CI_CAPACITY_BYTES
):
    """Build a POSIX raw block config for tests that swap in an SPDK engine."""
    return make_raw_block_core_config(make_raw_block_file(tmp_path, capacity_bytes))


class _FakeSpdkIoEngine:
    """In-memory stand-in for ``SpdkIoEngineFFI`` used to exercise the SPDK

    read/write buffer paths without loading the native library.

    It backs the NVMe device and every DMA-allocated buffer with Python
    bytearrays, and resolves the integer buffer pointers the FFI API passes to
    the buffers that back them. ``batch_io_submit`` copies between the device
    store and the resolved buffers so zero-copy round-trips behave like the
    real engine.

    Args:
        submit_rc: Return code ``batch_io_submit`` reports on success.
        wait_status: Return code ``wait_batch`` reports.
        header_bytes: Header size the fake writes per region.
        device_size: Value returned by ``get_device_size``.
    """

    def __init__(
        self,
        submit_rc: int = 0,
        wait_status: int = 0,
        header_bytes: int = RAW_BLOCK_CI_HEADER_BYTES,
        device_size: int = RAW_BLOCK_CI_CAPACITY_BYTES,
    ) -> None:
        self.submit_rc = submit_rc
        self.wait_status = wait_status
        self.header_bytes = header_bytes
        self.device_size = device_size
        self.batch_io_submit_calls: list[tuple] = []
        self.wait_batch_calls: list[int] = []
        self.registered: list[tuple[int, int]] = []
        self._retained: list[bytearray] = []

    def allocate_spdk_memory(
        self, size: int, align: int = 4096, numa_id: int = -1
    ) -> int:
        """Allocate SPDK-style DMA memory backed by a retained Python buffer."""
        buf = bytearray(size)
        self._retained.append(buf)
        return ctypes.addressof((ctypes.c_ubyte * size).from_buffer(buf))

    def free_spdk_memory(self, ptr: int) -> None:
        """No-op: the backing buffer is retained until the fake is dropped."""
        pass

    def register_external_memory(self, ptr: int, size: int) -> int:
        """Record a registered region and report success."""
        self.registered.append((int(ptr), int(size)))
        return 0

    def get_device_size(self) -> int:
        """Return the configured device size."""
        return self.device_size

    def get_max_io_depth(self) -> int:
        """Return no queued depths so the engine uses its default batch size."""
        return 0

    def shutdown_spdk_workers(self) -> None:
        """No-op shutdown for test cleanup."""
        pass

    def deinit(self) -> None:
        """No-op deinit for test cleanup."""
        pass

    def batch_io_submit(
        self, offsets, total_lens, buf_ptrs, count, op
    ) -> tuple[int, int]:
        self.batch_io_submit_calls.append(
            (list(offsets), list(total_lens), list(buf_ptrs), count, op)
        )
        # Populate each header so the core/engine can decode (slot_identity,
        # payload_len).
        hdr = int(total_lens[0]) if total_lens else self.header_bytes
        for i in range(count):
            view = ctypes.cast(buf_ptrs[i], ctypes.POINTER(ctypes.c_ubyte * hdr))
            view.contents[0:8] = list(b"LMCBLK01")
            view.contents[8:16] = list(i.to_bytes(8, "little"))
            view.contents[16:24] = list(int(total_lens[i]).to_bytes(8, "little"))
        return self.submit_rc, 0xABCDEF

    def wait_batch(self, batch_id: int) -> int:
        self.wait_batch_calls.append(batch_id)
        return self.wait_status


def _make_spdk_engine(
    tmp_path: Path,
    fake: _FakeSpdkIoEngine,
    header_pool_size: int = 4,
) -> SpdkBlockIO:
    """Build a SpdkBlockIO backed by the injected fake FFI."""
    return SpdkBlockIO(
        transport_type="tcp",
        target_ip="127.0.0.1",
        target_port="4420",
        target_nqn="nqn.2016-06.io.spdk:cnode1",
        block_align=RAW_BLOCK_CI_BLOCK_ALIGN,
        header_bytes=RAW_BLOCK_CI_HEADER_BYTES,
        header_pool_size=header_pool_size,
        spdk_ffi=fake,  # type: ignore[arg-type]  # test double mimics SpdkIoEngineFFI
    )


class TestSPDKBufferPools:
    """Tests for SPDK DMA header buffer pools."""

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
            header_size=4096,
            pool_size=4,
            spdk_engine=mock_spdk,
        )

        assert mock_spdk.allocate_spdk_memory.call_count == 4
        stats = pool.stats()
        assert stats["available"] == 4
        assert stats["in_use"] == 0
        assert stats["total"] == 4

        buf = pool.acquire()
        assert pool.stats()["available"] == 3
        assert pool.stats()["in_use"] == 1
        pool.release(buf)
        assert pool.stats()["available"] == 4

    def test_acquire_grows_pool_on_exhaustion(self, tmp_path):
        """Test acquire allocates new buffers as the pool is exhausted and
        release returns them for reuse."""
        mock_spdk = MagicMock()
        allocated = []

        def mock_allocate(size, align, numa_id=-1):
            ptr = 0x100000000 + len(allocated) * size
            allocated.append(ptr)
            return ptr

        mock_spdk.allocate_spdk_memory.side_effect = mock_allocate
        mock_spdk.free_spdk_memory.return_value = None

        pool = HeaderBufferPool(
            header_size=4096,
            pool_size=2,
            spdk_engine=mock_spdk,
        )

        assert mock_spdk.allocate_spdk_memory.call_count == 2

        bufs = [pool.acquire() for _ in range(5)]
        assert len(bufs) == 5
        assert mock_spdk.allocate_spdk_memory.call_count == 5
        stats = pool.stats()
        assert stats["available"] == 0
        assert stats["in_use"] == 5
        assert stats["total"] == 5

        for buf in bufs:
            pool.release(buf)
        stats = pool.stats()
        assert stats["available"] == 5
        assert stats["in_use"] == 0
        assert stats["total"] == 5

        # Releasing returns buffers for reuse without new allocations.
        bufs = [pool.acquire() for _ in range(3)]
        assert len(bufs) == 3
        assert mock_spdk.allocate_spdk_memory.call_count == 5

    def test_release_rejects_foreign_buffer(self, tmp_path):
        """Test release raises when given a buffer not from this pool."""
        mock_spdk = MagicMock()
        mock_spdk.allocate_spdk_memory.side_effect = lambda size, align, numa_id=-1: (
            0x100000000
        )
        mock_spdk.free_spdk_memory.return_value = None

        pool = HeaderBufferPool(
            header_size=4096,
            pool_size=1,
            spdk_engine=mock_spdk,
        )

        with pytest.raises(ValueError, match="Buffer not from this HeaderBufferPool"):
            pool.release(object())

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
            header_size=4096,
            pool_size=3,
            spdk_engine=mock_spdk,
        )

        pool.acquire()
        pool.cleanup()

        assert len(freed_ptrs) == 3
        assert pool.stats()["available"] == 0


class TestSpdkBlockIOBufferRegistration:
    """Tests for SpdkBlockIO buffer-registration tracking."""

    def test_registers_external_and_header_pool_regions(self, tmp_path):
        """A buffer counts as registered when its address lies within an
        external hugepage region or one of the header pool's DMA buffers."""
        fake = _FakeSpdkIoEngine()
        engine = _make_spdk_engine(tmp_path, fake)
        try:
            # External region: [0x100000000, 0x101000000), size=16 MiB.
            engine.register_io_buffers([0x100000000], [0x1000000])
            assert engine.is_buffer_registered(0x100000000) is True
            assert engine.is_buffer_registered(0x100800000) is True
            assert engine.is_buffer_registered(0x100FFFFFF) is True
            assert engine.is_buffer_registered(0x101000000) is False

            # Header pool region: acquire a pooled buffer and check its address.
            buf = engine._acquire_buffer(RAW_BLOCK_CI_HEADER_BYTES)
            ptr = SpdkBlockIO._buffer_ptr(buf)
            assert engine.is_buffer_registered(ptr) is True
            assert engine.is_buffer_registered(0x500000000) is False

            # Fresh DMA region: a buffer larger than the pooled header size is
            # freshly DMA-allocated and must count as registered until released.
            size = engine._header_buffer_size + RAW_BLOCK_CI_BLOCK_ALIGN
            buf = engine._acquire_buffer(size)
            fresh_ptr = SpdkBlockIO._buffer_ptr(buf)
            assert engine.is_buffer_registered(fresh_ptr) is True
            assert engine.is_buffer_registered(fresh_ptr + 1) is True

            engine._release_buffer(buf)
            assert engine.is_buffer_registered(fresh_ptr) is False
        finally:
            engine.close()

    def test_capacity_bytes_reports_device_size(self, tmp_path):
        """capacity_bytes returns the NVMe device size reported by SPDK."""
        fake = _FakeSpdkIoEngine(device_size=0x40000000)
        engine = _make_spdk_engine(tmp_path, fake)
        try:
            assert engine.supports_buffer_registration() is True
            assert engine.capacity_bytes() == 0x40000000
        finally:
            engine.close()


class TestSpdkBlockIOReadWritePath:
    """Tests for the SpdkBlockIO batched write/read buffer paths."""

    @staticmethod
    def _register_buffer(engine: SpdkBlockIO, buf: bytearray) -> int:
        """Register ``buf``'s address so the zero-copy registration check passes."""
        ptr = ctypes.addressof((ctypes.c_ubyte * len(buf)).from_buffer(buf))
        engine.register_io_buffers([ptr], [len(buf)])
        return ptr

    def test_write_buffers_submits_registered_buffers(self, tmp_path):
        """A successful write submits offsets/total_lens and waits for the batch."""
        fake = _FakeSpdkIoEngine(submit_rc=0, wait_status=0)
        engine = _make_spdk_engine(tmp_path, fake)
        try:
            buf = bytearray(bytes([1]) * 4096)
            self._register_buffer(engine, buf)

            engine.write_buffers([0x1000], [buf], [4096], [4096])

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
            engine.close()

    def test_write_buffers_rejects_unregistered_buffer(self, tmp_path):
        """A buffer that is not registered with SPDK is rejected."""
        fake = _FakeSpdkIoEngine()
        engine = _make_spdk_engine(tmp_path, fake)
        try:
            unregistered = bytearray(bytes([2]) * 4096)
            with pytest.raises(RuntimeError, match="not registered with SPDK"):
                engine.write_buffers([0x1000], [unregistered], [4096], [4096])
        finally:
            engine.close()

    def test_write_buffers_submission_failure(self, tmp_path):
        """A non-zero submit return code is surfaced as an error."""
        fake = _FakeSpdkIoEngine(submit_rc=1, wait_status=0)
        engine = _make_spdk_engine(tmp_path, fake)
        try:
            buf = bytearray(bytes([1]) * 4096)
            self._register_buffer(engine, buf)
            with pytest.raises(
                RuntimeError, match="SPDK batched write submission failed"
            ):
                engine.write_buffers([0x1000], [buf], [4096], [4096])
            assert fake.wait_batch_calls == []
        finally:
            engine.close()

    def test_write_buffers_wait_failure(self, tmp_path):
        """A non-zero wait status is surfaced as an error."""
        fake = _FakeSpdkIoEngine(submit_rc=0, wait_status=-1)
        engine = _make_spdk_engine(tmp_path, fake)
        try:
            buf = bytearray(bytes([1]) * 4096)
            self._register_buffer(engine, buf)
            with pytest.raises(
                RuntimeError, match="SPDK write failed for batch_id=11259375"
            ):
                engine.write_buffers([0x1000], [buf], [4096], [4096])
        finally:
            engine.close()

    def test_read_buffers_submits_registered_buffers(self, tmp_path):
        """A successful read submits offsets/total_lens and waits for the batch."""
        fake = _FakeSpdkIoEngine(submit_rc=0, wait_status=0)
        engine = _make_spdk_engine(tmp_path, fake)
        try:
            buf = bytearray(bytes([0]) * 4096)
            self._register_buffer(engine, buf)

            result = engine.read_buffers([0x1000], [buf], [4096], [4096])

            assert result == [True]
            assert len(fake.batch_io_submit_calls) == 1
            offsets, total_lens, buf_ptrs, count, op = fake.batch_io_submit_calls[0]
            assert offsets == [0x1000]
            assert total_lens == [4096]
            assert count == 1
            assert op == IO_READ
            assert fake.wait_batch_calls == [0xABCDEF]
        finally:
            engine.close()

    def test_read_buffers_rejects_short_buffer(self, tmp_path):
        """An output buffer shorter than the payload is rejected."""
        fake = _FakeSpdkIoEngine()
        engine = _make_spdk_engine(tmp_path, fake)
        try:
            buf = bytearray(bytes([0]) * 1024)
            self._register_buffer(engine, buf)
            with pytest.raises(
                ValueError, match="output buffer shorter than payload_len"
            ):
                engine.read_buffers([0x1000], [buf], [4096], [4096])
            assert fake.batch_io_submit_calls == []
        finally:
            engine.close()


class TestSpdkBlockIOReadBatchKeys:
    """Tests for SpdkBlockIO.read_batch_keys."""

    def test_read_batch_keys_uses_default_when_depth_unknown(self, tmp_path):
        """With no reported max I/O depth, batching uses the default size."""
        fake = _FakeSpdkIoEngine()
        engine = _make_spdk_engine(tmp_path, fake)
        try:
            assert engine.read_batch_keys() == _DEFAULT_SPDK_READ_BATCH_KEYS
            assert _DEFAULT_SPDK_READ_BATCH_KEYS <= _MAX_SPDK_READ_BATCH_KEYS_CAP
        finally:
            engine.close()


@requires_rust_raw_block_io
class TestRecoveryHeaderDispatch:
    """Tests for slot-header recovery dispatch to the batched read path.

    These verify that ``RawBlockCore._read_slot_headers`` routes SPDK reads to
    the shared batched path, submits one read per header through the engine and
    decodes them, falling back to serial reads on failure so no header is lost.
    """

    def _make_spdk_core(self, tmp_path):
        """Build a core configured for SPDK with an injected fake engine.

        The core is configured with ``io_engine='spdk'`` so recovery dispatch
        routes through the shared batched read path, and the real engine is
        replaced by an injected fake-FFI-backed :class:`SpdkBlockIO`.
        """
        config = _make_posix_config(tmp_path)
        core = RawBlockCore(config, key_namespace="object")
        core.io_engine = "spdk"
        fake = _FakeSpdkIoEngine(submit_rc=0, wait_status=0)
        engine = _make_spdk_engine(tmp_path, fake)
        core._io_engine = engine
        return core, engine, fake

    def test_batched_reads_decode_headers(self, tmp_path):
        """The batched spdk path submits one read per header and decodes them."""
        core, _engine, fake = self._make_spdk_core(tmp_path)
        try:
            offsets = [0x1000, 0x2000, 0x3000]
            headers = core._read_slot_headers(offsets)
            assert len(fake.batch_io_submit_calls) == 1
            offs, lens, _ptrs, count, op = fake.batch_io_submit_calls[0]
            assert offs == offsets
            assert lens == [RAW_BLOCK_CI_HEADER_BYTES] * 3
            assert count == 3
            assert op == IO_READ
            assert len(headers) == 3
            assert [identity for identity, _ in headers] == [0, 1, 2]
        finally:
            core.close()

    def test_batched_falls_back_to_serial_on_failure(self, tmp_path):
        """A failed batch falls back to serial per-slot reads (no data loss)."""
        core, _engine, _fake = self._make_spdk_core(tmp_path)
        fake_fail = _FakeSpdkIoEngine(submit_rc=0, wait_status=-1)
        core._io_engine = _make_spdk_engine(tmp_path, fake_fail)
        try:
            serial = MagicMock(return_value=(7, 7))
            with patch.object(core, "_read_slot_header", serial):
                headers = core._read_slot_headers([0x1000, 0x2000])
            called_offsets = {c.args[0] for c in serial.call_args_list}
            assert called_offsets == {0x1000, 0x2000}
            assert all(header == (7, 7) for header in headers)
        finally:
            core.close()

    def test_single_offset_still_decodes(self, tmp_path):
        """A single offset is read and decoded through the serial dispatch path."""
        core, _engine, _fake = self._make_spdk_core(tmp_path)
        try:
            headers = core._read_slot_headers([0x1000])
            assert headers == [(0, RAW_BLOCK_CI_HEADER_BYTES)]
        finally:
            core.close()
