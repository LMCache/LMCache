# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the GDS cuFile context (``GDSContext``).

Most tests are pure (no cuFile): they exercise the public interface
(singleton/no-op semantics, the <=16 MiB region split observed at the ``ca``
cuFile seam, and the registered-region mapping driven through
:meth:`GDSContext.transfer_async`). The ``test_gds_*_roundtrip`` tests
exercise the real cuFile DMA path and are skipped unless CUDA + nvidia-fs
(real GDS) are present.
"""

# Standard
from types import SimpleNamespace
import os

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.memory_manager import GDSL1MemoryManager
from lmcache.v1.gpu_connector import _cufile_async as ca
from lmcache.v1.gpu_connector.gds_context import (
    GDSContext,
    SlabDirection,
    get_gds_context,
    initialize_gds_context,
)


def _fake_stream(handle: int):
    """A stand-in for ``torch_dev.current_stream()`` (no CUDA needed)."""
    return SimpleNamespace(cuda_stream=handle, synchronize=lambda: None)


requires_gds = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.path.exists("/proc/driver/nvidia-fs/stats")),
    reason="needs CUDA + nvidia-fs (real GPUDirect Storage)",
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Drop the process-global GDSContext between tests."""
    get_gds_context.cache_clear()
    yield
    get_gds_context.cache_clear()


class TestSingleton:
    def test_singleton_identity(self):
        assert get_gds_context() is get_gds_context()

    def test_fresh_context_is_off(self):
        assert GDSContext().initialized is False

    def test_initialize_with_none_is_noop(self):
        ctx = initialize_gds_context(None)
        assert ctx is get_gds_context()
        assert ctx.initialized is False


class TestRegisterGpuBuffer:
    def test_noop_when_uninitialized(self, monkeypatch):
        ctx = GDSContext()
        registered = []
        monkeypatch.setattr(ca, "register_buffer", registered.append)
        # GDS off -> registers nothing, makes no cuFile calls.
        ctx.register_gpu_buffer(torch.empty(4096, dtype=torch.uint8))
        assert registered == []

    def test_splits_buffer_into_regions(self, monkeypatch):
        ctx = GDSContext()
        ctx.initialized = True
        # Record each cuFile registration's byte size at the ca seam.
        sizes = []
        monkeypatch.setattr(
            ca,
            "register_buffer",
            lambda buf: sizes.append(buf.numel() * buf.element_size()),
        )
        monkeypatch.setattr(ca, "register_stream", lambda raw: None)
        monkeypatch.setattr(torch_dev, "current_stream", lambda: _fake_stream(0))

        # The whole buffer is registered in <=16 MiB regions, irrespective of
        # any chunk/slot layout. A 40 MiB buffer -> 16 + 16 + 8 MiB.
        # A CPU tensor is fine: the cuFile calls are mocked.
        buf = torch.empty(40 << 20, dtype=torch.uint8)
        ctx.register_gpu_buffer(buf)

        assert sizes == [16 << 20, 16 << 20, 8 << 20]


class TestResolveBuffer:
    """Region mapping: a buffer slice resolves to ``(region base, offset)``,
    exercised through the public ``transfer_async`` path."""

    def _registered_ctx(self, monkeypatch, buf: torch.Tensor):
        """Register ``buf``; capture the ``(base, offset)`` that
        ``transfer_async`` resolves a slice to before handing it to the slab."""
        ctx = GDSContext()
        ctx.initialized = True
        monkeypatch.setattr(ca, "register_buffer", lambda b: None)
        monkeypatch.setattr(ca, "register_stream", lambda raw: None)
        monkeypatch.setattr(torch_dev, "current_stream", lambda: _fake_stream(0))
        ctx.register_gpu_buffer(buf)
        resolved: list[tuple[int, int]] = []
        monkeypatch.setattr(
            ctx,
            "_slab_write",
            lambda slab_offset, size, dev_offset, buf_base: resolved.append(
                (buf_base, dev_offset)
            ),
        )
        return ctx, resolved

    def test_maps_slice_to_base_and_offset(self, monkeypatch):
        buf = torch.empty(8192, dtype=torch.uint8)
        ctx, resolved = self._registered_ctx(monkeypatch, buf)
        mem_obj = SimpleNamespace(get_size=lambda: 4096, slab_offset=0)
        # A slice 4 KiB into the region must map to (region base, offset 4096).
        ctx.transfer_async(mem_obj, buf[4096:], SlabDirection.WRITE)
        assert resolved == [(buf.data_ptr(), 4096)]


class TestPerStreamRegistration:
    """Each distinct stream is cuFile-registered once and deregistered once its
    last region is gone -- observed at the ``ca`` seam (no private state)."""

    def test_register_and_deregister_per_stream(self, monkeypatch):
        ctx = GDSContext()
        ctx.initialized = True
        reg_str: list[int] = []
        dereg_str: list[int] = []
        dereg_buf: list[int] = []
        monkeypatch.setattr(ca, "register_buffer", lambda b: None)
        monkeypatch.setattr(
            ca, "deregister_buffer", lambda b: dereg_buf.append(b.data_ptr())
        )
        monkeypatch.setattr(ca, "register_stream", reg_str.append)
        monkeypatch.setattr(ca, "deregister_stream", dereg_str.append)

        def use_stream(handle: int):
            monkeypatch.setattr(
                torch_dev, "current_stream", lambda: _fake_stream(handle)
            )

        buf_a = torch.empty(24 << 20, dtype=torch.uint8)  # 2 regions on stream 11
        buf_b = torch.empty(4096, dtype=torch.uint8)  # 1 region on stream 22
        use_stream(11)
        ctx.register_gpu_buffer(buf_a)
        use_stream(22)
        ctx.register_gpu_buffer(buf_b)
        # Each distinct stream registered exactly once.
        assert reg_str == [11, 22]

        # Deregistering buf_b frees stream 22's only region -> stream 22 dropped.
        use_stream(22)
        ctx.deregister_gpu_buffer(buf_b)
        assert dereg_str == [22]
        # Stream 11 still has 2 regions (24 MiB -> 16 + 8), so not yet dropped.
        use_stream(11)
        ctx.deregister_gpu_buffer(buf_a)
        assert dereg_str == [22, 11]
        assert len(dereg_buf) == 3  # all three slots deregistered


@requires_gds
def test_gds_two_stream_write_read(tmp_path):
    """Two CUDA streams each register their own buffer and round-trip a chunk
    through real cuFile DMA; verify the data stays isolated per stream."""
    cfg = GdsL1Config(file_location=str(tmp_path), size_in_bytes=64 << 20)
    chunk_bytes = 8 << 20
    ctx = GDSContext()
    ctx.initialize(cfg)
    mgr = GDSL1MemoryManager(cfg)

    def register_and_write(stream, pattern):
        """Register a buffer on ``stream`` and write ``pattern`` to a chunk."""
        with torch.cuda.stream(stream):
            buf = torch.empty(chunk_bytes, dtype=torch.uint8, device="cuda")
            ctx.register_gpu_buffer(buf)
            err, objs = mgr.allocate(
                MemoryLayoutDesc(
                    shapes=[torch.Size([chunk_bytes])], dtypes=[torch.uint8]
                ),
                1,
            )
            assert err == L1Error.SUCCESS
            buf.fill_(pattern)
            torch.cuda.synchronize()
            ctx.transfer_async(objs[0], buf, SlabDirection.WRITE)
            torch.cuda.synchronize()
        return buf, objs[0]

    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    try:
        buf_a, mem_a = register_and_write(stream_a, 0xA1)
        buf_b, mem_b = register_and_write(stream_b, 0xB2)

        # Read each chunk back on its own stream; each must see its own pattern,
        # confirming the two streams' buffers/regions don't clobber each other.
        for stream, buf, mem, pattern in (
            (stream_a, buf_a, mem_a, 0xA1),
            (stream_b, buf_b, mem_b, 0xB2),
        ):
            with torch.cuda.stream(stream):
                buf.zero_()
                torch.cuda.synchronize()
                ctx.transfer_async(mem, buf, SlabDirection.READ)
                torch.cuda.synchronize()
                expected = torch.full((chunk_bytes,), pattern, dtype=torch.uint8)
                assert torch.equal(buf.cpu(), expected)

        # Deregister each buffer on its own stream.
        for stream, buf in ((stream_a, buf_a), (stream_b, buf_b)):
            with torch.cuda.stream(stream):
                ctx.deregister_gpu_buffer(buf)
    finally:
        ctx.close()


@requires_gds
def test_gds_write_read_roundtrip(tmp_path):
    """Cold write then read of a chunk through the real cuFile DMA path."""
    cfg = GdsL1Config(file_location=str(tmp_path), size_in_bytes=64 << 20)
    ctx = GDSContext()
    ctx.initialize(cfg)
    try:
        chunk_bytes = 8 << 20
        buf = torch.empty(chunk_bytes, dtype=torch.uint8, device="cuda")
        ctx.register_gpu_buffer(buf)

        mgr = GDSL1MemoryManager(cfg)
        err, objs = mgr.allocate(
            MemoryLayoutDesc(shapes=[torch.Size([chunk_bytes])], dtypes=[torch.uint8]),
            1,
        )
        assert err == L1Error.SUCCESS
        mem_obj = objs[0]

        buf.fill_(0xAB)
        torch.cuda.synchronize()
        ctx.transfer_async(mem_obj, buf, SlabDirection.WRITE)

        buf.zero_()
        torch.cuda.synchronize()
        ctx.transfer_async(mem_obj, buf, SlabDirection.READ)
        torch.cuda.synchronize()

        expected = torch.full((chunk_bytes,), 0xAB, dtype=torch.uint8)
        assert torch.equal(buf.cpu(), expected)
    finally:
        ctx.close()


@requires_gds
def test_gds_chunk_larger_than_region_roundtrip(tmp_path):
    """A chunk larger than the 16 MiB cuFile region cap round-trips correctly.

    Exercises the multi-region registration and the split (per-segment) DMA
    path: a 24 MiB chunk is registered/transferred as a 16 MiB + 8 MiB pair.
    """
    cfg = GdsL1Config(file_location=str(tmp_path), size_in_bytes=64 << 20)
    ctx = GDSContext()
    ctx.initialize(cfg)
    try:
        chunk_bytes = 24 << 20  # > 16 MiB -> two registered regions / two DMAs
        buf = torch.empty(chunk_bytes, dtype=torch.uint8, device="cuda")
        ctx.register_gpu_buffer(buf)

        mgr = GDSL1MemoryManager(cfg)
        err, objs = mgr.allocate(
            MemoryLayoutDesc(shapes=[torch.Size([chunk_bytes])], dtypes=[torch.uint8]),
            1,
        )
        assert err == L1Error.SUCCESS
        mem_obj = objs[0]

        # Position-dependent pattern: a mis-offset or swapped segment (e.g. the
        # second segment using the wrong slab offset) would corrupt the bytes
        # around the 16 MiB boundary, which a uniform fill would not catch.
        pattern = (torch.arange(chunk_bytes, dtype=torch.int64) % 251).to(torch.uint8)
        buf.copy_(pattern.cuda())
        torch.cuda.synchronize()
        ctx.transfer_async(mem_obj, buf, SlabDirection.WRITE)

        buf.zero_()
        torch.cuda.synchronize()
        ctx.transfer_async(mem_obj, buf, SlabDirection.READ)
        torch.cuda.synchronize()

        assert torch.equal(buf.cpu(), pattern)
    finally:
        ctx.close()
