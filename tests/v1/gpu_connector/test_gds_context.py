# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the GDS cuFile context (``GDSContext``).

Most tests are pure (no cuFile): they exercise the public interface
(singleton/no-op semantics, the per-slot split observed at the ``ca`` cuFile
seam, and the registered-region mapping driven through
:meth:`GDSContext.write_async`). The final ``test_gds_write_read_roundtrip``
exercises the real cuFile DMA path and is skipped unless CUDA + nvidia-fs
(real GDS) are present.
"""

# Standard
from types import SimpleNamespace
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.memory_manager import GDSL1MemoryManager
from lmcache.v1.gpu_connector import _cufile_async as ca
from lmcache.v1.gpu_connector.gds_context import (
    GDSContext,
    get_gds_context,
    initialize_gds_context,
)

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
        ctx.register_gpu_buffer(torch.empty(4096, dtype=torch.uint8), 4096)
        assert registered == []

    def test_splits_buffer_into_slots(self, monkeypatch):
        ctx = GDSContext()
        ctx.initialized = True
        # Record each cuFile registration's byte size at the ca seam.
        sizes = []
        monkeypatch.setattr(
            ca,
            "register_buffer",
            lambda buf: sizes.append(buf.numel() * buf.element_size()),
        )
        monkeypatch.setattr(ctx, "_ensure_stream_registered", lambda: None)

        # 4 slots of 4 KiB. A CPU tensor is fine: the cuFile calls are mocked.
        buf = torch.empty(4 * 4096, dtype=torch.uint8)
        ctx.register_gpu_buffer(buf, 4096)

        # Each slot is registered separately, all the requested slot size.
        assert sizes == [4096, 4096, 4096, 4096]


class TestResolveBuffer:
    """Region mapping, exercised through the public ``write_async`` path."""

    def _registered_ctx(self, monkeypatch, buf: torch.Tensor):
        """Register ``buf`` as one slot; capture the ``(base, offset)`` that
        ``write_async`` resolves a slice to before handing it to the slab."""
        ctx = GDSContext()
        ctx.initialized = True
        monkeypatch.setattr(ca, "register_buffer", lambda b: None)
        monkeypatch.setattr(ctx, "_ensure_stream_registered", lambda: None)
        ctx.register_gpu_buffer(buf, buf.numel())
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
        ctx.write_async(mem_obj, buf[4096:])
        assert resolved == [(buf.data_ptr(), 4096)]

    def test_pointer_past_region_raises(self, monkeypatch):
        buf = torch.empty(8192, dtype=torch.uint8)
        ctx, _ = self._registered_ctx(monkeypatch, buf)
        mem_obj = SimpleNamespace(get_size=lambda: 4096, slab_offset=0)
        past = SimpleNamespace(data_ptr=lambda: buf.data_ptr() + 8192)
        with pytest.raises(ValueError):
            ctx.write_async(mem_obj, past)

    def test_pointer_below_region_raises(self, monkeypatch):
        buf = torch.empty(8192, dtype=torch.uint8)
        ctx, _ = self._registered_ctx(monkeypatch, buf)
        mem_obj = SimpleNamespace(get_size=lambda: 4096, slab_offset=0)
        below = SimpleNamespace(data_ptr=lambda: 0x10)
        with pytest.raises(ValueError):
            ctx.write_async(mem_obj, below)


@requires_gds
def test_gds_write_read_roundtrip(tmp_path):
    """Cold write then read of a chunk through the real cuFile DMA path."""
    cfg = GdsL1Config(file_location=str(tmp_path), size_in_bytes=64 << 20)
    ctx = GDSContext()
    ctx.initialize(cfg)
    try:
        chunk_bytes = 8 << 20
        buf = torch.empty(chunk_bytes, dtype=torch.uint8, device="cuda")
        ctx.register_gpu_buffer(buf, chunk_bytes)

        mgr = GDSL1MemoryManager(cfg)
        err, objs = mgr.allocate(
            MemoryLayoutDesc(shapes=[torch.Size([chunk_bytes])], dtypes=[torch.uint8]),
            1,
        )
        assert err == L1Error.SUCCESS
        mem_obj = objs[0]

        buf.fill_(0xAB)
        torch.cuda.synchronize()
        ctx.write_async(mem_obj, buf)

        buf.zero_()
        torch.cuda.synchronize()
        ctx.read_async(mem_obj, buf)
        torch.cuda.synchronize()

        expected = torch.full((chunk_bytes,), 0xAB, dtype=torch.uint8)
        assert torch.equal(buf.cpu(), expected)
    finally:
        ctx.close()
