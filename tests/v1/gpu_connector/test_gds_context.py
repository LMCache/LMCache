# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the GDS cuFile context (``GDSContext``).

Most tests are pure (no cuFile): the singleton/no-op semantics, the per-slot
split, slab-size rounding, and ``_resolve_buffer`` mapping. The final
``test_gds_write_read_roundtrip`` exercises the real cuFile DMA path and is
skipped unless CUDA + nvidia-fs (real GDS) are present.
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
    def test_noop_when_uninitialized(self):
        ctx = GDSContext()
        # GDS off -> registers nothing, touches no cuFile.
        ctx.register_gpu_buffer(torch.empty(4096, dtype=torch.uint8), 4096)
        assert ctx._buffers == []
        assert ctx._base_ptrs == []

    def test_splits_buffer_into_slots(self, monkeypatch):
        ctx = GDSContext()
        ctx.initialized = True
        monkeypatch.setattr(ca, "register_buffer", lambda buf: None)
        monkeypatch.setattr(ctx, "_ensure_stream_registered", lambda: None)

        # 4 slots of 4 KiB. A CPU tensor is fine: the cuFile calls are mocked
        # and _register_region no longer device-checks.
        buf = torch.empty(4 * 4096, dtype=torch.uint8)
        ctx.register_gpu_buffer(buf, 4096)

        assert len(ctx._buffers) == 4
        assert ctx._nbytes == [4096, 4096, 4096, 4096]
        assert ctx._base_ptrs == sorted(ctx._base_ptrs)


class TestResolveBuffer:
    def _ctx_with_region(self, base: int, size: int) -> GDSContext:
        ctx = GDSContext()
        ctx._base_ptrs = [base]
        ctx._nbytes = [size]
        return ctx

    def test_maps_slice_to_base_and_offset(self):
        ctx = self._ctx_with_region(0x1000, 8192)
        buf = SimpleNamespace(data_ptr=lambda: 0x1000 + 4096)
        assert ctx._resolve_buffer(buf) == (0x1000, 4096)

    def test_pointer_past_region_raises(self):
        ctx = self._ctx_with_region(0x1000, 8192)
        with pytest.raises(ValueError):
            ctx._resolve_buffer(SimpleNamespace(data_ptr=lambda: 0x1000 + 8192))

    def test_pointer_below_region_raises(self):
        ctx = self._ctx_with_region(0x1000, 8192)
        with pytest.raises(ValueError):
            ctx._resolve_buffer(SimpleNamespace(data_ptr=lambda: 0x10))


class TestInitializeRounding:
    def test_slab_size_rounded_up_to_alignment(self, monkeypatch, tmp_path):
        ctx = GDSContext()
        monkeypatch.setattr(ctx, "_open_and_register_slab", lambda use_direct_io: None)
        ctx.initialize(
            GdsL1Config(file_location=str(tmp_path), size_in_bytes=3 * 4096 + 1)
        )
        assert ctx.initialized is True
        assert ctx._slab_size == 4 * 4096
        assert ctx._slab_path.endswith("lmcache_gds_slab.bin")


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
