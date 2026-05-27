# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the slab-file GDS L1 backend.

Focus is on the parts that do not require a real cuFile driver:

- :class:`SlabAddressManager` — allocator semantics, coalescing,
  OOM, ``mark_used`` overlap rejection.
- :class:`GdsL1Backend` — lookup / create_memory_obj / free_entry,
  ``get_memory_usage``, index persistence + reload.
- :class:`GdsMemoryObj` — disk-anchored surface (``tensor`` is None,
  ``byte_array`` / ``data_ptr`` raise).

Tests skip themselves on hosts without CUDA. Where I/O is exercised
end-to-end we force ``use_gds=False`` so the backend takes the POSIX
fallback path; that path uses the same address manager and the same
``GdsScratchAllocator`` surface, so it's a meaningful cross-check
without needing nvidia-fs loaded in CI.
"""

# Standard
import asyncio
import os
import shutil
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.distributed.gds_l1 import (
    GdsL1Backend,
    GdsMemoryObj,
    SlabAddressManager,
    _CUFILE_ALIGNMENT,
)
from lmcache.v1.memory_management import MemoryFormat

# --- Fixtures --------------------------------------------------------


@pytest.fixture
def gds_root(tmp_path):
    """A scratch directory usable as ``gds_path``."""
    root = tmp_path / "gds_l1_root"
    root.mkdir()
    yield str(root)
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def loop():
    """A fresh asyncio event loop running in a background thread.

    The slab design doesn't use the loop today, but the constructor
    still requires one to preserve API compatibility with the previous
    async-scan design.
    """
    new_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=new_loop.run_forever, daemon=True)
    thread.start()
    yield new_loop
    new_loop.call_soon_threadsafe(new_loop.stop)
    thread.join(timeout=2.0)
    new_loop.close()


def _make_config(gds_path: str, slab_size_gb: float = 0.25) -> GdsL1Config:
    """Construct a minimal :class:`GdsL1Config` for tests.

    Forces ``use_gds=False`` so tests run through the POSIX fallback;
    that path exercises the same allocator + memory-obj contract as
    the cuFile path but doesn't need nvidia-fs loaded.

    Args:
        gds_path: Test scratch directory.
        slab_size_gb: Slab size to allocate. Default 0.25 GiB
            (256 MiB) — small enough to keep tests fast but big enough
            for multiple-chunk scenarios.
    """
    return GdsL1Config(
        gds_path=gds_path,
        gds_path_sharding="by_gpu",
        use_gds=False,
        use_direct_io=False,
        slab_size_gb=slab_size_gb,
    )


def _object_key(seed: int = 0) -> ObjectKey:
    """Return a deterministic ``ObjectKey`` for tests."""
    return ObjectKey(
        chunk_hash=(seed.to_bytes(4, "big") + b"\0" * 28),
        model_name=f"test-model-{seed}",
        kv_rank=0,
    )


def _layout(shape: torch.Size, dtype: torch.dtype = torch.float16) -> MemoryLayoutDesc:
    return MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])


# --- SlabAddressManager ---------------------------------------------


class TestSlabAddressManager:
    """Allocator behaves like a first-fit free-list."""

    def test_allocate_basic(self):
        sm = SlabAddressManager(total_size=64 * 1024)
        off = sm.allocate(4096)
        assert off == 0
        assert sm.used_bytes() == 4096
        assert sm.free_bytes() == 64 * 1024 - 4096

    def test_allocate_aligns_up(self):
        sm = SlabAddressManager(total_size=64 * 1024)
        # Request below the 4 KiB alignment rounds up.
        sm.allocate(100)
        assert sm.used_bytes() == 4096

    def test_allocate_returns_none_on_oom(self):
        sm = SlabAddressManager(total_size=4096)
        first = sm.allocate(4096)
        assert first == 0
        assert sm.allocate(4096) is None

    def test_free_coalesces_adjacent_regions(self):
        sm = SlabAddressManager(total_size=64 * 1024)
        a = sm.allocate(8192)
        b = sm.allocate(8192)
        c = sm.allocate(8192)
        sm.free(a, 8192)
        sm.free(c, 8192)
        # Two disjoint free regions plus the tail: free_bytes = 16K + tail.
        sm.free(b, 8192)
        # Everything coalesced back to a single region.
        assert sm.used_bytes() == 0
        # And the next big allocation should fit at offset 0.
        assert sm.allocate(64 * 1024) == 0

    def test_mark_used_carves_region(self):
        sm = SlabAddressManager(total_size=64 * 1024)
        sm.mark_used(8192, 4096)
        # Allocation should skip past [8192, 12288).
        off = sm.allocate(16 * 1024)
        # First-fit picks [0, 8192) which isn't big enough — moves on
        # and picks the tail region.
        assert off == 12288

    def test_mark_used_rejects_overlap(self):
        sm = SlabAddressManager(total_size=64 * 1024)
        sm.allocate(4096)
        with pytest.raises(RuntimeError):
            sm.mark_used(0, 4096)


# --- GdsMemoryObj surface --------------------------------------------


class TestGdsMemoryObjSurface:
    """``.tensor`` is None, ``.byte_array`` / ``.data_ptr`` raise."""

    @pytest.fixture
    def backend(self, gds_root, loop):
        b = GdsL1Backend(_make_config(gds_root), loop=loop, dst_device="cuda:0")
        yield b
        b.close()

    def test_tensor_is_none(self, backend):
        mo = backend.create_memory_obj(
            key=_object_key(seed=1),
            layout_desc=_layout(torch.Size([128, 64])),
        )
        assert mo is not None
        assert mo.tensor is None
        assert mo.raw_tensor is None
        assert mo.get_tensor(0) is None

    def test_byte_array_raises(self, backend):
        mo = backend.create_memory_obj(
            key=_object_key(seed=2),
            layout_desc=_layout(torch.Size([128, 64])),
        )
        with pytest.raises(NotImplementedError):
            _ = mo.byte_array

    def test_data_ptr_raises(self, backend):
        mo = backend.create_memory_obj(
            key=_object_key(seed=3),
            layout_desc=_layout(torch.Size([128, 64])),
        )
        with pytest.raises(NotImplementedError):
            _ = mo.data_ptr


# --- GdsL1Backend lookup / create / free -----------------------------


class TestGdsL1BackendIndex:
    """``lookup`` / ``create_memory_obj`` / ``record_entry`` / ``free_entry``."""

    @pytest.fixture
    def backend(self, gds_root, loop):
        b = GdsL1Backend(_make_config(gds_root), loop=loop, dst_device="cuda:0")
        yield b
        b.close()

    def test_lookup_empty(self, backend):
        keys = [_object_key(seed=i) for i in range(3)]
        assert backend.lookup(keys) == [False, False, False]

    def test_create_and_record(self, backend):
        key = _object_key(seed=42)
        mo = backend.create_memory_obj(
            key=key, layout_desc=_layout(torch.Size([4096]), torch.uint8)
        )
        assert mo is not None
        assert mo.size == 4096
        # Not yet recorded.
        assert backend.lookup([key]) == [False]
        backend.record_entry(mo)
        assert backend.lookup([key]) == [True]
        assert backend.get_hot_cache_size() == 1

    def test_free_drops_index_entry(self, backend):
        key = _object_key(seed=7)
        mo = backend.create_memory_obj(
            key=key, layout_desc=_layout(torch.Size([4096]), torch.uint8)
        )
        backend.record_entry(mo)
        backend.free_entry(mo)
        assert backend.lookup([key]) == [False]

    def test_create_memory_obj_from_index(self, backend):
        key = _object_key(seed=8)
        mo = backend.create_memory_obj(
            key=key, layout_desc=_layout(torch.Size([4096]), torch.uint8)
        )
        backend.record_entry(mo)
        resurrected = backend.create_memory_obj_from_index(key)
        assert resurrected is not None
        assert resurrected.slab_offset == mo.slab_offset
        assert resurrected.size == mo.size

    def test_oom_returns_none(self, gds_root, loop):
        # Tiny slab — one 4 KiB chunk fits, the second must OOM.
        b = GdsL1Backend(
            _make_config(gds_root, slab_size_gb=4096 / (1 << 30)),
            loop=loop,
            dst_device="cuda:0",
        )
        try:
            mo1 = b.create_memory_obj(
                key=_object_key(seed=1), layout_desc=_layout(torch.Size([4096]), torch.uint8)
            )
            assert mo1 is not None
            mo2 = b.create_memory_obj(
                key=_object_key(seed=2), layout_desc=_layout(torch.Size([4096]), torch.uint8)
            )
            assert mo2 is None
        finally:
            b.close()

    def test_get_memory_usage(self, backend):
        used, total = backend.get_memory_usage()
        assert used == 0
        assert total >= _CUFILE_ALIGNMENT
        mo = backend.create_memory_obj(
            key=_object_key(seed=99),
            layout_desc=_layout(torch.Size([4096]), torch.uint8),
        )
        backend.record_entry(mo)
        used2, total2 = backend.get_memory_usage()
        assert used2 == 4096
        assert total2 == total


# --- Index persistence ------------------------------------------------


class TestGdsL1BackendPersistence:
    """Index survives backend close + reopen."""

    def test_persist_and_reload(self, gds_root, loop):
        b1 = GdsL1Backend(_make_config(gds_root), loop=loop, dst_device="cuda:0")
        try:
            key = _object_key(seed=123)
            mo = b1.create_memory_obj(
                key=key,
                layout_desc=_layout(torch.Size([8192]), torch.uint8),
                fmt=MemoryFormat.KV_2LTD,
            )
            b1.record_entry(mo)
            slab_offset = mo.slab_offset
        finally:
            b1.close()

        # New backend over the same path — index loads, the region is
        # still marked used, and lookup succeeds.
        b2 = GdsL1Backend(_make_config(gds_root), loop=loop, dst_device="cuda:0")
        try:
            assert b2.lookup([key]) == [True]
            resurrected = b2.create_memory_obj_from_index(key)
            assert resurrected is not None
            assert resurrected.slab_offset == slab_offset
            assert resurrected.size == 8192
            used, _ = b2.get_memory_usage()
            assert used == 8192
        finally:
            b2.close()

    def test_corrupt_index_starts_empty(self, gds_root, loop):
        # Write a deliberately-broken index file.
        with open(os.path.join(gds_root, "lmcache_gds_index.json"), "w") as f:
            f.write("not a json document")
        b = GdsL1Backend(_make_config(gds_root), loop=loop, dst_device="cuda:0")
        try:
            assert b.get_hot_cache_size() == 0
        finally:
            b.close()


# --- POSIX round-trip (CUDA required) --------------------------------

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="POSIX round-trip uses cudaMemcpy and needs a CUDA device",
)


@cuda_required
class TestPosixRoundTrip:
    """End-to-end write→read via the POSIX fallback.

    Same ``GdsScratchAllocator`` code path the production cuFile path
    uses; ``use_gds=False`` swaps the bottom layer for pread/pwrite +
    cudaMemcpy so this can run on any CUDA host (no nvidia-fs needed).
    """

    def test_write_then_read_matches(self, gds_root, loop):
        b = GdsL1Backend(_make_config(gds_root), loop=loop, dst_device="cuda:0")
        try:
            chunk_bytes = 8192
            buf = torch.empty(chunk_bytes, dtype=torch.uint8, device="cuda:0")
            b.scratch_allocator.register_gpu_buffer(buf)
            try:
                buf.fill_(0xAB)
                torch.cuda.synchronize()

                key = _object_key(seed=777)
                mo = b.create_memory_obj(
                    key=key,
                    layout_desc=_layout(torch.Size([chunk_bytes]), torch.uint8),
                )
                b.scratch_allocator.cufile_write_from(mo, buf)

                buf.zero_()
                torch.cuda.synchronize()
                b.scratch_allocator.cufile_read_into(mo, buf)
                torch.cuda.synchronize()
                expected = torch.full(
                    (chunk_bytes,), 0xAB, dtype=torch.uint8
                )
                assert torch.equal(buf.cpu(), expected)
            finally:
                b.scratch_allocator.deregister_gpu_buffer()
        finally:
            b.close()
