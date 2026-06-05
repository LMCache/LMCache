# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the slab-file GDS L1 backend.

Focus is on the parts that do not require a real cuFile driver:

- :class:`SlabAddressManager` — allocator semantics, coalescing,
  OOM, ``mark_used`` overlap rejection.
- :class:`GdsL1Backend` — lookup / create_memory_obj / free_entry_from_index,
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
import os
import shutil

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.distributed.gds_l1 import (
    _CUFILE_ALIGNMENT,
    GdsL1Backend,
    SlabAddressManager,
)

# --- Fixtures --------------------------------------------------------


@pytest.fixture
def gds_root(tmp_path):
    """A scratch directory usable as ``gds_path``."""
    root = tmp_path / "gds_l1_root"
    root.mkdir()
    yield str(root)
    shutil.rmtree(root, ignore_errors=True)


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
        # Freeing the middle region coalesces all three.
        sm.free(b, 8192)
        assert sm.used_bytes() == 0
        assert sm.allocate(64 * 1024) == 0

    def test_mark_used_carves_region(self):
        sm = SlabAddressManager(total_size=64 * 1024)
        sm.mark_used(8192, 4096)
        # [0, 8192) too small for 16 KiB; first-fit lands at 12288.
        off = sm.allocate(16 * 1024)
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
    def backend(self, gds_root):
        b = GdsL1Backend(_make_config(gds_root), dst_device="cuda:0")
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
    """``lookup`` / ``create_memory_obj`` / ``record_entry`` /
    ``free_entry_from_index``."""

    @pytest.fixture
    def backend(self, gds_root):
        b = GdsL1Backend(_make_config(gds_root), dst_device="cuda:0")
        yield b
        b.close()

    def test_missing_keys_not_in_index(self, backend):
        keys = [_object_key(seed=i) for i in range(3)]
        assert all(backend.create_memory_obj_from_index(k) is None for k in keys)

    def test_create_and_record(self, backend):
        key = _object_key(seed=42)
        mo = backend.create_memory_obj(
            key=key, layout_desc=_layout(torch.Size([4096]), torch.uint8)
        )
        assert mo is not None
        assert mo.get_size() == 4096
        # Not in the index until _record_entry.
        assert backend.create_memory_obj_from_index(key) is None
        backend._record_entry(mo)
        assert backend.create_memory_obj_from_index(key) is not None
        assert len(backend._index) == 1

    def test_free_drops_index_entry(self, backend):
        key = _object_key(seed=7)
        mo = backend.create_memory_obj(
            key=key, layout_desc=_layout(torch.Size([4096]), torch.uint8)
        )
        backend._record_entry(mo)
        backend.free_entry_from_index(mo)
        assert backend.create_memory_obj_from_index(key) is None

    def test_create_memory_obj_from_index(self, backend):
        key = _object_key(seed=8)
        mo = backend.create_memory_obj(
            key=key, layout_desc=_layout(torch.Size([4096]), torch.uint8)
        )
        backend._record_entry(mo)
        resurrected = backend.create_memory_obj_from_index(key)
        assert resurrected is not None
        assert resurrected.slab_offset == mo.slab_offset
        assert resurrected.get_size() == mo.get_size()

    def test_oom_returns_none(self, gds_root):
        # Slab sized to fit exactly one 4 KiB chunk.
        b = GdsL1Backend(
            _make_config(gds_root, slab_size_gb=4096 / (1 << 30)),
            dst_device="cuda:0",
        )
        try:
            mo1 = b.create_memory_obj(
                key=_object_key(seed=1),
                layout_desc=_layout(torch.Size([4096]), torch.uint8),
            )
            assert mo1 is not None
            mo2 = b.create_memory_obj(
                key=_object_key(seed=2),
                layout_desc=_layout(torch.Size([4096]), torch.uint8),
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
        backend._record_entry(mo)
        used2, total2 = backend.get_memory_usage()
        assert used2 == 4096
        assert total2 == total


# --- Index persistence ------------------------------------------------


class TestGdsL1BackendPersistence:
    """Index survives backend close + reopen."""

    def test_persist_and_reload(self, gds_root):
        b1 = GdsL1Backend(_make_config(gds_root), dst_device="cuda:0")
        try:
            key = _object_key(seed=123)
            mo = b1.create_memory_obj(
                key=key,
                layout_desc=_layout(torch.Size([8192]), torch.uint8),
            )
            b1._record_entry(mo)
            slab_offset = mo.slab_offset
        finally:
            b1.close()

        # Reopen over the same path.
        b2 = GdsL1Backend(_make_config(gds_root), dst_device="cuda:0")
        try:
            resurrected = b2.create_memory_obj_from_index(key)
            assert resurrected is not None
            assert resurrected.slab_offset == slab_offset
            assert resurrected.get_size() == 8192
            used, _ = b2.get_memory_usage()
            assert used == 8192
        finally:
            b2.close()

    def test_corrupt_index_starts_empty(self, gds_root):
        # Write a corrupt index file before opening.
        with open(os.path.join(gds_root, "lmcache_gds_index.json"), "w") as f:
            f.write("not a json document")
        b = GdsL1Backend(_make_config(gds_root), dst_device="cuda:0")
        try:
            assert len(b._index) == 0
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

    def test_write_then_read_matches(self, gds_root):
        b = GdsL1Backend(_make_config(gds_root), dst_device="cuda:0")
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
                expected = torch.full((chunk_bytes,), 0xAB, dtype=torch.uint8)
                assert torch.equal(buf.cpu(), expected)
            finally:
                b.scratch_allocator.deregister_gpu_buffer()
        finally:
            b.close()
