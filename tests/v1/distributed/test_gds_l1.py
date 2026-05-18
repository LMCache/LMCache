# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the GDS L1 backend module.

Focus is on the parts that do not require cuFile or a real GPU:
metadata header round-trip, disk-path mapping, ``GdsMemoryObj``
surface, handle-cache LRU semantics, and the startup scan + lookup
flow. End-to-end read/write tests against actual cuFile hardware
live in the buildkite GDS lane (see ``.buildkite/k3_tests``).
"""

# Standard
from unittest import mock
import asyncio
import os
import shutil
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.gds_l1 import (
    _METADATA_MAX_SIZE,
    CuFileHandleCache,
    GdsL1Backend,
    GdsMemoryObj,
    GdsScratchAllocator,
    key_to_disk_path,
    pack_metadata,
    unpack_metadata,
)
from lmcache.v1.memory_management import MemoryFormat, MemoryObjMetadata

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

    ``GdsL1Backend`` requires a running loop to schedule its async
    metadata scan; tests get a real one to mirror production.
    """
    new_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=new_loop.run_forever, daemon=True)
    thread.start()
    yield new_loop
    new_loop.call_soon_threadsafe(new_loop.stop)
    thread.join(timeout=2.0)
    new_loop.close()


def _make_config(gds_path: str) -> LMCacheEngineConfig:
    """Construct a minimal :class:`LMCacheEngineConfig` for GDS L1 tests."""
    return LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        gds_path=gds_path,
        gds_path_sharding="by_gpu",
        lmcache_instance_id="test_gds_l1",
        gds_buffer_size=256,
        use_gds=False,  # force POSIX fallback so tests don't need cuFile
        extra_config={"use_direct_io": False},
    )


def _object_key(seed: int = 0) -> ObjectKey:
    """Return a deterministic ``ObjectKey`` for tests."""
    return ObjectKey(
        chunk_hash=(seed.to_bytes(4, "big") + b"\0" * 28),
        model_name=f"test-model-{seed}",
        kv_rank=0,
    )


# --- pack/unpack metadata --------------------------------------------


def test_pack_metadata_roundtrip():
    """``pack_metadata`` -> ``unpack_metadata`` reproduces the inputs."""
    shape = torch.Size([2, 4, 256, 128])
    dtype = torch.float16
    nbytes = shape.numel() * dtype.itemsize
    fmt = MemoryFormat.KV_2LTD

    packed = pack_metadata(
        nbytes=nbytes,
        shape=shape,
        dtype=dtype,
        fmt=fmt,
        lmcache_version="1",
    )

    assert len(packed) == _METADATA_MAX_SIZE
    out_shape, out_dtype, out_nbytes, out_fmt, extra = unpack_metadata(packed)
    assert out_shape == shape
    assert out_dtype == dtype
    assert out_nbytes == nbytes
    assert out_fmt == fmt
    assert extra["lmcache_version"] == "1"


def test_pack_metadata_rejects_unsupported_dtype():
    """``pack_metadata`` raises when handed an unmapped dtype."""
    with pytest.raises(RuntimeError, match="Unsupported dtype"):
        pack_metadata(
            nbytes=4,
            shape=torch.Size([1]),
            dtype=torch.complex64,
            fmt=MemoryFormat.BINARY,
        )


# --- key_to_disk_path ------------------------------------------------


def test_key_to_disk_path_uses_two_level_hashing(tmp_path):
    """Filenames live under ``<hash[:2]>/<hash[2:4]>/<urlquoted_key>.<suffix>``."""
    key = _object_key(seed=0xCAFEBABE)
    full_path, subdir_key, l1, l2 = key_to_disk_path(
        key, base_path=str(tmp_path), data_suffix=".kvcache.safetensors"
    )

    hash_hex = key.chunk_hash.hex()
    assert l1 == hash_hex[:2]
    assert l2 == hash_hex[2:4]
    assert subdir_key == l1 + l2
    assert full_path.startswith(os.path.join(str(tmp_path), l1, l2))
    assert full_path.endswith(".kvcache.safetensors")


# --- GdsMemoryObj surface --------------------------------------------


def _make_scratch_allocator() -> GdsScratchAllocator:
    """Build a scratch allocator without touching the backend.

    For tests that only exercise the ``GdsMemoryObj`` ↔ allocator
    relationship; the allocator's I/O paths are not invoked.
    """
    return GdsScratchAllocator(backend=mock.Mock())


def _make_gds_mem_obj(disk_path: str = "/tmp/fake.safetensors") -> GdsMemoryObj:
    """Construct a ``GdsMemoryObj`` for surface tests."""
    meta = MemoryObjMetadata(
        shape=torch.Size([2, 256, 64]),
        dtype=torch.float16,
        address=0,
        phy_size=2 * 256 * 64 * 2,
        ref_count=0,
        pin_count=0,
        fmt=MemoryFormat.KV_2LTD,
    )
    return GdsMemoryObj(
        key=_object_key(),
        disk_path=disk_path,
        file_offset=_METADATA_MAX_SIZE,
        metadata=meta,
        parent_allocator=_make_scratch_allocator(),
    )


def test_gds_memory_obj_tensor_is_none_at_rest():
    """The disk-anchored object has no live tensor body."""
    mo = _make_gds_mem_obj()
    assert mo.tensor is None
    assert mo.raw_tensor is None
    assert mo.get_tensor(0) is None


def test_gds_memory_obj_byte_array_raises():
    """byte_array is not supported — bytes are on disk, not in memory."""
    mo = _make_gds_mem_obj()
    with pytest.raises(NotImplementedError, match="byte_array"):
        _ = mo.byte_array


def test_gds_memory_obj_data_ptr_raises():
    """data_ptr is not supported — gpu_ops uses gpu_buffer.data_ptr directly."""
    mo = _make_gds_mem_obj()
    with pytest.raises(NotImplementedError, match="data_ptr"):
        _ = mo.data_ptr


def test_gds_memory_obj_parent_returns_scratch_allocator():
    """``parent()`` returns the allocator so gpu_ops can isinstance-dispatch."""
    mo = _make_gds_mem_obj()
    assert isinstance(mo.parent(), GdsScratchAllocator)


def test_gds_memory_obj_ref_count_lifecycle():
    """Ref-counting works the same as ``TensorMemoryObj``."""
    mo = _make_gds_mem_obj()
    assert mo.get_ref_count() == 0
    mo.ref_count_up()
    mo.ref_count_up()
    assert mo.get_ref_count() == 2
    mo.ref_count_down()
    assert mo.get_ref_count() == 1
    mo.ref_count_down()
    assert mo.get_ref_count() == 0


def test_gds_memory_obj_pin_unpin_lifecycle():
    """Pin/unpin track pin_count without breaking invariants."""
    mo = _make_gds_mem_obj()
    assert not mo.is_pinned
    assert mo.can_evict
    mo.pin()
    assert mo.is_pinned
    assert not mo.can_evict
    mo.unpin()
    assert not mo.is_pinned


# --- CuFileHandleCache -----------------------------------------------


def test_handle_cache_acquire_without_gds_module_raises():
    """Acquire raises clearly when no gds_module is configured."""
    cache = CuFileHandleCache(max_handles=4, gds_module=None)
    with pytest.raises(RuntimeError, match="cuFile not configured"):
        cache.acquire("/tmp/fake", "r")


def _mock_gds_module_with_counter() -> tuple[mock.Mock, dict]:
    """Build a mock gds_module whose ``CuFile`` returns trackable mocks."""
    counter = {"opens": 0, "closes": 0}

    def _make_handle(path, mode, use_direct_io=False):
        counter["opens"] += 1

        def _close():
            counter["closes"] += 1

        handle = mock.Mock(spec=["close", "read", "write"])
        handle.close = mock.Mock(side_effect=_close)
        return handle

    gds_module = mock.Mock()
    gds_module.CuFile = mock.Mock(side_effect=_make_handle)
    return gds_module, counter


def test_handle_cache_reuses_handles_for_same_key():
    """Two acquires for the same (path, mode) share one handle."""
    gds_module, counter = _mock_gds_module_with_counter()
    cache = CuFileHandleCache(max_handles=4, gds_module=gds_module)

    h1 = cache.acquire("/tmp/a", "r")
    h2 = cache.acquire("/tmp/a", "r")
    assert h1 is h2
    assert counter["opens"] == 1
    cache.release("/tmp/a", "r")
    cache.release("/tmp/a", "r")


def test_handle_cache_evicts_lru_idle_entry_when_full():
    """The oldest idle entry is dropped when capacity is exceeded."""
    gds_module, counter = _mock_gds_module_with_counter()
    cache = CuFileHandleCache(max_handles=2, gds_module=gds_module)

    cache.acquire("/tmp/a", "r")
    cache.release("/tmp/a", "r")
    cache.acquire("/tmp/b", "r")
    cache.release("/tmp/b", "r")
    # Capacity now at 2; inserting a third should evict /tmp/a (oldest idle).
    cache.acquire("/tmp/c", "r")
    cache.release("/tmp/c", "r")

    assert counter["opens"] == 3
    assert counter["closes"] == 1  # /tmp/a was closed


def test_handle_cache_close_drops_all():
    """``close()`` closes every cached handle."""
    gds_module, counter = _mock_gds_module_with_counter()
    cache = CuFileHandleCache(max_handles=4, gds_module=gds_module)
    cache.acquire("/tmp/a", "r")
    cache.release("/tmp/a", "r")
    cache.acquire("/tmp/b", "r")
    cache.release("/tmp/b", "r")
    cache.close()
    assert counter["closes"] == 2


# --- GdsScratchAllocator alignment -----------------------------------


def test_scratch_allocator_rejects_misaligned_buffer():
    """A buffer whose byte size is not a 4 KiB multiple is rejected."""
    backend_mock = mock.Mock()
    backend_mock.use_gds = False
    backend_mock.gds_module = None
    alloc = GdsScratchAllocator(backend=backend_mock)

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    buf = torch.empty(4097, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="4 KiB"):
        alloc.register_gpu_buffer(buf)


def test_scratch_allocator_accepts_aligned_buffer_under_posix():
    """An aligned buffer is accepted in POSIX-fallback mode (no cuFile call)."""
    backend_mock = mock.Mock()
    backend_mock.use_gds = False
    backend_mock.gds_module = None
    alloc = GdsScratchAllocator(backend=backend_mock)

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    buf = torch.empty(8192, dtype=torch.uint8, device="cuda")
    alloc.register_gpu_buffer(buf)  # no raise
    alloc.deregister_gpu_buffer()


# --- GdsL1Backend scan + lookup --------------------------------------


def _write_pretend_disk_chunk(
    backend: GdsL1Backend,
    key: ObjectKey,
    shape: torch.Size,
    dtype: torch.dtype,
    fmt: MemoryFormat,
) -> str:
    """Write a fake data file + metadata sidecar for ``key``.

    Used to seed the backend's startup scan in tests that don't
    actually issue a write through the cuFile/POSIX path.
    """
    nbytes = shape.numel() * dtype.itemsize
    metadata_bytes = pack_metadata(
        nbytes=nbytes,
        shape=shape,
        dtype=dtype,
        fmt=fmt,
        lmcache_version="1",
    )
    data_path, _, _, _ = key_to_disk_path(
        key, base_path=backend.gds_path, data_suffix=backend.data_suffix
    )
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    # Data file: metadata header followed by zeroed payload.
    with open(data_path, "wb") as f:
        f.write(metadata_bytes)
        f.write(b"\x00" * nbytes)
    # Sidecar metadata file — what the scan reads.
    with open(data_path + ".metadata", "wb") as f:
        f.write(metadata_bytes)
    return data_path


def test_backend_scan_picks_up_existing_keys(gds_root, loop):
    """Pre-existing on-disk chunks appear in the hot index after scan."""
    config = _make_config(gds_root)
    backend = GdsL1Backend(config=config, loop=loop, dst_device="cuda:0")
    try:
        # Wait for the initial (empty) scan to complete.
        backend.wait_for_scan(timeout=5.0)
        assert backend.get_hot_cache_size() == 0

        # Drop a fake chunk on disk, then construct a fresh backend so
        # its scan sees it.
        seed_key = _object_key(seed=42)
        _write_pretend_disk_chunk(
            backend,
            seed_key,
            shape=torch.Size([2, 64]),
            dtype=torch.float16,
            fmt=MemoryFormat.KV_2LTD,
        )
    finally:
        backend.close()

    backend2 = GdsL1Backend(config=config, loop=loop, dst_device="cuda:0")
    try:
        backend2.wait_for_scan(timeout=5.0)
        assert backend2.get_hot_cache_size() == 1
        assert backend2.lookup([seed_key]) == [True]
        assert backend2.lookup([_object_key(seed=999)]) == [False]
    finally:
        backend2.close()


def test_backend_create_memory_obj_from_index_returns_disk_anchored(gds_root, loop):
    """A scanned entry resolves to a ``GdsMemoryObj`` pointing at its file."""
    config = _make_config(gds_root)
    seed_key = _object_key(seed=7)
    backend = GdsL1Backend(config=config, loop=loop, dst_device="cuda:0")
    try:
        backend.wait_for_scan(timeout=5.0)
        data_path = _write_pretend_disk_chunk(
            backend,
            seed_key,
            shape=torch.Size([2, 128]),
            dtype=torch.float16,
            fmt=MemoryFormat.KV_2LTD,
        )
    finally:
        backend.close()

    backend2 = GdsL1Backend(config=config, loop=loop, dst_device="cuda:0")
    try:
        backend2.wait_for_scan(timeout=5.0)
        gds_obj = backend2.create_memory_obj_from_index(seed_key)
        assert gds_obj is not None
        assert gds_obj.disk_path == data_path
        assert gds_obj.file_offset == _METADATA_MAX_SIZE
        assert gds_obj.tensor is None
        assert isinstance(gds_obj.parent(), GdsScratchAllocator)
    finally:
        backend2.close()


def test_backend_create_memory_obj_for_new_write_is_disk_anchored(gds_root, loop):
    """``create_memory_obj`` mints a fresh disk-anchored entry for writes."""
    config = _make_config(gds_root)
    backend = GdsL1Backend(config=config, loop=loop, dst_device="cuda:0")
    try:
        backend.wait_for_scan(timeout=5.0)
        layout = MemoryLayoutDesc(
            shapes=[torch.Size([2, 64])],
            dtypes=[torch.float16],
        )
        new_key = _object_key(seed=123)
        gds_obj = backend.create_memory_obj(new_key, layout, fmt=MemoryFormat.KV_2LTD)
        assert gds_obj.disk_path.startswith(gds_root) or (
            gds_obj.disk_path.startswith(backend.gds_path)
        )
        assert gds_obj.file_offset == _METADATA_MAX_SIZE
        assert gds_obj.get_size() == 2 * 64 * 2
    finally:
        backend.close()


def test_backend_lookup_returns_false_for_unknown(gds_root, loop):
    """``lookup`` returns ``False`` for keys not in the hot index."""
    config = _make_config(gds_root)
    backend = GdsL1Backend(config=config, loop=loop, dst_device="cuda:0")
    try:
        backend.wait_for_scan(timeout=5.0)
        result = backend.lookup([_object_key(seed=1), _object_key(seed=2)])
        assert result == [False, False]
    finally:
        backend.close()
