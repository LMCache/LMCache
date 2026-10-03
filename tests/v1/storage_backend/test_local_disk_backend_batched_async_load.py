# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import patch
import asyncio
import os
import shutil
import tempfile
import threading

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend

_SHAPE = torch.Size([28, 2, 256, 8, 128])
_DTYPE = torch.bfloat16


def _make_backend(memory_allocator) -> tuple[LocalDiskBackend, LocalCPUBackend, str]:
    temp_dir = tempfile.mkdtemp()
    cfg = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_disk=temp_dir,
        max_local_disk_size=1.0,
        lmcache_instance_id="batched-async-load",
    )
    cpu = LocalCPUBackend(
        LMCacheEngineConfig.from_legacy(chunk_size=256),
        memory_allocator=memory_allocator,
    )
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    backend = LocalDiskBackend(
        config=cfg,
        loop=loop,
        local_cpu_backend=cpu,
        dst_device="cuda:0",
    )
    return backend, cpu, temp_dir


def _key(key_id: int) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test_model",
        world_size=3,
        worker_id=1,
        chunk_hash=hash(key_id),
        dtype=_DTYPE,
    )


def test_batched_async_load_concurrent_evict_does_not_raise(memory_allocator) -> None:
    """A concurrent remove(force=True) racing the unlocked disk read inside
    batched_async_load_bytes_from_disk must not crash with a KeyError; the
    load for that key should be dropped as a miss, matching how
    load_bytes_from_disk / _load_chunk_into_memory treat the same race.

    batched_get_non_blocking pins both the disk metadata and the staging
    MemoryObj before dispatching to this function specifically to prevent
    this window, but remove(force=True) is a deliberate external override
    that does not check pin_count, so the race is still reachable (e.g. a
    second in-flight request for the same key that lands its
    remove_after_retrieve while this prefetch is mid-read). A barrier is
    injected into read_file, a dependency and not the function under fix,
    so the eviction lands between the disk read and the metadata lookup.
    """
    backend, cpu, temp_dir = _make_backend(memory_allocator)
    try:
        key = _key(700)
        path = backend._key_to_path(key)
        nbytes = _DTYPE.itemsize
        for s in _SHAPE:
            nbytes *= s
        with open(path, "wb") as f:
            f.write(os.urandom(nbytes))
        backend.insert_key(
            key,
            size=nbytes,
            shape=_SHAPE,
            dtype=_DTYPE,
            fmt=MemoryFormat.KV_2LTD,
        )
        backend.dict[key].pin()

        mem_obj = cpu.allocate(_SHAPE, _DTYPE, MemoryFormat.KV_2LTD, busy_loop=False)
        assert mem_obj is not None

        entered_read = threading.Event()
        evicted = threading.Event()
        original_read_file = backend.read_file

        def delaying_read_file(k, buffer, p):
            entered_read.set()
            evicted.wait(timeout=5)
            return original_read_file(k, buffer, p)

        holder: dict = {}

        def loader():
            try:
                holder["res"] = backend.batched_async_load_bytes_from_disk(
                    paths=[path], keys=[key], memory_objs=[mem_obj]
                )
            except Exception as exc:
                holder["err"] = exc

        with patch.object(backend, "read_file", side_effect=delaying_read_file):
            t = threading.Thread(target=loader)
            t.start()
            assert entered_read.wait(timeout=5), "loader never reached read_file"
            backend.remove(key, force=True)
            evicted.set()
            t.join(timeout=10)

        assert "err" not in holder, (
            f"batched_async_load_bytes_from_disk raised: {holder['err']!r}"
        )
        assert holder.get("res") == [mem_obj]
        assert key not in backend.dict
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_batched_async_load_hit_recovers_metadata_and_unpins(memory_allocator) -> None:
    """Undisturbed happy path: the key is still present when the disk read
    completes, so cached_positions must be recovered from disk metadata and
    the metadata's pin released, exactly as before this fix's refactor.
    """
    backend, cpu, temp_dir = _make_backend(memory_allocator)
    try:
        key = _key(701)
        path = backend._key_to_path(key)
        nbytes = _DTYPE.itemsize
        for s in _SHAPE:
            nbytes *= s
        with open(path, "wb") as f:
            f.write(os.urandom(nbytes))
        positions = torch.arange(_SHAPE[2])
        backend.insert_key(
            key,
            size=nbytes,
            shape=_SHAPE,
            dtype=_DTYPE,
            fmt=MemoryFormat.KV_2LTD,
            cached_positions=positions,
        )
        backend.dict[key].pin()

        mem_obj = cpu.allocate(_SHAPE, _DTYPE, MemoryFormat.KV_2LTD, busy_loop=False)
        assert mem_obj is not None

        result = backend.batched_async_load_bytes_from_disk(
            paths=[path], keys=[key], memory_objs=[mem_obj]
        )

        assert result == [mem_obj]
        assert torch.equal(mem_obj.metadata.cached_positions, positions)
        assert not backend.dict[key].is_pinned
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
