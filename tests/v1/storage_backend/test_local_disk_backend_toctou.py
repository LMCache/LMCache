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


def _make_backend(memory_allocator) -> tuple[LocalDiskBackend, str]:
    temp_dir = tempfile.mkdtemp()
    cfg = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_disk=temp_dir,
        max_local_disk_size=1.0,
        lmcache_instance_id="toctou",
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
    return backend, temp_dir


def _key(key_id: int) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test_model",
        world_size=3,
        worker_id=1,
        chunk_hash=hash(key_id),
        dtype=_DTYPE,
    )


def test_get_blocking_concurrent_evict_returns_none(memory_allocator) -> None:
    """A concurrent remove(force=True) during the unlocked disk read must
    not crash get_blocking; it should surface as a cache miss (None).

    The disk read in load_bytes_from_disk runs without disk_lock (by
    design). A barrier is injected into read_file -- a dependency, not the
    function under fix -- so the eviction lands in the window between the
    membership check and the self.dict[key] access. On the unfixed code
    that access raises KeyError out of get_blocking; on the fixed code it
    returns None.
    """
    backend, temp_dir = _make_backend(memory_allocator)
    try:
        key = _key(500)
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

        entered_read = threading.Event()
        evicted = threading.Event()
        original_read_file = backend.read_file

        def delaying_read_file(k, buffer, p):
            entered_read.set()
            evicted.wait(timeout=5)
            return original_read_file(k, buffer, p)

        holder: dict = {}

        def reader():
            try:
                holder["res"] = backend.get_blocking(key)
            except Exception as exc:
                holder["err"] = exc

        with patch.object(backend, "read_file", side_effect=delaying_read_file):
            t = threading.Thread(target=reader)
            t.start()
            assert entered_read.wait(timeout=5), "reader never reached read_file"
            backend.remove(key, force=True)
            evicted.set()
            t.join(timeout=10)

        assert "err" not in holder, f"get_blocking raised: {holder['err']!r}"
        assert holder.get("res") is None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
