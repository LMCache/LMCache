# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from concurrent.futures import Future
import asyncio
import ctypes
import os
import tempfile
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    AdHocMemoryAllocator,
    MemoryFormat,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import (
    LocalCPUBackend,
)
from lmcache.v1.storage_backend.plugins.rust_remote_backend import (
    RustRemoteBackend,
)


def _has_ext() -> bool:
    try:
        # Third Party
        import lmcache_rust_remote_backend_io  # noqa: F401

        return True
    except Exception:
        return False


def _find_connector_fs_lib() -> str:
    """Locate the built lmcache_connector_fs shared library."""
    # tests/v1/storage_backend/test_*.py -> 4 levels up
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    candidates = []
    for profile in ("release", "debug"):
        for ext in (".so", ".dylib"):
            candidates.append(
                os.path.join(
                    project_root,
                    "rust",
                    "connector_fs",
                    "target",
                    profile,
                    "liblmcache_connector_fs" + ext,
                )
            )
    for path in candidates:
        if os.path.isfile(path):
            return path
    return ""


def _has_connector_fs() -> bool:
    lib_path = _find_connector_fs_lib()
    if not lib_path:
        return False
    try:
        ctypes.cdll.LoadLibrary(lib_path)
        return True
    except OSError:
        return False


@pytest.fixture
def loop_in_thread():
    loop = asyncio.new_event_loop()
    t = threading.Thread(
        target=loop.run_forever,
        name="test-loop",
        daemon=True,
    )
    t.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=5)
        loop.close()


def _make_config_and_metadata(fs_path: str):
    connector_lib = _find_connector_fs_lib()
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=0.1,
        lmcache_instance_id="test_rust_remote_backend",
    )
    config.storage_plugins = []
    config.extra_config = {
        "rust_remote.connector_lib": connector_lib,
        "rust_remote.connector.base_path": fs_path,
    }
    metadata = LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 256, 8, 128),
    )
    return config, metadata


_skip_no_ext = pytest.mark.skipif(
    not _has_ext(),
    reason=("lmcache_rust_remote_backend_io extension not installed"),
)
_skip_no_fs = pytest.mark.skipif(
    not _has_connector_fs(),
    reason="lmcache_connector_fs .so not built",
)


@_skip_no_ext
@_skip_no_fs
def test_rust_remote_backend_put_get_roundtrip(memory_allocator, loop_in_thread):
    """Test basic put/get roundtrip."""
    with tempfile.TemporaryDirectory() as td:
        config, metadata = _make_config_and_metadata(td)

        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = RustRemoteBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            key = CacheEngineKey("test_model", 1, 0, 12345, torch.bfloat16)
            allocator = AdHocMemoryAllocator(device="cpu")
            obj = allocator.allocate(
                [torch.Size([2, 16, 8, 128])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj is not None
            assert obj.tensor is not None
            obj.tensor.fill_(7)
            expected = bytes(obj.byte_array)

            futs = backend.batched_submit_put_task([key], [obj])
            assert futs is not None
            assert isinstance(futs[0], Future)
            futs[0].result(timeout=10)

            assert backend.contains(key)

            out = backend.get_blocking(key)
            assert out is not None
            assert bytes(out.byte_array) == expected
        finally:
            backend.close()


@_skip_no_ext
@_skip_no_fs
def test_rust_remote_backend_contains_and_remove(memory_allocator, loop_in_thread):
    """Test contains and remove."""
    with tempfile.TemporaryDirectory() as td:
        config, metadata = _make_config_and_metadata(td)

        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = RustRemoteBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            key = CacheEngineKey("test_model", 1, 0, 99999, torch.bfloat16)

            assert backend.contains(key) is False
            assert backend.get_blocking(key) is None

            allocator = AdHocMemoryAllocator(device="cpu")
            obj = allocator.allocate(
                [torch.Size([2, 16, 8, 128])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj is not None
            assert obj.tensor is not None
            obj.tensor.fill_(42)

            futs = backend.batched_submit_put_task([key], [obj])
            assert futs is not None
            futs[0].result(timeout=10)

            assert backend.contains(key) is True

            removed = backend.remove(key)
            assert removed is True

            assert backend.contains(key) is False
            assert backend.get_blocking(key) is None
        finally:
            backend.close()


@_skip_no_ext
@_skip_no_fs
def test_rust_remote_backend_multiple_keys(memory_allocator, loop_in_thread):
    """Test writing and reading multiple keys."""
    with tempfile.TemporaryDirectory() as td:
        config, metadata = _make_config_and_metadata(td)

        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = RustRemoteBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            allocator = AdHocMemoryAllocator(device="cpu")
            num_keys = 5
            keys = [
                CacheEngineKey(
                    "test_model",
                    1,
                    0,
                    i,
                    torch.bfloat16,
                )
                for i in range(num_keys)
            ]
            objs = []
            expected_data = []
            for i in range(num_keys):
                obj = allocator.allocate(
                    [torch.Size([2, 16, 8, 128])],
                    [torch.bfloat16],
                    fmt=MemoryFormat.KV_T2D,
                )
                assert obj is not None
                assert obj.tensor is not None
                obj.tensor.fill_(i + 1)
                expected_data.append(bytes(obj.byte_array))
                objs.append(obj)

            futs = backend.batched_submit_put_task(keys, objs)
            assert futs is not None
            for f in futs:
                f.result(timeout=10)

            for i, key in enumerate(keys):
                assert backend.contains(key)
                out = backend.get_blocking(key)
                assert out is not None
                assert bytes(out.byte_array) == expected_data[i]
        finally:
            backend.close()
