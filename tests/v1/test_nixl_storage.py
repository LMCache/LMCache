# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import asyncio
import os
import sys
import threading

# Third Party
import pytest
import torch

pytest.importorskip("nixl", reason="nixl package is required for nixl tests")

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import PagedTensorMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend import CreateStorageBackends
from lmcache.v1.storage_backend.nixl_storage_backend import NixlStorageBackend


def create_key(chunk_hash: str):
    return CacheEngineKey(
        model_name="meta-llama/Llama-3.1-70B-Instruct",
        world_size=8,
        worker_id=0,
        chunk_hash=int(chunk_hash, base=16),
        dtype=torch.bfloat16,
    )


def run(config: LMCacheEngineConfig, shape, dtype):
    BACKEND_NAME = "NixlStorageBackend"
    keys = []
    objs = []
    keys.append(
        create_key("e3229141e680fb413d2c5d3ebb416c4ad300d381e309fc9e417757b91406c157")
    )
    keys.append(
        create_key("e3229141e680fb413d2c5d3ebb416c4ad300d381e309fc9e417757b91406d268")
    )
    keys.append(
        create_key("e3229141e680fb413d2c5d3ebb416c4ad300d381e309fc9e417757b91406e379")
    )
    bad_key = create_key("deadbeefdeadbeef")

    thread_loop = None
    thread = None
    try:
        thread_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=thread_loop.run_forever)
        thread.start()

        metadata = LMCacheMetadata(
            model_name="Llama-3.1-70B-Instruct",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=dtype,
            kv_shape=shape,
        )

        backends = CreateStorageBackends(
            config,
            metadata,
            thread_loop,
            dst_device=config.nixl_buffer_device,  # Pass the device directly
        )
        assert len(backends) == 2  # NixlStorageBackend + LocalCPUBackend
        assert BACKEND_NAME in backends

        nixl_backend = backends[BACKEND_NAME]
        assert isinstance(nixl_backend, NixlStorageBackend)
        assert isinstance(nixl_backend.memory_allocator, PagedTensorMemoryAllocator)
        assert nixl_backend is not None
        assert nixl_backend.memory_allocator is not None

        for key in keys:
            assert not nixl_backend.contains(key, False)
            assert not nixl_backend.exists_in_put_tasks(key)

            obj = nixl_backend.memory_allocator.allocate(shape=shape, dtype=dtype)
            assert obj is not None
            assert obj.tensor is not None
            objs.append(obj)

        # small tensor changes for data validation
        objs[0].tensor[100, 200] = 1e-3
        objs[0].tensor[200, 100] = 1e-4

        objs[1].tensor[300, 400] = 1e-2
        objs[1].tensor[400, 300] = 1e-5

        objs[2].tensor[300, 400] = 3e-2
        objs[2].tensor[400, 300] = 4e-5

        # Insert first 2 keys
        first_keys = keys[0:2]
        first_objs = objs[0:2]
        nixl_backend.batched_submit_put_task(first_keys, first_objs)

        for key, obj in zip(first_keys, first_objs, strict=False):
            returned_memory_obj = nixl_backend.get_blocking(key)
            assert returned_memory_obj is not None
            assert returned_memory_obj.get_size() == obj.get_size()
            assert returned_memory_obj.get_shape() == obj.get_shape()
            assert returned_memory_obj.get_dtype() == obj.get_dtype()
            assert returned_memory_obj.metadata.address != obj.metadata.address
            assert torch.equal(returned_memory_obj.tensor, obj.tensor)

        obj_list = asyncio.run(
            nixl_backend.batched_get_non_blocking(lookup_id="test", keys=first_keys)
        )

        for i, obj in enumerate(first_objs):
            returned_memory_obj = obj_list[i]
            assert returned_memory_obj is not None
            assert returned_memory_obj.get_size() == obj.get_size()
            assert returned_memory_obj.get_shape() == obj.get_shape()
            assert returned_memory_obj.get_dtype() == obj.get_dtype()
            assert returned_memory_obj.metadata.address != obj.metadata.address
            assert torch.equal(returned_memory_obj.tensor, obj.tensor)

        def test_eviction(new_idx, old_idx):
            nixl_backend.batched_submit_put_task([keys[new_idx]], [objs[new_idx]])

            obj = nixl_backend.get_blocking(keys[new_idx])
            assert obj is not None
            assert obj.tensor is not None
            assert torch.equal(obj.tensor, objs[new_idx].tensor)

            obj = nixl_backend.get_blocking(keys[old_idx])
            assert obj is None

        ######## Test bad key lookup #########
        obj = nixl_backend.get_blocking(bad_key)
        assert obj is None

        ######## Test eviction #########
        obj = nixl_backend.get_blocking(keys[0])
        assert obj is not None

        # At this point, key 0 & key 1 are cached. Key 1 is LRU key.
        # Submitting key 2 should evict key 1.

        test_eviction(new_idx=2, old_idx=1)

        ######## Test pin #########
        val = nixl_backend.pin(keys[2])
        assert val is True

        obj = nixl_backend.get_blocking(keys[0])
        assert obj is not None

        # At this point, key 0 & key 2 are cached.
        # Key 2 is LRU key, but is pinned.
        # Submitting key 1 should evict key 0.

        test_eviction(new_idx=1, old_idx=0)

        ######## Test unpin #########
        val = nixl_backend.unpin(keys[2])
        assert val is True

        obj = nixl_backend.get_blocking(keys[1])
        assert obj is not None

        # At this point, key 1 & key 2 are cached.
        # Key 2 is LRU key, and is now unpinned.
        # Submitting key 0 should evict key 2.

        test_eviction(new_idx=0, old_idx=2)

        for backend in backends.values():
            backend.close()

    except Exception:
        raise
    finally:
        if thread_loop and thread_loop.is_running():
            thread_loop.call_soon_threadsafe(thread_loop.stop)
        if thread and thread.is_alive():
            thread.join()


@pytest.mark.no_shared_allocator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_nixl_gds_mt_cuda_backend():
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = [2048, 2048]

    config.nixl_buffer_device = "cuda:0"  # Use explicit device
    config.extra_config["nixl_backend"] = "GDS_MT"
    config.extra_config["enable_cuda"] = True

    run(config, shape, dtype)


@pytest.mark.no_shared_allocator
def test_nixl_gds_mt_cpu_backend():
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = [2048, 2048]

    config.nixl_buffer_device = "cpu"
    config.extra_config["nixl_backend"] = "GDS_MT"
    config.extra_config["enable_cuda"] = False

    run(config, shape, dtype)


@pytest.mark.no_shared_allocator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_nixl_gds_cuda_backend():
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = [2048, 2048]

    config.nixl_buffer_device = "cuda:0"  # Use explicit device
    config.extra_config["nixl_backend"] = "GDS"
    config.extra_config["enable_cuda"] = True

    run(config, shape, dtype)


@pytest.mark.no_shared_allocator
def test_nixl_gds_cpu_backend():
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = [2048, 2048]

    config.nixl_buffer_device = "cpu"
    config.extra_config["nixl_backend"] = "GDS"
    config.extra_config["enable_cuda"] = False

    run(config, shape, dtype)


@pytest.mark.no_shared_allocator
def test_nixl_posix_backend():
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = [2048, 2048]

    config.nixl_buffer_device = "cpu"
    config.extra_config["nixl_backend"] = "POSIX"
    config.extra_config["enable_cuda"] = False

    run(config, shape, dtype)


def _build_dynamic_file_backend(config, shape, dtype):
    """
    Build a NixlStorageBackend in dynamic-FILE mode and the surrounding
    event-loop thread. Returns (backend, backends, thread_loop, thread, keys,
    objs) so the caller can drive the test and tear everything down via
    ``_teardown_dynamic_file_backend``.
    """
    BACKEND_NAME = "NixlStorageBackend"

    keys = [
        create_key("e3229141e680fb413d2c5d3ebb416c4ad300d381e309fc9e417757b91406c157"),
        create_key("e3229141e680fb413d2c5d3ebb416c4ad300d381e309fc9e417757b91406d268"),
    ]

    thread_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=thread_loop.run_forever)
    thread.start()

    metadata = LMCacheMetadata(
        model_name="Llama-3.1-70B-Instruct",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=dtype,
        kv_shape=shape,
    )

    backends = CreateStorageBackends(
        config,
        metadata,
        thread_loop,
        dst_device=config.nixl_buffer_device,
    )
    nixl_backend = backends[BACKEND_NAME]
    assert isinstance(nixl_backend, NixlStorageBackend)

    objs = []
    for _ in keys:
        obj = nixl_backend.memory_allocator.allocate(shape=shape, dtype=dtype)
        assert obj is not None
        assert obj.tensor is not None
        objs.append(obj)

    objs[0].tensor[100, 200] = 1e-3
    objs[1].tensor[300, 400] = 1e-2

    return nixl_backend, backends, thread_loop, thread, keys, objs


def _teardown_dynamic_file_backend(backends, thread_loop, thread):
    for backend in backends.values():
        backend.close()
    if thread_loop and thread_loop.is_running():
        thread_loop.call_soon_threadsafe(thread_loop.stop)
    if thread and thread.is_alive():
        thread.join()


def run_dynamic_file(config, shape, dtype, tmp_path):
    """
    Exercise the dynamic-FILE backend's new code paths: contains/key_exists,
    put/get round-trip, and remove for both present and missing files.
    """
    nixl_backend, backends, thread_loop, thread, keys, objs = (
        _build_dynamic_file_backend(config, shape, dtype)
    )

    try:
        for key in keys:
            assert not nixl_backend.contains(key, False)
            assert not nixl_backend.exists_in_put_tasks(key)

        nixl_backend.batched_submit_put_task(keys, objs)

        for key in keys:
            assert nixl_backend.contains(key, False)

        files_after_put = set(os.listdir(str(tmp_path)))
        expected_files = {nixl_backend._format_object_key(k) for k in keys}
        assert expected_files.issubset(files_after_put), (
            f"missing files in {tmp_path}: {expected_files - files_after_put}"
        )

        for key, obj in zip(keys, objs, strict=False):
            returned = nixl_backend.get_blocking(key)
            assert returned is not None
            assert returned.get_size() == obj.get_size()
            assert returned.get_shape() == obj.get_shape()
            assert returned.get_dtype() == obj.get_dtype()
            assert torch.equal(returned.tensor, obj.tensor)

        first_remove = nixl_backend.remove(keys[0])
        assert first_remove is True
        assert not os.path.exists(
            os.path.join(str(tmp_path), nixl_backend._format_object_key(keys[0]))
        )

        # Removing an already-gone file must return False
        # instead of raising FileNotFoundError.
        second_remove = nixl_backend.remove(keys[0])
        assert second_remove is False
    finally:
        _teardown_dynamic_file_backend(backends, thread_loop, thread)


@pytest.mark.no_shared_allocator
def test_nixl_posix_dynamic_file_backend(tmp_path):
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = [2048, 2048]

    config.nixl_buffer_device = "cpu"
    config.extra_config["nixl_backend"] = "POSIX"
    config.extra_config["nixl_pool_size"] = 0  # dynamic mode
    config.extra_config["nixl_path"] = str(tmp_path)
    config.extra_config["enable_cuda"] = False

    run_dynamic_file(config, shape, dtype, tmp_path)


def _count_open_fds() -> int:
    return len(os.listdir("/proc/self/fd"))


@pytest.mark.no_shared_allocator
@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Requires /proc/self/fd to count open FDs",
)
def test_nixl_dynamic_file_fd_leak_on_setup_failure(tmp_path, monkeypatch):
    """
    Regression for Gemini G3 / Cursor C2: if any operation between the
    per-key ``os.open`` loop and ``release_storage_handler`` raises, the
    already-opened FDs must be closed instead of leaked. Driven in sync
    mode so the induced exception surfaces via ``future.result()``.
    """
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = [2048, 2048]

    config.nixl_buffer_device = "cpu"
    config.extra_config["nixl_backend"] = "POSIX"
    config.extra_config["nixl_pool_size"] = 0
    config.extra_config["nixl_path"] = str(tmp_path)
    config.extra_config["nixl_async_put"] = False
    config.extra_config["enable_cuda"] = False

    nixl_backend, backends, thread_loop, thread, keys, objs = (
        _build_dynamic_file_backend(config, shape, dtype)
    )

    try:
        baseline = _count_open_fds()

        def boom(*args, **kwargs):
            raise RuntimeError("induced failure")

        monkeypatch.setattr(nixl_backend.agent, "create_batched_storage_handler", boom)

        # Sync mode: batched_submit_put_task calls future.result(), so the
        # induced RuntimeError propagates here.
        with pytest.raises(RuntimeError):
            nixl_backend.batched_submit_put_task(keys, objs)

        with pytest.raises(RuntimeError):
            nixl_backend.get_blocking(keys[0])

        assert _count_open_fds() == baseline, "FDs leaked on transfer-setup failure"
    finally:
        _teardown_dynamic_file_backend(backends, thread_loop, thread)
