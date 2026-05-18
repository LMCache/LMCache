# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import asyncio
import contextlib
import functools
import os
import shutil
import sys
import tempfile
import threading
import uuid

# Third Party
import pytest
import torch

pytest.importorskip("nixl", reason="nixl package is required for nixl tests")

# Third Party
from nixl._api import nixl_agent as NixlAgent
from nixl._api import nixl_agent_config as NixlAgentConfig

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import PagedTensorMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend import CreateStorageBackends
from lmcache.v1.storage_backend.nixl_storage_backend import (
    NixlStorageBackend,
    NixlStorageConfig,
)
from lmcache.v1.transfer_channel.transfer_utils import get_correct_device

# cuFile-based backends (GDS, GDS_MT) need a GDS-capable filesystem
_TEST_TMPDIR = os.environ.get("LMCACHE_TEST_TMPDIR") or None


@functools.lru_cache(maxsize=None)
def _can_register_file_with_nixl_backend(backend: str) -> bool:
    """Probe ``cuFileHandleRegister`` via NIXL on the test scratch dir."""

    probe_dir = tempfile.mkdtemp(prefix="nixl_gds_probe_", dir=_TEST_TMPDIR)
    probe_path = os.path.join(probe_dir, "probe.bin")
    fd = -1
    try:
        agent = NixlAgent(
            f"NixlGdsProbe_{uuid.uuid4().hex}",
            NixlAgentConfig(backends=[]),
        )
        agent.create_backend(backend, {})
        fd = os.open(probe_path, os.O_CREAT | os.O_RDWR, 0o600)
        os.write(fd, b"\x00" * 4096)
        agent.register_memory([(0, 4096, fd, "")], mem_type="FILE")
        return True
    except Exception:
        return False
    finally:
        if fd >= 0:
            with contextlib.suppress(OSError):
                os.close(fd)
        shutil.rmtree(probe_dir, ignore_errors=True)


_GDS_SKIP_REASON = (
    "NIXL {backend} cannot register file handles in this environment; "
    "set LMCACHE_TEST_TMPDIR to a GDS-capable mount (ext4/xfs) to enable."
)


@pytest.fixture
def nixl_tmp_path():
    """Per-test scratch dir, honoring ``LMCACHE_TEST_TMPDIR``."""
    path = tempfile.mkdtemp(prefix="nixl_test_", dir=_TEST_TMPDIR)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


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
            dst_device=get_correct_device(
                config.nixl_buffer_device, metadata.worker_id
            ),
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

            obj = nixl_backend.memory_allocator.allocate(torch.Size(shape), dtype)
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

            returned_tensor = returned_memory_obj.tensor
            obj_tensor = obj.tensor
            assert returned_tensor is not None
            assert obj_tensor is not None
            assert torch.equal(returned_tensor, obj_tensor)

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

            returned_tensor = returned_memory_obj.tensor
            obj_tensor = obj.tensor
            assert returned_tensor is not None
            assert obj_tensor is not None
            assert torch.equal(returned_tensor, obj_tensor)

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
@pytest.mark.skipif(
    not _can_register_file_with_nixl_backend("GDS_MT"),
    reason=_GDS_SKIP_REASON.format(backend="GDS_MT"),
)
def test_nixl_gds_mt_cuda_backend(nixl_tmp_path):
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = torch.Size([2048, 2048])

    config.nixl_buffer_device = "cuda"
    config.extra_config["nixl_backend"] = "GDS_MT"
    config.extra_config["enable_cuda"] = True
    config.extra_config["nixl_path"] = nixl_tmp_path

    run(config, shape, dtype)


@pytest.mark.no_shared_allocator
@pytest.mark.skipif(
    not _can_register_file_with_nixl_backend("GDS_MT"),
    reason=_GDS_SKIP_REASON.format(backend="GDS_MT"),
)
def test_nixl_gds_mt_cpu_backend(nixl_tmp_path):
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = torch.Size([2048, 2048])

    config.nixl_buffer_device = "cpu"
    config.extra_config["nixl_backend"] = "GDS_MT"
    config.extra_config["enable_cuda"] = False
    config.extra_config["nixl_path"] = nixl_tmp_path

    run(config, shape, dtype)


@pytest.mark.no_shared_allocator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.skipif(
    not _can_register_file_with_nixl_backend("GDS"),
    reason=_GDS_SKIP_REASON.format(backend="GDS"),
)
def test_nixl_gds_cuda_backend(nixl_tmp_path):
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = torch.Size([2048, 2048])

    config.nixl_buffer_device = "cuda"
    config.extra_config["nixl_backend"] = "GDS"
    config.extra_config["enable_cuda"] = True
    config.extra_config["nixl_path"] = nixl_tmp_path

    run(config, shape, dtype)


@pytest.mark.no_shared_allocator
@pytest.mark.skipif(
    not _can_register_file_with_nixl_backend("GDS"),
    reason=_GDS_SKIP_REASON.format(backend="GDS"),
)
def test_nixl_gds_cpu_backend(nixl_tmp_path):
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = torch.Size([2048, 2048])

    config.nixl_buffer_device = "cpu"
    config.extra_config["nixl_backend"] = "GDS"
    config.extra_config["enable_cuda"] = False
    config.extra_config["nixl_path"] = nixl_tmp_path

    run(config, shape, dtype)


@pytest.mark.no_shared_allocator
def test_nixl_endpoint_list_empty_raises():
    """nixl_endpoint_list=[] should raise ValueError before any nixl ops."""
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")
    config.extra_config["nixl_endpoint_list"] = []

    metadata = LMCacheMetadata(
        model_name="Llama-3.1-70B-Instruct",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=[2048, 2048],
    )

    with pytest.raises(ValueError, match="nixl_endpoint_list is set but empty"):
        NixlStorageConfig.from_cache_engine_config(config, metadata)


@pytest.mark.no_shared_allocator
def test_nixl_endpoint_list_malformed_url_raises():
    """A non-http(s) entry in nixl_endpoint_list should raise ValueError."""
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")
    config.extra_config["nixl_endpoint_list"] = ["htps://typo.example.com"]
    config.extra_config["nixl_backend"] = "OBJ"

    metadata = LMCacheMetadata(
        model_name="Llama-3.1-70B-Instruct",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=[2048, 2048],
    )

    with pytest.raises(ValueError, match="is not a valid URL"):
        NixlStorageConfig.from_cache_engine_config(config, metadata)


@pytest.mark.no_shared_allocator
def test_nixl_posix_backend(nixl_tmp_path):
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = torch.Size([2048, 2048])

    config.nixl_buffer_device = "cpu"
    config.extra_config["nixl_backend"] = "POSIX"
    config.extra_config["enable_cuda"] = False
    config.extra_config["nixl_path"] = nixl_tmp_path

    run(config, shape, dtype)


_DYNAMIC_KV_SHAPE = (4, 2, 256, 8, 128)


def _build_dynamic_file_backend(config, dtype):
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
        kv_shape=_DYNAMIC_KV_SHAPE,
    )

    backends = CreateStorageBackends(
        config,
        metadata,
        thread_loop,
        dst_device=config.nixl_buffer_device,
    )
    nixl_backend = backends[BACKEND_NAME]
    assert isinstance(nixl_backend, NixlStorageBackend)

    # In dynamic mode the backend internally allocates with meta_shape
    # (derived from kv_shape via init_chunk_meta), so allocate the test
    # objects with the same shape so put/get round-trip shapes match.
    obj_shape = nixl_backend.meta_shape
    obj_dtype = nixl_backend.meta_dtype
    assert obj_shape is not None
    assert obj_dtype is not None

    obj_fmt = nixl_backend.meta_fmt
    assert obj_fmt is not None

    objs = []
    for _ in keys:
        obj = nixl_backend.memory_allocator.allocate(obj_shape, obj_dtype, obj_fmt)
        assert obj is not None
        assert obj.tensor is not None
        objs.append(obj)

    objs[0].tensor.zero_()
    objs[1].tensor.zero_()
    objs[0].tensor[0, 0, 100, 200] = 1
    objs[1].tensor[1, 0, 50, 300] = 1

    return nixl_backend, backends, thread_loop, thread, keys, objs


def _teardown_dynamic_file_backend(backends, thread_loop, thread, objs=()):
    for obj in objs:
        if obj is None:
            continue
        if obj.is_valid() and obj.get_ref_count() > 0:
            obj.ref_count_down()
    for backend in backends.values():
        backend.close()
    if thread_loop and thread_loop.is_running():
        thread_loop.call_soon_threadsafe(thread_loop.stop)
    if thread and thread.is_alive():
        thread.join()


def run_dynamic_file(config, dtype, tmp_path):
    """
    Exercise the dynamic-FILE backend's new code paths: contains/key_exists,
    put/get round-trip, and remove for both present and missing files.
    """
    nixl_backend, backends, thread_loop, thread, keys, objs = (
        _build_dynamic_file_backend(config, dtype)
    )

    retained_objs = list(objs)

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
            retained_objs.append(returned)
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
        _teardown_dynamic_file_backend(backends, thread_loop, thread, retained_objs)


@pytest.mark.no_shared_allocator
def test_nixl_posix_dynamic_file_backend(tmp_path):
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16

    config.nixl_buffer_device = "cpu"
    config.extra_config["nixl_backend"] = "POSIX"
    config.extra_config["nixl_pool_size"] = 0  # dynamic mode
    config.extra_config["nixl_path"] = str(tmp_path)
    config.extra_config["enable_cuda"] = False

    run_dynamic_file(config, dtype, tmp_path)


def _count_open_fds() -> int:
    return len(os.listdir("/proc/self/fd"))


@pytest.mark.no_shared_allocator
@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Requires /proc/self/fd to count open FDs",
)
def test_nixl_dynamic_file_fd_leak_on_setup_failure(tmp_path, monkeypatch):
    """
    If any operation between the per-key ``os.open`` loop and
    ``release_storage_handler`` raises, the already-opened FDs must be
    closed and the just-created files unlinked instead of leaked.
    """
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16

    config.nixl_buffer_device = "cpu"
    config.extra_config["nixl_backend"] = "POSIX"
    config.extra_config["nixl_pool_size"] = 0
    config.extra_config["nixl_path"] = str(tmp_path)
    config.extra_config["nixl_async_put"] = False
    config.extra_config["enable_cuda"] = False

    nixl_backend, backends, thread_loop, thread, keys, objs = (
        _build_dynamic_file_backend(config, dtype)
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

        assert _count_open_fds() == baseline, "FDs leaked on transfer-setup failure"

        # The put path opens the final key files with O_CREAT before
        # registering the storage handler, so a failure here must clean
        # up those just-created files.
        for key in keys:
            assert not os.path.exists(
                os.path.join(str(tmp_path), nixl_backend._format_object_key(key))
            ), "final key file leaked on transfer-setup failure"
    finally:
        _teardown_dynamic_file_backend(backends, thread_loop, thread, objs)


@pytest.mark.no_shared_allocator
def test_nixl_dynamic_file_no_leak_on_transfer_failure(tmp_path, monkeypatch):
    """
    When the NIXL transfer itself fails after the final key
    files have been opened with ``O_CREAT``, the backend must remove
    those empty / partially-written files.
    """
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16

    config.nixl_buffer_device = "cpu"
    config.extra_config["nixl_backend"] = "POSIX"
    config.extra_config["nixl_pool_size"] = 0
    config.extra_config["nixl_path"] = str(tmp_path)
    config.extra_config["nixl_async_put"] = False
    config.extra_config["enable_cuda"] = False

    nixl_backend, backends, thread_loop, thread, keys, objs = (
        _build_dynamic_file_backend(config, dtype)
    )

    try:

        def boom(*args, **kwargs):
            raise RuntimeError("induced post_blocking failure")

        monkeypatch.setattr(nixl_backend.agent, "post_blocking", boom)

        with pytest.raises(RuntimeError):
            nixl_backend.batched_submit_put_task(keys, objs)

        for key in keys:
            final_path = os.path.join(
                str(tmp_path), nixl_backend._format_object_key(key)
            )
            assert not os.path.exists(final_path), (
                f"final key file leaked on transfer failure: {final_path}"
            )
            assert not nixl_backend.contains(key, False), (
                "contains() reports key present after failed write"
            )
    finally:
        _teardown_dynamic_file_backend(backends, thread_loop, thread, objs)
