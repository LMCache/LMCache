# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import os
import shutil
import tempfile
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.config_base import _parse_local_disk
from lmcache.v1.memory_management import (
    AdHocMemoryAllocator,
    MemoryFormat,
    get_size_bytes,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend


class MockLookupServer:
    def __init__(self):
        self.removed_keys = []
        self.inserted_keys = []

    def batched_remove(self, keys):
        self.removed_keys.extend(keys)

    def batched_insert(self, keys):
        self.inserted_keys.extend(keys)


class MockLMCacheWorker:
    def __init__(self):
        self.messages = []

    def put_msg(self, msg):
        self.messages.append(msg)


def create_test_config(
    disk_path: str,
    max_disk_size: float = 1.0,
    local_disk_path_sharding: str = "by_gpu",
):
    """Create a test configuration for LocalDiskBackend."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_disk=disk_path,
        local_disk_path_sharding=local_disk_path_sharding,
        max_local_disk_size=max_disk_size,
        lmcache_instance_id="test_instance",
    )
    return config


def create_test_metadata():
    """Create a test metadata for LMCacheMetadata."""
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(28, 2, 256, 8, 128),
    )


def create_test_key(key_id: int = 0) -> CacheEngineKey:
    """Create a test CacheEngineKey."""
    return CacheEngineKey(
        model_name="test_model",
        world_size=3,
        worker_id=1,
        chunk_hash=hash(key_id),
        dtype=torch.bfloat16,
    )


@pytest.fixture
def temp_disk_path():
    """Create a temporary directory for disk storage tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def async_loop():
    """Create an asyncio event loop for testing."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# ----------------------------------------------------------------------------


@pytest.fixture
def local_cpu_backend(memory_allocator):
    """Create a LocalCPUBackend for testing."""
    config = LMCacheEngineConfig.from_legacy(chunk_size=256)
    return LocalCPUBackend(config, memory_allocator=memory_allocator)


@pytest.fixture
def local_disk_backend(temp_disk_path, async_loop, local_cpu_backend):
    """Create a LocalDiskBackend for testing."""
    config = create_test_config(temp_disk_path)
    return LocalDiskBackend(
        config=config,
        loop=async_loop,
        local_cpu_backend=local_cpu_backend,
        dst_device="cuda:0",
    )


class TestLocalDiskBackend:
    """Test cases for LocalDiskBackend."""

    def test_init(self, temp_disk_path, async_loop, local_cpu_backend):
        """Test LocalDiskBackend initialization."""
        config = create_test_config(temp_disk_path)
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
        )

        assert backend.dst_device == "cuda:0"
        assert backend.local_cpu_backend == local_cpu_backend
        assert backend.path == temp_disk_path
        assert os.path.exists(temp_disk_path)
        assert backend.lmcache_worker is None
        assert backend.instance_id == "test_instance"
        assert backend.usage == 0
        assert len(backend.dict) == 0

        local_cpu_backend.memory_allocator.close()

    def test_init_with_lookup_server_and_worker(
        self, temp_disk_path, async_loop, local_cpu_backend
    ):
        """Test LocalDiskBackend initialization with lookup server and worker."""
        config = create_test_config(temp_disk_path)
        lmcache_worker = MockLMCacheWorker()

        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
            lmcache_worker=lmcache_worker,
        )

        assert backend.lmcache_worker == lmcache_worker

        local_cpu_backend.memory_allocator.close()

    def test_str(self, local_disk_backend):
        """Test string representation."""
        assert str(local_disk_backend) == "LocalDiskBackend"
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_key_to_path(self, local_disk_backend):
        """Test key to path conversion."""
        key = create_test_key(1)
        path = local_disk_backend._key_to_path(key)

        expected_filename = key.to_string().replace("/", "-") + ".pt"
        assert path == os.path.join(local_disk_backend.path, expected_filename)

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_contains_key_not_exists(self, local_disk_backend):
        """Test contains() when key doesn't exist."""
        key = create_test_key(2)
        assert not local_disk_backend.contains(key)
        assert not local_disk_backend.contains(key, pin=True)

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_get_blocking_key_not_exists(self, local_disk_backend):
        """Test get_blocking() when key doesn't exist."""
        key = create_test_key(2)
        result = local_disk_backend.get_blocking(key)

        assert result is None

        local_disk_backend.local_cpu_backend.memory_allocator.close()


class TestMultiPathDiskBackend:
    """Test cases for multi-path (multi-device) LocalDiskBackend."""

    def test_init_multi_path(self, async_loop, local_cpu_backend):
        """Test initialisation with comma-separated paths."""
        dir_a = tempfile.mkdtemp()
        dir_b = tempfile.mkdtemp()
        try:
            combined = f"{dir_a},{dir_b}"
            config = create_test_config(combined)
            backend = LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:0",
            )

            # Path selected by device_id (0 % 2 = 0 -> dir_a)
            assert backend.path == dir_a
            # Both directories should exist
            assert os.path.isdir(dir_a)
            assert os.path.isdir(dir_b)
            # Block size is a plain int for the selected path
            assert isinstance(backend.os_disk_bs, int)
        finally:
            shutil.rmtree(dir_a, ignore_errors=True)
            shutil.rmtree(dir_b, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()

    def test_gpu_affinity_selects_path(self, async_loop, local_cpu_backend):
        """Different cuda devices select different paths via modulo."""
        dir_a = tempfile.mkdtemp()
        dir_b = tempfile.mkdtemp()
        try:
            combined = f"{dir_a},{dir_b}"
            config = create_test_config(combined)

            dirs_by_gpu = {}
            for device in ("cuda:0", "cuda:1"):
                backend = LocalDiskBackend(
                    config=config,
                    loop=async_loop,
                    local_cpu_backend=local_cpu_backend,
                    dst_device=device,
                )
                dirs_by_gpu[device] = backend.path

            assert dirs_by_gpu["cuda:0"] == dir_a
            assert dirs_by_gpu["cuda:1"] == dir_b
        finally:
            shutil.rmtree(dir_a, ignore_errors=True)
            shutil.rmtree(dir_b, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()

    def test_all_directories_created(self, async_loop, local_cpu_backend):
        """All paths in the list get their directories created."""
        base = tempfile.mkdtemp()
        try:
            paths = [os.path.join(base, f"nvme{i}") for i in range(3)]
            combined = ",".join(paths)
            config = create_test_config(combined)
            LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:0",
            )
            for p in paths:
                assert os.path.isdir(p), f"{p} should exist"
        finally:
            shutil.rmtree(base, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()

    def test_single_path_backward_compat(
        self, temp_disk_path, async_loop, local_cpu_backend
    ):
        """A single path (no commas) works exactly as before."""
        config = create_test_config(temp_disk_path)
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
        )
        assert backend.path == temp_disk_path
        local_cpu_backend.memory_allocator.close()

    def test_path_sharding_default(self, temp_disk_path, async_loop, local_cpu_backend):
        """Default local_disk_path_sharding is 'by_gpu' (backend inits OK)."""
        config = create_test_config(temp_disk_path)
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
        )
        assert backend.path == temp_disk_path
        local_cpu_backend.memory_allocator.close()

    def test_path_sharding_explicit_by_gpu(
        self, temp_disk_path, async_loop, local_cpu_backend
    ):
        """Explicitly setting local_disk_path_sharding='by_gpu' works."""
        config = create_test_config(temp_disk_path, local_disk_path_sharding="by_gpu")
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
        )
        assert backend.path == temp_disk_path
        local_cpu_backend.memory_allocator.close()

    def test_path_sharding_unsupported_raises(
        self, temp_disk_path, async_loop, local_cpu_backend
    ):
        """Unsupported local_disk_path_sharding raises ValueError."""
        config = create_test_config(
            temp_disk_path, local_disk_path_sharding="round_robin"
        )
        with pytest.raises(ValueError, match="Unsupported path sharding strategy"):
            LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:0",
            )

    def test_cpu_dst_device_defaults_to_first_path(self, async_loop, local_cpu_backend):
        """dst_device='cpu' should fall back to device_id=0."""
        dir_a = tempfile.mkdtemp()
        dir_b = tempfile.mkdtemp()
        try:
            combined = f"{dir_a},{dir_b}"
            config = create_test_config(combined)
            backend = LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cpu",
            )
            # device_id=0 -> 0 % 2 = 0 -> dir_a
            assert backend.path == dir_a
        finally:
            shutil.rmtree(dir_a, ignore_errors=True)
            shutil.rmtree(dir_b, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()


class TestParseLocalDisk:
    """Unit tests for the _parse_local_disk config parser."""

    def test_none(self):
        assert _parse_local_disk(None) is None

    def test_single_raw_path(self):
        assert _parse_local_disk("/mnt/nvme0/cache/") == "/mnt/nvme0/cache/"

    def test_single_file_uri(self):
        assert _parse_local_disk("file:///mnt/nvme0/cache/") == "/mnt/nvme0/cache/"

    def test_single_file_uri_no_trailing_slash(self):
        assert _parse_local_disk("file:///mnt/nvme0/cache") == "/mnt/nvme0/cache"

    def test_comma_separated_raw(self):
        result = _parse_local_disk("/mnt/nvme0/,/mnt/nvme1/")
        assert result == "/mnt/nvme0/,/mnt/nvme1/"

    def test_comma_separated_file_uris(self):
        result = _parse_local_disk("file:///mnt/nvme0/,file:///mnt/nvme1/")
        assert result == "/mnt/nvme0/,/mnt/nvme1/"

    def test_mixed_uri_and_raw(self):
        result = _parse_local_disk("file:///mnt/nvme0/,/mnt/nvme1/")
        assert result == "/mnt/nvme0/,/mnt/nvme1/"

    def test_whitespace_around_paths(self):
        result = _parse_local_disk("  /mnt/nvme0/ , /mnt/nvme1/  ")
        assert result == "/mnt/nvme0/,/mnt/nvme1/"

    def test_empty_string(self):
        assert _parse_local_disk("") is None
# ----------------------------------------------------------------------------
# Repro for production incident 2026-04-09 — neuralwatt/inference_frontend#1900
# ----------------------------------------------------------------------------


def _mla_shapes_and_dtypes(
    num_layers: int = 4,
    chunk_size: int = 16,
    k_rope_dim: int = 132,
    latent_dim: int = 576,
):
    """Return (shapes, dtypes) mimicking a two-group MLA KV cache.

    Group 0: K_rope  — [1, num_layers, chunk_size, k_rope_dim], uint8
    Group 1: Latent  — [1, num_layers, chunk_size, latent_dim], bfloat16

    Default sizes are deliberately small so eviction fires within a few
    insertions against a small max_local_disk_size.
    """
    shapes = [
        torch.Size([1, num_layers, chunk_size, k_rope_dim]),
        torch.Size([1, num_layers, chunk_size, latent_dim]),
    ]
    dtypes = [torch.uint8, torch.bfloat16]
    return shapes, dtypes


def _make_mla_memory_obj():
    """Allocate a TensorMemoryObj with two-group MLA layout via AdHoc."""
    shapes, dtypes = _mla_shapes_and_dtypes()
    allocator = AdHocMemoryAllocator(device="cpu")
    memory_obj = allocator.allocate(shapes, dtypes, fmt=MemoryFormat.KV_MLA_FMT)
    assert memory_obj is not None
    return memory_obj


def _make_mla_key(key_id: int) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="mla_test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=hash(("mla", key_id)),
        dtype=torch.bfloat16,
    )


@pytest.fixture
def running_async_loop():
    """An asyncio event loop that is actually driven by a background thread.

    The default `async_loop` fixture in this file creates a loop but never
    runs it, so submit_put_task's `run_coroutine_threadsafe` would never
    flush. The multi-group accounting test must observe completed disk
    writes, so it needs a real running loop.
    """
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    if loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
    if thread.is_alive():
        thread.join(timeout=2.0)
    if not loop.is_closed():
        loop.close()


@pytest.fixture
def small_local_disk_backend(temp_disk_path, running_async_loop, local_cpu_backend):
    """LocalDiskBackend with a tiny max_local_disk_size to force eviction."""
    # 1 MiB cap — large enough to fit a couple of small MLA chunks but
    # small enough that eviction fires within a handful of inserts.
    config = create_test_config(temp_disk_path, max_disk_size=1.0 / 1024.0)
    backend = LocalDiskBackend(
        config=config,
        loop=running_async_loop,
        local_cpu_backend=local_cpu_backend,
        dst_device="cuda",
    )
    yield backend
    backend.close()


def _wait_for_put(backend: LocalDiskBackend, key: CacheEngineKey, timeout: float = 5.0):
    """Block until submit_put_task's background flush has finished for `key`.

    submit_put_task returns None and we cannot await its future from sync
    code, so we poll the in-progress put-task tracker.
    """
    # Standard
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not backend.disk_worker.exists_in_put_tasks(key):
            return
        time.sleep(0.01)
    raise TimeoutError(f"put task for key {key} did not finish within {timeout}s")


class TestLocalDiskBackendMultiGroupAccounting:
    """Repro for GLM-5 production incident 2026-04-09
    (neuralwatt/inference_frontend#1900, #1903).

    Hypothesis: LocalDiskBackend eviction uses meta.size (singular)
    populated from memory_obj.get_physical_size(), which does not match
    the true on-disk byte count (len(memory_obj.byte_array)) for
    multi-group MLA objects. The fix referenced in #1903 is to compute
    size from meta.shapes / meta.dtypes (plural) so the eviction policy
    honors the actual on-disk footprint.

    These tests assert two invariants that must hold after every put:

      1. backend.current_cache_size == sum of bytes actually on disk
      2. sum of bytes on disk <= backend.max_cache_size

    If invariant 1 drifts (in either direction), eviction accounting is
    inconsistent with reality. If invariant 2 is violated, the disk has
    grown past the configured cap — the production-observed failure
    mode that exhausts the LMCache CPU staging pool.
    """

    @pytest.mark.xfail(
        reason="repro for neuralwatt/inference_frontend#1900 — fix pending",
        strict=False,
    )
    def test_current_cache_size_tracks_actual_disk_usage_multi_group(
        self, small_local_disk_backend
    ):
        """Insert multi-group MLA objects until eviction triggers; after
        every put assert accounting matches the actual on-disk footprint.
        """
        backend = small_local_disk_backend

        # Sanity-check the test is exercising a multi-group object.
        probe = _make_mla_memory_obj()
        try:
            shapes, dtypes = _mla_shapes_and_dtypes()
            true_bytes = get_size_bytes(shapes, dtypes)
            assert probe.metadata.shapes is not None
            assert len(probe.metadata.shapes) == 2, (
                "test fixture must produce a two-group MLA memory obj"
            )
            assert len(probe.byte_array) == true_bytes, (
                f"buffer length {len(probe.byte_array)} does not match "
                f"sum-of-groups {true_bytes}"
            )
        finally:
            del probe

        num_inserts = 24
        for i in range(num_inserts):
            mo = _make_mla_memory_obj()
            key = _make_mla_key(i)
            backend.submit_put_task(key, mo)
            _wait_for_put(backend, key)

            actual = 0
            for k in list(backend.dict.keys()):
                path = backend._key_to_path(k)
                if os.path.exists(path):
                    actual += os.path.getsize(path)

            accounted = int(backend.current_cache_size)
            assert accounted == actual, (
                f"accounting drift at i={i}: "
                f"current_cache_size={accounted}, "
                f"actual_disk_bytes={actual}, "
                f"max_cache_size={backend.max_cache_size}"
            )
            assert actual <= backend.max_cache_size, (
                f"disk overflowed cap at i={i}: "
                f"actual_disk_bytes={actual}, "
                f"max_cache_size={backend.max_cache_size}"
            )

    @pytest.mark.xfail(
        reason="repro for neuralwatt/inference_frontend#1900 — fix pending",
        strict=False,
    )
    def test_insert_key_size_matches_buffer_length_multi_group(
        self, small_local_disk_backend
    ):
        """A simpler synchronous variant of the accounting check.

        Calls submit_put_task once with a multi-group MLA object and
        asserts that the size recorded in the disk dict (meta.size)
        equals the on-disk file size (== len(buffer) for the bytes
        written by async_save_bytes_to_disk).

        meta.size is set from memory_obj.get_physical_size() in
        async_save_bytes_to_disk, which the bug hypothesis says drifts
        from len(byte_array) for multi-group objects.
        """
        backend = small_local_disk_backend
        mo = _make_mla_memory_obj()
        key = _make_mla_key(0)

        true_bytes = len(mo.byte_array)
        backend.submit_put_task(key, mo)
        _wait_for_put(backend, key)

        assert key in backend.dict, "put did not complete or insert_key not called"
        meta = backend.dict[key]
        path = backend._key_to_path(key)
        assert os.path.exists(path), f"disk file missing for key: {path}"
        on_disk = os.path.getsize(path)

        assert on_disk == true_bytes, (
            f"on-disk size {on_disk} != true buffer bytes {true_bytes}"
        )
        assert meta.size == on_disk, (
            f"meta.size used by eviction accounting ({meta.size}) does "
            f"not match actual on-disk bytes ({on_disk}) for multi-group "
            f"MLA object with shapes={meta._shapes} dtypes={meta._dtypes}"
        )

    @pytest.mark.xfail(
        reason="repro for neuralwatt/inference_frontend#1900 — fix pending",
        strict=False,
    )
    def test_production_allocator_path_drifts_for_multi_group(
        self, small_local_disk_backend
    ):
        """Same accounting check, but uses local_cpu_backend.allocate()
        — the production code path. MixedMemoryAllocator/TensorMemory
        Allocator align allocations up to 4096 bytes, so phy_size is
        > len(byte_array) for multi-group MLA objects whose true byte
        sum is not 4096-aligned. The eviction loop accounts in
        phy_size, but only len(buffer) bytes ever hit the disk, so
        current_cache_size persistently over-reports actual disk usage.

        This is the inverse direction from the AdHocMemoryAllocator
        test: there phy_size=0 under-reports; here phy_size>len(buffer)
        over-reports. Both are accounting drift, both are caught by the
        same invariant.
        """
        backend = small_local_disk_backend
        cpu = backend.local_cpu_backend
        shapes, dtypes = _mla_shapes_and_dtypes()

        # Allocate via the production path: local_cpu_backend.allocate
        # → MixedMemoryAllocator → TensorMemoryAllocator. eviction=False
        # so we never page-evict during this test.
        mo = cpu.allocate(shapes, dtypes, fmt=MemoryFormat.KV_MLA_FMT,
                          eviction=False)
        assert mo is not None, "production allocator returned None"

        # The bug: phy_size != len(byte_array) for non-page-aligned
        # multi-group MLA chunks.
        phy = mo.get_physical_size()
        true_bytes = len(mo.byte_array)
        # We assert this *informationally* — the test below catches the
        # downstream consequence regardless of which direction it drifts.
        if phy != true_bytes:
            # Expected for the production allocator: phy is 4096-aligned
            # round-up of get_size_bytes(shapes, dtypes).
            pass

        key = _make_mla_key(99)
        backend.submit_put_task(key, mo)
        _wait_for_put(backend, key)

        path = backend._key_to_path(key)
        assert os.path.exists(path), f"disk file missing for key: {path}"
        on_disk = os.path.getsize(path)
        accounted = int(backend.current_cache_size)

        assert on_disk == true_bytes, (
            f"on-disk bytes ({on_disk}) should equal true buffer bytes "
            f"({true_bytes}) — async_save_bytes_to_disk writes len(buffer)"
        )
        assert accounted == on_disk, (
            f"production-allocator accounting drift: "
            f"current_cache_size={accounted}, on_disk_bytes={on_disk}, "
            f"phy_size={phy}, len(buffer)={true_bytes}, "
            f"shapes={shapes}, dtypes={dtypes}"
        )
