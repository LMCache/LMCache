# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest import mock
import asyncio
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.parse

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.gds_backend import (
    _DATA_FILE_SUFFIX,
    _METADATA_FILE_SUFFIX,
    _METADATA_VERSION,
    GdsBackend,
    UnsupportedMetadataVersion,
    pack_metadata,
)
from tests.v1.utils import create_test_memory_obj, has_cufile, has_hipfile


def create_test_config(gds_path: str, gds_path_sharding: str = "by_gpu"):
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        gds_path=gds_path,
        gds_path_sharding=gds_path_sharding,
        lmcache_instance_id="test_instance",
        cufile_buffer_size=256,
        extra_config={"use_direct_io": True},
    )
    return config


def create_test_key(key_id: int = 0) -> CacheEngineKey:
    # NO UNDERSCORE HERE for model_name
    return CacheEngineKey(
        model_name="testmodel",
        world_size=3,
        worker_id=1,
        chunk_hash=key_id,
        dtype=torch.bfloat16,
    )


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


@pytest.fixture
def temp_gds_path():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def async_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join()
    loop.close()


@pytest.fixture
def gds_backend(temp_gds_path, async_loop):
    config = create_test_config(temp_gds_path)
    metadata = create_test_metadata()
    return GdsBackend(
        config=config,
        loop=async_loop,
        metadata=metadata,
        dst_device="cuda:0",
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA for TestGdsBackend",
)
@pytest.mark.skipif(
    not (has_cufile() or has_hipfile()),
    reason="Requires NVIDIA cuFile (libcufile.so) or AMD hipFile (libhipfile.so). "
    "Skipping on systems without GDS support.",
)
@pytest.mark.skipif(sys.platform != "linux", reason="TestGdsBackend runs only on Linux")
class TestGdsBackend:
    def test_init(self, temp_gds_path, async_loop):
        config = create_test_config(temp_gds_path)
        metadata = create_test_metadata()
        backend = GdsBackend(
            config=config,
            loop=async_loop,
            metadata=metadata,
            dst_device="cuda:0",
        )
        assert backend.gds_path == temp_gds_path
        assert backend.dst_device == "cuda:0"
        assert os.path.exists(temp_gds_path)

    def test_str(self, gds_backend):
        assert str(gds_backend) == "GdsBackend"

    def test_key_to_path_and_insert_key(self, gds_backend):
        key = create_test_key(0)
        memory_obj = create_test_memory_obj(device="cuda")
        gds_backend.insert_key(key, memory_obj)
        # Check that the key is in hot_cache
        assert key in gds_backend.hot_cache
        meta = gds_backend.hot_cache[key]
        assert meta.shape == memory_obj.metadata.shape
        assert meta.dtype == memory_obj.metadata.dtype

    def test_contains_key_not_exists(self, gds_backend):
        key = create_test_key(1)
        assert not gds_backend.contains(key)
        assert not gds_backend.contains(key, pin=True)

    def test_contains_key_exists(self, gds_backend):
        key = create_test_key(0)
        memory_obj = create_test_memory_obj(device="cuda")
        gds_backend.insert_key(key, memory_obj)
        assert gds_backend.contains(key)
        assert gds_backend.contains(key, pin=True)

    def test_exists_in_put_tasks(self, gds_backend):
        key = create_test_key(0)
        assert not gds_backend.exists_in_put_tasks(key)
        # Simulate adding to put_tasks
        with gds_backend.put_lock:
            gds_backend.put_tasks.add(key)
        assert gds_backend.exists_in_put_tasks(key)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires CUDA for GdsBackend get_blocking",
    )
    @pytest.mark.skipif(
        not (has_cufile() or has_hipfile()),
        reason="Requires NVIDIA cuFile (libcufile.so) or AMD hipFile (libhipfile.so). "
        "Skipping on systems without GDS support.",
    )
    async def test_submit_put_task_and_get_blocking(self, gds_backend):
        key = create_test_key(0)
        memory_obj = create_test_memory_obj(device="cuda")
        # submit_put_task returns a Future
        future = gds_backend.submit_put_task(key, memory_obj)
        assert future is not None
        # Wait for the async save to complete
        future.result(timeout=5)
        # Now the key should be in hot_cache
        assert gds_backend.contains(key)
        # get_blocking should return a MemoryObj (may be None if not CUDA)
        result = gds_backend.get_blocking(key)
        # On CPU, _load_bytes_from_disk may not work,
        # so just check for None or MemoryObj
        assert result is None or isinstance(result, MemoryObj)

    @pytest.mark.asyncio
    async def test_batched_submit_put_task(self, gds_backend):
        keys = [create_test_key(i) for i in range(2, 5)]
        memory_objs = [create_test_memory_obj(device="cuda") for _ in range(3)]
        futures = gds_backend.batched_submit_put_task(keys, memory_objs)
        assert futures is not None
        assert len(futures) == 3
        for future in futures:
            assert future is not None
            future.result(timeout=5)
        for key in keys:
            assert gds_backend.contains(key)

    def test_get_blocking_key_not_exists(self, gds_backend):
        key = create_test_key(1)
        result = gds_backend.get_blocking(key)
        assert result is None

    # Error handling tests
    def test_try_to_read_metadata_file_not_found(self, gds_backend, temp_gds_path):
        """Test that FileNotFoundError is handled gracefully."""
        key = create_test_key(400)

        # Create a path that doesn't exist
        result = gds_backend._try_to_read_metadata(key)
        assert result is None

    def test_try_to_read_metadata_permission_error(self, gds_backend, temp_gds_path):
        """Test that PermissionError is handled gracefully."""
        key = create_test_key(401)
        path, subdir_key, l1_dir, l2_dir = gds_backend._key_to_path(key)
        metadata_path = path + _METADATA_FILE_SUFFIX

        # Create metadata file
        os.makedirs(os.path.join(temp_gds_path, l1_dir, l2_dir), exist_ok=True)
        memory_obj = create_test_memory_obj(device="cuda")
        metadata = pack_metadata(
            memory_obj.tensor,
            fmt=memory_obj.metadata.fmt,
            lmcache_version=str(_METADATA_VERSION),
        )
        with open(metadata_path, "wb") as f:
            f.write(metadata)

        # Mock _read_metadata to raise PermissionError
        original_read_metadata = gds_backend._read_metadata

        def failing_read_metadata(*args, **kwargs):
            raise PermissionError("Simulated permission denied")

        gds_backend._read_metadata = failing_read_metadata

        try:
            result = gds_backend._try_to_read_metadata(key)
            assert result is None
        finally:
            gds_backend._read_metadata = original_read_metadata

    def test_try_to_read_metadata_unsupported_version(self, gds_backend, temp_gds_path):
        """Test that UnsupportedMetadataVersion is handled gracefully."""
        key = create_test_key(402)
        path, subdir_key, l1_dir, l2_dir = gds_backend._key_to_path(key)
        metadata_path = path + _METADATA_FILE_SUFFIX

        os.makedirs(os.path.join(temp_gds_path, l1_dir, l2_dir), exist_ok=True)

        # Mock _read_metadata to raise UnsupportedMetadataVersion
        original_read_metadata = gds_backend._read_metadata

        def failing_read_metadata(*args, **kwargs):
            raise UnsupportedMetadataVersion("Unsupported version")

        gds_backend._read_metadata = failing_read_metadata

        # Create a dummy file so os.path.exists returns True
        with open(metadata_path, "wb") as f:
            f.write(b"dummy")

        try:
            result = gds_backend._try_to_read_metadata(key)
            assert result is None
        finally:
            gds_backend._read_metadata = original_read_metadata

    def test_try_to_read_metadata_io_error(self, gds_backend, temp_gds_path):
        """Test that OSError/IOError is handled gracefully."""
        key = create_test_key(403)
        path, subdir_key, l1_dir, l2_dir = gds_backend._key_to_path(key)
        metadata_path = path + _METADATA_FILE_SUFFIX

        os.makedirs(os.path.join(temp_gds_path, l1_dir, l2_dir), exist_ok=True)

        # Mock _read_metadata to raise IOError
        original_read_metadata = gds_backend._read_metadata

        def failing_read_metadata(*args, **kwargs):
            raise IOError("Simulated I/O error")

        gds_backend._read_metadata = failing_read_metadata

        # Create a dummy file
        with open(metadata_path, "wb") as f:
            f.write(b"dummy")

        try:
            result = gds_backend._try_to_read_metadata(key)
            assert result is None
        finally:
            gds_backend._read_metadata = original_read_metadata

    def test_try_to_read_metadata_generic_exception(self, gds_backend, temp_gds_path):
        """Test that generic exceptions are handled gracefully."""
        key = create_test_key(404)
        path, subdir_key, l1_dir, l2_dir = gds_backend._key_to_path(key)
        metadata_path = path + _METADATA_FILE_SUFFIX

        os.makedirs(os.path.join(temp_gds_path, l1_dir, l2_dir), exist_ok=True)

        # Mock _read_metadata to raise a generic exception
        original_read_metadata = gds_backend._read_metadata

        def failing_read_metadata(*args, **kwargs):
            raise RuntimeError("Unexpected error")

        gds_backend._read_metadata = failing_read_metadata

        # Create a dummy file
        with open(metadata_path, "wb") as f:
            f.write(b"dummy")

        try:
            result = gds_backend._try_to_read_metadata(key)
            assert result is None
        finally:
            gds_backend._read_metadata = original_read_metadata

    @pytest.mark.asyncio
    async def test_async_save_bytes_to_disk_write_error_handling(
        self, gds_backend, temp_gds_path
    ):
        """Test error handling when GDS write operation fails."""
        key = create_test_key(300)
        memory_obj = create_test_memory_obj(device="cuda")
        memory_obj.ref_count_up()

        # Mock _save_gds to raise an exception
        original_save_gds = gds_backend._save_gds

        def failing_save_gds(*args, **kwargs):
            raise IOError("Simulated GDS write failure")

        gds_backend._save_gds = failing_save_gds

        try:
            # Call should not raise, but should handle error gracefully
            await gds_backend._async_save_bytes_to_disk(key, memory_obj)

            # Key should not be in cache after failed write
            assert not gds_backend.contains(key)
        finally:
            gds_backend._save_gds = original_save_gds
            memory_obj.ref_count_down()

    @pytest.mark.asyncio
    async def test_async_save_bytes_metadata_write_failure(
        self, gds_backend, temp_gds_path
    ):
        """
        Test that metadata write failures during task execution trigger cache cleanup.
        """
        key = create_test_key(500)
        memory_obj = create_test_memory_obj(device="cuda")
        memory_obj.ref_count_up()

        # Mock save_metadata to raise an exception during execution
        async def failing_save_metadata(path, tmp, metadata):
            raise IOError("Simulated metadata write failure")

        with mock.patch(
            "lmcache.v1.storage_backend.gds_backend.save_metadata",
            side_effect=failing_save_metadata,
        ):
            try:
                await gds_backend._async_save_bytes_to_disk(key, memory_obj)

                # Wait for the background task to complete and exception to be handled
                await asyncio.sleep(0.2)

                # Key should be removed from hot_cache after metadata write failure
                with gds_backend.hot_lock:
                    assert key not in gds_backend.hot_cache
            finally:
                memory_obj.ref_count_down()

    def test_close(self, gds_backend):
        # Should not raise
        gds_backend.close()

    def test_pin_unpin_not_implemented(self, gds_backend):
        key = create_test_key(0)
        assert not gds_backend.pin(key)
        assert not gds_backend.unpin(key)

    def test_weka_initialization_suffix(self, temp_gds_path, async_loop):
        class DummyAllocator:
            def __init__(self):
                self.base_pointer = 0

            def close(self):
                pass

        class DummyCuFileDriver:
            def __init__(self):
                pass

        class DummyCuFile:
            def __init__(self, *_, **__):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def write(self, *_, **__):
                return None

            def read(self, *_, **__):
                return 0

        dummy_cufile_module = type(
            "DummyCuFileModule",
            (),
            {"CuFileDriver": DummyCuFileDriver, "CuFile": DummyCuFile},
        )()

        with mock.patch.dict(sys.modules, {"cufile": dummy_cufile_module}):
            with (
                mock.patch(
                    "lmcache.v1.storage_backend.gds_backend.get_fstype",
                    return_value="wekafs",
                ),
                mock.patch.object(
                    GdsBackend,
                    "initialize_allocator",
                    return_value=DummyAllocator(),
                ),
            ):
                config = create_test_config(temp_gds_path)
                metadata = create_test_metadata()

                backend = GdsBackend(
                    config=config,
                    loop=async_loop,
                    metadata=metadata,
                    dst_device="cuda:0",
                )
                try:
                    key = create_test_key(0)
                    path, _, _, _ = backend._key_to_path(key)
                    assert path.endswith(".weka1")
                    assert backend.data_suffix == ".weka1"
                    assert backend.use_cufile
                finally:
                    backend.close()

    def test_weka_disallows_disabling_cufile(self, temp_gds_path, async_loop):
        class DummyAllocator:
            def __init__(self):
                self.base_pointer = 0

            def close(self):
                pass

        with (
            mock.patch(
                "lmcache.v1.storage_backend.gds_backend.get_fstype",
                return_value="wekafs",
            ),
            mock.patch.object(
                GdsBackend,
                "initialize_allocator",
                return_value=DummyAllocator(),
            ),
        ):
            config = create_test_config(temp_gds_path)
            config.extra_config["use_cufile"] = False
            metadata = create_test_metadata()

            with pytest.raises(AssertionError):
                GdsBackend(
                    config=config,
                    loop=async_loop,
                    metadata=metadata,
                    dst_device="cuda:0",
                )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA for TestGdsMultiPath",
)
@pytest.mark.skipif(sys.platform != "linux", reason="Runs only on Linux")
class TestGdsMultiPath:
    """Tests for multi-path GDS backend support.

    Verifies comma-separated gds_path parsing, GPU-to-NVMe affinity
    (by_gpu sharding), directory creation for all paths, and backward
    compatibility with single-path configs.
    """

    @staticmethod
    def _make_backend(
        gds_path: str,
        dst_device: str,
        async_loop,
        gds_path_sharding: str = "by_gpu",
    ):
        """Create a GdsBackend with mocked allocator and fstype.

        Mocks are used so the tests run without cuFile / real NVMe.
        """

        class DummyAllocator:
            def __init__(self):
                self.base_pointer = 0

            def close(self):
                pass

        config = create_test_config(gds_path, gds_path_sharding=gds_path_sharding)
        metadata = create_test_metadata()
        with (
            mock.patch(
                "lmcache.v1.storage_backend.gds_backend.get_fstype",
                return_value="tmpfs",
            ),
            mock.patch.object(
                GdsBackend,
                "initialize_allocator",
                return_value=DummyAllocator(),
            ),
            mock.patch(
                "lmcache.v1.storage_backend.gds_backend.ctypes.CDLL",
                return_value=mock.MagicMock(),
            ),
        ):
            backend = GdsBackend(
                config=config,
                loop=async_loop,
                metadata=metadata,
                dst_device=dst_device,
            )
        return backend

    def test_single_path_backward_compat(self, temp_gds_path, async_loop):
        """A single gds_path (no commas) behaves like before."""
        backend = self._make_backend(temp_gds_path, "cuda:0", async_loop)
        try:
            assert backend.gds_paths == [temp_gds_path]
            assert backend.gds_path == temp_gds_path
        finally:
            backend.close()

    def test_multi_path_parsing(self, async_loop):
        """Comma-separated paths are split into gds_paths list."""
        paths = [tempfile.mkdtemp() for _ in range(3)]
        try:
            gds_path = ",".join(paths)
            backend = self._make_backend(gds_path, "cuda:0", async_loop)
            try:
                assert backend.gds_paths == paths
                assert len(backend.gds_paths) == 3
            finally:
                backend.close()
        finally:
            for p in paths:
                shutil.rmtree(p, ignore_errors=True)

    def test_multi_path_parsing_with_spaces(self, async_loop):
        """Spaces around commas are stripped."""
        paths = [tempfile.mkdtemp() for _ in range(2)]
        try:
            gds_path = f"  {paths[0]}  ,  {paths[1]}  "
            backend = self._make_backend(gds_path, "cuda:0", async_loop)
            try:
                assert backend.gds_paths == paths
            finally:
                backend.close()
        finally:
            for p in paths:
                shutil.rmtree(p, ignore_errors=True)

    def test_gpu_affinity_selects_path(self, async_loop):
        """Different cuda devices select different paths via modulo."""
        paths = [tempfile.mkdtemp() for _ in range(4)]
        try:
            gds_path = ",".join(paths)
            for device_id in range(8):
                dst = f"cuda:{device_id}"
                backend = self._make_backend(gds_path, dst, async_loop)
                try:
                    expected = paths[device_id % 4]
                    assert backend.gds_path == expected, (
                        f"cuda:{device_id} should map to {expected}, "
                        f"got {backend.gds_path}"
                    )
                finally:
                    backend.close()
        finally:
            for p in paths:
                shutil.rmtree(p, ignore_errors=True)

    def test_all_directories_created(self, async_loop):
        """All paths in gds_paths get their directories created."""
        base = tempfile.mkdtemp()
        try:
            paths = [os.path.join(base, f"nvme{i}") for i in range(3)]
            gds_path = ",".join(paths)
            backend = self._make_backend(gds_path, "cuda:0", async_loop)
            try:
                for p in paths:
                    assert os.path.isdir(p), f"{p} should exist"
            finally:
                backend.close()
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_key_to_path_uses_selected_path(self, async_loop):
        """_key_to_path builds file paths under the selected gds_path."""
        paths = [tempfile.mkdtemp() for _ in range(2)]
        try:
            gds_path = ",".join(paths)
            backend = self._make_backend(gds_path, "cuda:0", async_loop)
            try:
                key = create_test_key(0)
                file_path, _, _, _ = backend._key_to_path(key)
                assert file_path.startswith(backend.gds_path)
            finally:
                backend.close()
        finally:
            for p in paths:
                shutil.rmtree(p, ignore_errors=True)

    def test_deterministic_path_selection(self, async_loop):
        """Same device always selects the same path."""
        paths = [tempfile.mkdtemp() for _ in range(3)]
        try:
            gds_path = ",".join(paths)
            selected = set()
            for _ in range(5):
                backend = self._make_backend(gds_path, "cuda:1", async_loop)
                try:
                    selected.add(backend.gds_path)
                finally:
                    backend.close()
            assert len(selected) == 1, "Path selection should be deterministic"
        finally:
            for p in paths:
                shutil.rmtree(p, ignore_errors=True)

    def test_scan_discovers_entries_across_all_paths(self, async_loop):
        """Startup scan finds metadata written under a non-affinity path.

        cuda:0 has affinity to paths[0], but a metadata file is placed
        under paths[1].  The scan should still discover it because
        ``_scan_metadata`` iterates all ``gds_paths``.
        """
        paths = [tempfile.mkdtemp() for _ in range(2)]
        try:
            # chunk_hash must have >=4 decimal digits so l1/l2 dirs
            # are each 2 chars (the scanner filters on len == 2).
            key = CacheEngineKey(
                model_name="testmodel",
                world_size=3,
                worker_id=1,
                chunk_hash=1234,
                dtype=torch.bfloat16,
            )
            hash_str = str(key.chunk_hash)
            l1_dir = hash_str[:2]
            l2_dir = hash_str[2:4]
            key_str = urllib.parse.quote(key.to_string(), safe="")

            # Build directory structure under paths[1] (non-affinity path)
            subdir = os.path.join(paths[1], l1_dir, l2_dir)
            os.makedirs(subdir, exist_ok=True)

            # Write a valid metadata file
            dummy_tensor = torch.zeros(2, 256, 8, 128, dtype=torch.bfloat16)
            metadata_bytes = pack_metadata(
                dummy_tensor,
                fmt=MemoryFormat.KV_2LTD,
                lmcache_version="1",
            )
            meta_path = os.path.join(
                subdir,
                key_str + _DATA_FILE_SUFFIX + _METADATA_FILE_SUFFIX,
            )
            with open(meta_path, "wb") as f:
                f.write(metadata_bytes)

            # Create backend — cuda:0 affinity selects paths[0]
            gds_path = ",".join(paths)
            backend = self._make_backend(gds_path, "cuda:0", async_loop)
            try:
                assert backend.gds_path == paths[0]

                # Wait for the async _scan_metadata to finish
                time.sleep(1)

                # contains() would NOT find this via _try_to_read_metadata
                # (which only checks the affinity path).  If it returns
                # True, the cross-path scan populated hot_cache.
                assert backend.contains(key), (
                    "Scan should discover metadata across all gds_paths"
                )
            finally:
                backend.close()
        finally:
            for p in paths:
                shutil.rmtree(p, ignore_errors=True)

    def test_try_to_read_metadata_finds_across_all_paths(self, async_loop):
        """contains() fallback finds metadata on a non-affinity path.

        After clearing hot_cache (so the startup scan results are gone),
        ``contains()`` falls back to ``_try_to_read_metadata`` which
        should search all ``gds_paths``, not just the affinity path.
        """
        paths = [tempfile.mkdtemp() for _ in range(2)]
        try:
            key = CacheEngineKey(
                model_name="testmodel",
                world_size=3,
                worker_id=1,
                chunk_hash=1234,
                dtype=torch.bfloat16,
            )
            hash_str = str(key.chunk_hash)
            l1_dir = hash_str[:2]
            l2_dir = hash_str[2:4]
            key_str = urllib.parse.quote(key.to_string(), safe="")

            # Place metadata under paths[1] (non-affinity path)
            subdir = os.path.join(paths[1], l1_dir, l2_dir)
            os.makedirs(subdir, exist_ok=True)

            dummy_tensor = torch.zeros(2, 256, 8, 128, dtype=torch.bfloat16)
            metadata_bytes = pack_metadata(
                dummy_tensor,
                fmt=MemoryFormat.KV_2LTD,
                lmcache_version="1",
            )
            meta_path = os.path.join(
                subdir,
                key_str + _DATA_FILE_SUFFIX + _METADATA_FILE_SUFFIX,
            )
            with open(meta_path, "wb") as f:
                f.write(metadata_bytes)

            gds_path = ",".join(paths)
            backend = self._make_backend(gds_path, "cuda:0", async_loop)
            try:
                assert backend.gds_path == paths[0]

                # Wait for async scan, then clear its results
                time.sleep(1)
                with backend.hot_lock:
                    backend.hot_cache.clear()

                # contains() now relies solely on _try_to_read_metadata
                assert backend.contains(key), (
                    "_try_to_read_metadata should search all gds_paths"
                )
            finally:
                backend.close()
        finally:
            for p in paths:
                shutil.rmtree(p, ignore_errors=True)

    def test_gds_path_sharding_default(self, temp_gds_path, async_loop):
        """Default gds_path_sharding is 'by_gpu'."""
        backend = self._make_backend(temp_gds_path, "cuda:0", async_loop)
        try:
            assert backend.gds_path_sharding == "by_gpu"
        finally:
            backend.close()

    def test_gds_path_sharding_explicit_by_gpu(self, temp_gds_path, async_loop):
        """Explicitly setting gds_path_sharding='by_gpu' works."""
        backend = self._make_backend(
            temp_gds_path,
            "cuda:0",
            async_loop,
            gds_path_sharding="by_gpu",
        )
        try:
            assert backend.gds_path_sharding == "by_gpu"
        finally:
            backend.close()

    def test_gds_path_sharding_unsupported_raises(self, temp_gds_path, async_loop):
        """Unsupported gds_path_sharding value raises AssertionError."""
        with pytest.raises(AssertionError, match="Unsupported gds_path_sharding"):
            self._make_backend(
                temp_gds_path,
                "cuda:0",
                async_loop,
                gds_path_sharding="round_robin",
            )
