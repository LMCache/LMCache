# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Samsung Electronics Co., Ltd.All Rights Reserved
# Authors: Wenwen Chen <wenwen.chen@samsung.com>

"""Tests for Hf3fsL2AdapterConfig and Hf3fs native connector.

This module tests:
1. Hf3fsL2AdapterConfig validation (Python-only, no extension needed)
2. Native C++ 3FS connector I/O operations (requires extension + 3FS cluster)
3. MP mode L2 adapter end-to-end operations (requires extension + 3FS cluster)

Build-time behavior:
- Hf3fs extension is optional — controlled by BUILD_HF3FS or HF3FS_INCLUDE_DIR
- If extension is not built, all hf3fs tests are skipped

Runtime test gating:
- Extension missing → skip all hf3fs tests
- Extension present, cluster unavailable → config tests only
- Extension present, cluster available → full test suite
"""

# Standard
from pathlib import Path
from typing import Any
import os
import shutil
import time
import uuid

# Third Party
import pytest
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.l2_adapters.hf3fs_l2_adapter import Hf3fsL2AdapterConfig
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    _object_key_to_string,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Environment / mode detection
# ---------------------------------------------------------------------------

# 3FS cluster params from environment
_HF3FS_MOUNT_POINT = os.environ.get("HF3FS_MOUNT_POINT", "")


def _has_hf3fs_extension() -> bool:
    """Check if lmcache_hf3fs C++ extension is available."""
    try:
        # First Party
        from lmcache.lmcache_hf3fs import LMCacheHf3fsClient  # noqa: F401

        return True
    except ImportError:
        return False


def _has_3fs_cluster() -> bool:
    """Check if 3FS cluster is accessible.

    HF3FS_MOUNT_POINT must be set and contain a 3fs-virt subdirectory.
    """
    mp = os.environ.get("HF3FS_MOUNT_POINT")
    if not mp:
        return False
    return os.path.isdir(os.path.join(mp, "3fs-virt"))


# ---------------------------------------------------------------------------
# Skip markers (mirrors test_mooncake_store_l2_adapter.py pattern)
# ---------------------------------------------------------------------------

requires_hf3fs = pytest.mark.skipif(
    not _has_hf3fs_extension(),
    reason="lmcache_hf3fs C++ extension not available",
)

requires_hf3fs_cluster = pytest.mark.skipif(
    not _has_hf3fs_extension() or not _has_3fs_cluster(),
    reason=("lmcache_hf3fs C++ extension not available or 3FS cluster not accessible"),
)

# Empty layout used for lookup-and-lock calls (no payload, just existence check)
_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_object_key(chunk_id: int, model_name: str = "test_model") -> ObjectKey:
    """Create an ObjectKey using LMCache's IntHash2Bytes."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
    )


def create_memory_obj(size: int = 1024, fill_value: float = 1.0) -> TensorMemoryObj:
    """Create a TensorMemoryObj for testing."""
    raw_data = torch.empty(size, dtype=torch.float32)
    raw_data.fill_(fill_value)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.float32,
        address=0,
        phy_size=size * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def _wait_for_completions(client, timeout: float = 10.0):
    """Wait for eventfd signal and drain completions.

    Polls the eventfd until completions are available or timeout expires.
    """
    # Standard
    import select

    event_fd = client.event_fd()
    start_time = os.times().elapsed if hasattr(os, "times") else 0

    while True:
        completions = client.drain_completions()
        if completions:
            return completions

        elapsed = os.times().elapsed if hasattr(os, "times") else 0
        if elapsed - start_time > timeout:
            raise TimeoutError(f"Timed out waiting for completions after {timeout}s")

        try:
            select.select([event_fd], [], [], 0.1)
        except (ValueError, OSError):
            break

    return client.drain_completions()


def _wait_for_store(adapter, timeout: float = 10.0):
    """Wait for the L2 adapter's store eventfd to be signaled.

    The NativeConnectorL2Adapter uses a background demux thread that polls
    the native client's eventfd. This helper waits for the demux thread to
    process completions and signal the store eventfd.
    """
    # Standard
    import select

    store_fd = adapter.get_store_event_fd()
    start = os.times().elapsed if hasattr(os, "times") else 0

    while True:
        completed = adapter.pop_completed_store_tasks()
        if completed:
            return completed

        elapsed = os.times().elapsed if hasattr(os, "times") else 0
        if elapsed - start > timeout:
            raise TimeoutError(
                f"Timed out waiting for store completion after {timeout}s"
            )

        try:
            select.select([store_fd], [], [], 0.1)
        except (ValueError, OSError):
            break

    return adapter.pop_completed_store_tasks()


def _wait_for_lookup(adapter, timeout: float = 10.0):
    """Wait for the L2 adapter's lookup to complete.

    Polls the internal _completed_lookups dict directly, which is more
    reliable than waiting on the eventfd since the demux thread may have
    processing delays.
    """
    start = os.times().elapsed if hasattr(os, "times") else 0

    while True:
        elapsed = os.times().elapsed if hasattr(os, "times") else 0
        if elapsed - start > timeout:
            raise TimeoutError(f"Timed out waiting for lookup after {timeout}s")

        with adapter._lock:
            if adapter._completed_lookups:
                return
        time.sleep(0.1)


def _wait_for_load(adapter, timeout: float = 10.0):
    """Wait for the L2 adapter's load to complete."""
    start = os.times().elapsed if hasattr(os, "times") else 0

    while True:
        elapsed = os.times().elapsed if hasattr(os, "times") else 0
        if elapsed - start > timeout:
            raise TimeoutError(f"Timed out waiting for load after {timeout}s")

        with adapter._lock:
            if adapter._completed_loads:
                return
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hf3fs_test_mount_point(tmp_path) -> Path:
    """Create and yield a 3FS mount point directory.

    When 3FS cluster is available, returns the HF3FS_MOUNT_POINT directory.
    Otherwise returns a tmp_path subdirectory for config-only tests.
    """
    if _has_3fs_cluster():
        return Path(_HF3FS_MOUNT_POINT)
    mp = tmp_path / "3fs_mount"
    mp.mkdir()
    return mp


@pytest.fixture
def hf3fs_test_base_path(hf3fs_test_mount_point: Path, tmp_path) -> Path:
    """Create and yield a base path under the mount point.

    When 3FS cluster is available, creates a unique test subdirectory.
    Otherwise creates a tmp_path subdirectory for config-only tests.
    """
    if _has_3fs_cluster():
        test_dir = hf3fs_test_mount_point / f"test_{uuid.uuid4().hex[:8]}"
        test_dir.mkdir(parents=True, exist_ok=True)
        return test_dir
    bp = hf3fs_test_mount_point / "path1"
    bp.mkdir(exist_ok=True)
    return bp


@pytest.fixture
def hf3fs_test_multi_base_paths(hf3fs_test_mount_point: Path, tmp_path):
    """Create and yield multiple base paths under the mount point.

    When 3FS cluster is available, creates two unique test subdirectories.
    Otherwise creates tmp_path subdirectories for config-only tests.
    """
    if _has_3fs_cluster():
        path1 = hf3fs_test_mount_point / f"test_{uuid.uuid4().hex[:8]}_a"
        path2 = hf3fs_test_mount_point / f"test_{uuid.uuid4().hex[:8]}_b"
        path1.mkdir(parents=True, exist_ok=True)
        path2.mkdir(parents=True, exist_ok=True)
        return path1, path2
    path1 = hf3fs_test_mount_point / "path1"
    path2 = hf3fs_test_mount_point / "path2"
    path1.mkdir(exist_ok=True)
    path2.mkdir(exist_ok=True)
    return path1, path2


@pytest.fixture
def hf3fs_native_client(hf3fs_test_base_path):
    """Provide LMCacheHf3fsClient for integration tests.

    Requires both the C++ extension and 3FS cluster.
    """
    test_dir = hf3fs_test_base_path
    mount_point = test_dir.parent
    base_paths = str(test_dir)

    # First Party
    from lmcache.lmcache_hf3fs import LMCacheHf3fsClient

    client = LMCacheHf3fsClient(
        mount_point=str(mount_point),
        base_paths=base_paths,
        num_workers=2,
        ior_entries=128,
        io_depth=0,
        numa_id=-1,
        iov_size=209715200,
        time_out=200,
        enable_key_buffer=True,
    )
    yield client, str(mount_point), base_paths
    client.close()
    # Cleanup test directory if it was created for this test
    if _has_3fs_cluster() and test_dir.exists():
        shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture
def hf3fs_l2_adapter(hf3fs_native_client):
    """Provide NativeConnectorL2Adapter wrapping LMCacheHf3fsClient."""
    # First Party
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
        NativeConnectorL2Adapter,
    )

    client = hf3fs_native_client[0]
    return NativeConnectorL2Adapter(
        client,
        type_name="Hf3fsL2Adapter",
    )


# =============================================================================
# Config Unit Tests (no C++ extension needed)
# =============================================================================


class TestHf3fsL2AdapterConfig:
    """Unit tests for Hf3fsL2AdapterConfig validation."""

    def test_valid_config(self, hf3fs_test_mount_point, hf3fs_test_base_path):
        """Test valid config creation."""
        config_dict = {
            "mount_point": str(hf3fs_test_mount_point),
            "base_paths": str(hf3fs_test_base_path),
            "num_workers": 4,
        }
        config = Hf3fsL2AdapterConfig.from_dict(config_dict)
        assert config.mount_point == str(hf3fs_test_mount_point)
        assert config.base_paths == str(hf3fs_test_base_path)
        assert config.num_workers == 4
        assert config.ior_entries == 256  # default
        assert config.io_depth == 0  # default
        assert config.numa_id == -1  # default
        assert config.iov_size == 209715200  # default

    def test_invalid_mount_point_not_exists(self):
        """Test validation fails when mount_point doesn't exist."""
        config_dict = {
            "mount_point": "/nonexistent/path",
            "base_paths": "/some/path",
        }
        with pytest.raises(ValueError, match="does not exist"):
            Hf3fsL2AdapterConfig.from_dict(config_dict)

    def test_invalid_mount_point_not_directory(self, tmp_path):
        """Test validation fails when mount_point is not a directory."""
        mount_point = tmp_path / "file"
        mount_point.touch()

        config_dict = {
            "mount_point": str(mount_point),
            "base_paths": str(tmp_path),
        }
        with pytest.raises(ValueError, match="is not a directory"):
            Hf3fsL2AdapterConfig.from_dict(config_dict)

    def test_invalid_base_path_not_subdirectory(self, hf3fs_test_mount_point, tmp_path):
        """Test validation fails when base_path is not subdirectory."""
        other_path = tmp_path / "other_path"
        other_path.mkdir()

        config_dict = {
            "mount_point": str(hf3fs_test_mount_point),
            "base_paths": str(other_path),
        }
        with pytest.raises(ValueError, match="not a subdirectory"):
            Hf3fsL2AdapterConfig.from_dict(config_dict)

    def test_invalid_base_path_not_exists(self, hf3fs_test_mount_point):
        """Test validation fails when base_path doesn't exist."""
        base_path = hf3fs_test_mount_point / "nonexistent"

        config_dict = {
            "mount_point": str(hf3fs_test_mount_point),
            "base_paths": str(base_path),
        }
        with pytest.raises(ValueError, match="does not exist"):
            Hf3fsL2AdapterConfig.from_dict(config_dict)

    def test_invalid_ior_entries_range(self, hf3fs_test_base_path):
        """Test validation fails for ior_entries outside [128, 1024]."""
        config_dict = {
            "mount_point": str(hf3fs_test_base_path.parent),
            "base_paths": str(hf3fs_test_base_path),
            "ior_entries": 50,
        }
        with pytest.raises(ValueError, match="range"):
            Hf3fsL2AdapterConfig.from_dict(config_dict)

        config_dict["ior_entries"] = 2000
        with pytest.raises(ValueError, match="range"):
            Hf3fsL2AdapterConfig.from_dict(config_dict)

    def test_invalid_io_depth_range(self, hf3fs_test_base_path):
        """Test validation fails for io_depth outside [-128, 128]."""
        config_dict = {
            "mount_point": str(hf3fs_test_base_path.parent),
            "base_paths": str(hf3fs_test_base_path),
            "io_depth": -200,
        }
        with pytest.raises(ValueError, match="range"):
            Hf3fsL2AdapterConfig.from_dict(config_dict)

        config_dict["io_depth"] = 200
        with pytest.raises(ValueError, match="range"):
            Hf3fsL2AdapterConfig.from_dict(config_dict)

    def test_invalid_iov_size_range(self, hf3fs_test_base_path):
        """Test validation fails for iov_size outside [100MB, 2GB]."""
        config_dict = {
            "mount_point": str(hf3fs_test_base_path.parent),
            "base_paths": str(hf3fs_test_base_path),
            "iov_size": 50000000,  # 50MB
        }
        with pytest.raises(ValueError, match="range"):
            Hf3fsL2AdapterConfig.from_dict(config_dict)

        config_dict["iov_size"] = 3000000000  # 3GB
        with pytest.raises(ValueError, match="range"):
            Hf3fsL2AdapterConfig.from_dict(config_dict)

    def test_invalid_num_workers(self, hf3fs_test_base_path):
        """Test validation fails for invalid num_workers."""
        config_dict = {
            "mount_point": str(hf3fs_test_base_path.parent),
            "base_paths": str(hf3fs_test_base_path),
            "num_workers": 0,
        }
        with pytest.raises(ValueError, match="positive"):
            Hf3fsL2AdapterConfig.from_dict(config_dict)

    def test_multi_path_config(self, hf3fs_test_multi_base_paths):
        """Test multi-path configuration."""
        path1, path2 = hf3fs_test_multi_base_paths
        config_dict = {
            "mount_point": str(path1.parent),
            "base_paths": f"{path1},{path2}",
        }
        config = Hf3fsL2AdapterConfig.from_dict(config_dict)
        assert config.base_paths == f"{path1},{path2}"

    def test_help_method(self):
        """Test help method returns documentation."""
        help_text = Hf3fsL2AdapterConfig.help()
        assert "mount_point" in help_text
        assert "base_paths" in help_text
        assert "num_workers" in help_text
        assert "ior_entries" in help_text
        assert "io_depth" in help_text
        assert "numa_id" in help_text
        assert "iov_size" in help_text


# =============================================================================
# Per-Op Workers Config Tests (no C++ extension needed)
# =============================================================================


class TestHf3fsPerOpWorkersConfig:
    """Unit tests for per_op_workers field in Hf3fsL2AdapterConfig."""

    def _make_config_dict(
        self,
        mount_point: Path,
        base_path: Path,
        **overrides: object,
    ) -> dict:
        """Build a base config dict with valid paths and apply overrides."""
        d = {
            "mount_point": str(mount_point),
            "base_paths": str(base_path),
            "num_workers": 4,
        }
        d.update(overrides)
        return d

    def test_default_per_op_workers_is_none(self, tmp_path: Path) -> None:
        """When per_op_workers is absent from dict, config should have None."""
        mp = tmp_path / "mp"
        mp.mkdir()
        bp = mp / "bp"
        bp.mkdir()
        config = Hf3fsL2AdapterConfig.from_dict(
            {
                "mount_point": str(mp),
                "base_paths": str(bp),
                "num_workers": 4,
            }
        )
        assert config.per_op_workers is None

    def test_per_op_workers_with_valid_dict(self, tmp_path: Path) -> None:
        """per_op_workers dict is stored as-is when all values are valid."""
        mp = tmp_path / "mp"
        mp.mkdir()
        bp = mp / "bp"
        bp.mkdir()
        config = Hf3fsL2AdapterConfig.from_dict(
            {
                "mount_point": str(mp),
                "base_paths": str(bp),
                "num_workers": 4,
                "per_op_workers": {"retrieve": 4, "store": 2},
            }
        )
        assert config.per_op_workers == {"retrieve": 4, "store": 2}

    def test_per_op_workers_with_all_lane_keys(self, tmp_path: Path) -> None:
        """All four lane keys are accepted."""
        mp = tmp_path / "mp"
        mp.mkdir()
        bp = mp / "bp"
        bp.mkdir()
        config = Hf3fsL2AdapterConfig.from_dict(
            {
                "mount_point": str(mp),
                "base_paths": str(bp),
                "num_workers": 4,
                "per_op_workers": {
                    "lookup": 2,
                    "retrieve": 4,
                    "store": 4,
                    "delete": 1,
                },
            }
        )
        assert config.per_op_workers == {
            "lookup": 2,
            "retrieve": 4,
            "store": 4,
            "delete": 1,
        }

    def test_per_op_workers_negative_raises_value_error(self, tmp_path: Path) -> None:
        """A negative worker count raises ValueError."""
        mp = tmp_path / "mp"
        mp.mkdir()
        bp = mp / "bp"
        bp.mkdir()
        with pytest.raises(ValueError, match="per_op_workers\\["):
            Hf3fsL2AdapterConfig.from_dict(
                {
                    "mount_point": str(mp),
                    "base_paths": str(bp),
                    "num_workers": 4,
                    "per_op_workers": {"retrieve": -1},
                }
            )

    def test_help_includes_per_op_workers(
        self,
    ) -> None:
        """help() string documents the new field."""
        help_text = Hf3fsL2AdapterConfig.help()
        assert "per_op_workers" in help_text
        assert "retrieve" in help_text
        assert "store" in help_text


# =============================================================================
# Key Buffer Config Tests (no C++ extension)
# =============================================================================


class TestHf3fsKeyBufferConfig:
    """Tests for Hf3fsL2AdapterConfig.enable_key_buffer."""

    def _make_hf3fs_config(
        self,
        tmp_path: Any,
        enable_key_buffer: bool | None = None,
    ) -> Hf3fsL2AdapterConfig:
        """Create an Hf3fsL2AdapterConfig with standard test defaults."""
        bp = tmp_path / "path1"
        bp.mkdir()
        kwargs: dict[str, Any] = {
            "mount_point": str(tmp_path),
            "base_paths": str(bp),
            "num_workers": 4,
        }
        if enable_key_buffer is not None:
            kwargs["enable_key_buffer"] = enable_key_buffer
        return Hf3fsL2AdapterConfig(**kwargs)

    def test_default_is_true(self, tmp_path: Any) -> None:
        """Default enable_key_buffer is True."""
        config = self._make_hf3fs_config(tmp_path)
        assert config.enable_key_buffer is True

    def test_explicitly_disabled(self, tmp_path: Any) -> None:
        """Setting enable_key_buffer=False is stored as False."""
        config = self._make_hf3fs_config(tmp_path, enable_key_buffer=False)
        assert config.enable_key_buffer is False

    def test_from_dict_default(self, tmp_path: Any) -> None:
        """from_dict with no enable_key_buffer key defaults to True."""
        bp = tmp_path / "path1"
        bp.mkdir()
        d: dict[str, Any] = {
            "type": "hf3fs",
            "mount_point": str(tmp_path),
            "base_paths": str(bp),
            "num_workers": 4,
            "ior_entries": 256,
            "io_depth": 0,
            "numa_id": -1,
            "iov_size": 209715200,
        }
        config = Hf3fsL2AdapterConfig.from_dict(d)
        assert config.enable_key_buffer is True

    def test_from_dict_invalid_type(self, tmp_path: Any) -> None:
        """from_dict rejects non-boolean enable_key_buffer."""
        bp = tmp_path / "path1"
        bp.mkdir()
        d: dict[str, Any] = {
            "type": "hf3fs",
            "mount_point": str(tmp_path),
            "base_paths": str(bp),
            "num_workers": 4,
            "ior_entries": 256,
            "io_depth": 0,
            "numa_id": -1,
            "iov_size": 209715200,
            "enable_key_buffer": "yes",
        }
        with pytest.raises(ValueError, match="boolean"):
            Hf3fsL2AdapterConfig.from_dict(d)


# =============================================================================
# Factory Tests
# =============================================================================


class TestHf3fsAdapterFactory:
    """Tests for Hf3fs adapter factory behavior."""

    @requires_hf3fs_cluster
    def test_adapter_creation(self, hf3fs_test_mount_point, hf3fs_test_base_path):
        """Test adapter can be created with LMCacheHf3fsClient."""
        # First Party
        from lmcache.lmcache_hf3fs import LMCacheHf3fsClient
        from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
            NativeConnectorL2Adapter,
        )

        client = LMCacheHf3fsClient(
            mount_point=str(hf3fs_test_mount_point),
            base_paths=str(hf3fs_test_base_path),
            num_workers=2,
            ior_entries=128,
            io_depth=0,
            numa_id=-1,
            iov_size=209715200,
            time_out=200,
            enable_key_buffer=True,
        )
        assert client is not None
        try:
            adapter = NativeConnectorL2Adapter(
                client,
                type_name="Hf3fsL2Adapter",
            )
            assert adapter is not None
        finally:
            adapter.close()
            client.close()

    def test_adapter_creation_no_extension(
        self, hf3fs_test_mount_point, hf3fs_test_base_path
    ):
        """Test adapter creation fails when C++ extension is not available."""
        if _has_hf3fs_extension():
            pytest.skip("C++ Hf3fs extension is available")

        config = Hf3fsL2AdapterConfig.from_dict(
            {
                "mount_point": str(hf3fs_test_mount_point),
                "base_paths": str(hf3fs_test_base_path),
                "num_workers": 2,
            }
        )

        # First Party
        from lmcache.v1.distributed.l2_adapters.hf3fs_l2_adapter import (
            _create_hf3fs_l2_adapter,
        )

        with pytest.raises(RuntimeError, match="requires the C\\+\\+ extension"):
            _create_hf3fs_l2_adapter(config)


# =============================================================================
# Integration Tests (require C++ extension + 3FS cluster)
# =============================================================================


@requires_hf3fs_cluster
class TestHf3fsNativeConnector:
    """Test C++ native 3FS connector I/O operations.

    These tests require:
    1. The C++ Hf3fs extension (lmcache_hf3fs) to be built
    2. A running 3FS cluster (HF3FS_MOUNT_POINT + 3fs-virt subdirectory)
    """

    def test_client_creation(self, hf3fs_native_client):
        """Test LMCacheHf3fsClient creation and basic parameters."""
        client = hf3fs_native_client[0]
        assert client is not None
        # Verify the client can be closed without error
        client.close()

    def test_single_put_get_roundtrip(self, hf3fs_native_client):
        """Test single key write-read cycle with data verification."""
        client = hf3fs_native_client[0]
        key = _object_key_to_string(create_object_key(100))
        memory_obj = create_memory_obj(1024)
        data = bytes(memory_obj.byte_array)

        # Put
        client.submit_batch_set([key], [memoryview(data)])
        _wait_for_completions(client)

        # Get
        buf = bytearray(len(data))
        client.submit_batch_get([key], [memoryview(buf)])
        _wait_for_completions(client)

        assert bytes(buf) == data

    def test_batch_put_get(self, hf3fs_native_client):
        """Test multi-key batch operations."""
        client = hf3fs_native_client[0]
        keys = [_object_key_to_string(create_object_key(i)) for i in range(5)]
        memory_objs = [create_memory_obj(1024) for _ in keys]
        data_list = [bytes(obj.byte_array) for obj in memory_objs]
        memviews = [memoryview(d) for d in data_list]

        # Batch put
        client.submit_batch_set(keys, memviews)
        _wait_for_completions(client)

        # Batch get
        buffers = [bytearray(len(d)) for d in data_list]
        client.submit_batch_get(keys, [memoryview(b) for b in buffers])
        _wait_for_completions(client)

        for i, (expected, buf) in enumerate(zip(data_list, buffers, strict=False)):
            assert bytes(buf) == expected, f"Key {i} mismatch"

    def test_exists_check(self, hf3fs_native_client):
        """Test exists() for existing and non-existing keys."""
        client = hf3fs_native_client[0]
        existing_key = _object_key_to_string(create_object_key(200))
        non_existing_key = _object_key_to_string(create_object_key(999))

        # Check non-existing key
        client.submit_batch_exists([non_existing_key])
        completions = _wait_for_completions(client)
        assert len(completions) == 1
        assert completions[0][3] == [False]

        # Store and check existing key
        memory_obj = create_memory_obj(512)
        data = bytes(memory_obj.byte_array)
        client.submit_batch_set([existing_key], [memoryview(data)])
        _wait_for_completions(client)

        client.submit_batch_exists([existing_key])
        completions = _wait_for_completions(client)
        assert len(completions) == 1
        assert completions[0][3] == [True]

    def test_delete_key(self, hf3fs_native_client):
        """Test write then delete, verify key no longer exists."""
        client = hf3fs_native_client[0]
        key = _object_key_to_string(create_object_key(300))
        memory_obj = create_memory_obj(256)
        data = bytes(memory_obj.byte_array)

        # Store
        client.submit_batch_set([key], [memoryview(data)])
        _wait_for_completions(client)

        # Verify exists
        client.submit_batch_exists([key])
        completions = _wait_for_completions(client)
        assert completions[0][3] == [True]

        # Delete
        if hasattr(client, "submit_batch_delete"):
            client.submit_batch_delete([key])
            _wait_for_completions(client)

        # Verify deleted
        client.submit_batch_exists([key])
        completions = _wait_for_completions(client)
        assert completions[0][3] == [False]

    def test_multi_path_distribution(self):
        """Test that data is distributed across multiple base paths."""
        # First Party
        from lmcache.lmcache_hf3fs import LMCacheHf3fsClient

        # Create two subdirectories under the 3FS mount point
        mp = Path(_HF3FS_MOUNT_POINT)
        path1 = mp / f"test_multi_{uuid.uuid4().hex[:8]}_a"
        path2 = mp / f"test_multi_{uuid.uuid4().hex[:8]}_b"
        path1.mkdir(parents=True, exist_ok=True)
        path2.mkdir(parents=True, exist_ok=True)
        try:
            client = LMCacheHf3fsClient(
                mount_point=str(mp),
                base_paths=f"{path1},{path2}",
                num_workers=2,
                ior_entries=128,
                io_depth=0,
                numa_id=-1,
                iov_size=209715200,
                time_out=200,
                enable_key_buffer=True,
            )

            # Store keys with different hashes
            keys = [_object_key_to_string(create_object_key(i)) for i in range(10)]
            memory_objs = [create_memory_obj(256) for _ in keys]
            data_list = [bytes(obj.byte_array) for obj in memory_objs]
            client.submit_batch_set(keys, [memoryview(d) for d in data_list])
            _wait_for_completions(client)

            # Verify all keys exist
            client.submit_batch_exists(keys)
            completions = _wait_for_completions(client)
            assert all(completions[0][3])

            client.close()
        finally:
            shutil.rmtree(path1, ignore_errors=True)
            shutil.rmtree(path2, ignore_errors=True)

    def test_persistence_across_clients(self, hf3fs_native_client):
        """Test that data persists after client close and can be read by new client."""
        client, mount_point, base_paths = hf3fs_native_client
        key = _object_key_to_string(create_object_key(400))
        memory_obj = create_memory_obj(512)
        data = bytes(memory_obj.byte_array)

        # Store with first client
        client.submit_batch_set([key], [memoryview(data)])
        _wait_for_completions(client)

        # Close first client
        client.close()

        # Create new client and read using the same mount_point and base_paths
        # First Party
        from lmcache.lmcache_hf3fs import LMCacheHf3fsClient

        new_client = LMCacheHf3fsClient(
            mount_point=mount_point,
            base_paths=base_paths,
            num_workers=2,
            ior_entries=128,
            io_depth=0,
            numa_id=-1,
            iov_size=209715200,
            time_out=200,
            enable_key_buffer=True,
        )

        # Read
        buf = bytearray(len(data))
        new_client.submit_batch_get([key], [memoryview(buf)])
        _wait_for_completions(new_client)

        assert bytes(buf) == data
        new_client.close()


@requires_hf3fs_cluster
class TestHf3fsL2AdapterIntegration:
    """Test Hf3fs L2 adapter end-to-end operations with 3FS cluster.

    These tests require:
    1. The C++ Hf3fs extension (lmcache_hf3fs) to be built
    2. A running 3FS cluster (HF3FS_MOUNT_POINT + 3fs-virt subdirectory)
    """

    def test_store_and_lookup(self, hf3fs_l2_adapter):
        """Test store and lookup operations."""
        keys = [create_object_key(1), create_object_key(2)]
        memory_objs = [create_memory_obj(1024) for _ in keys]

        task_id = hf3fs_l2_adapter.submit_store_task(keys, memory_objs)
        # Wait for the background demux thread to process completions
        completed = _wait_for_store(hf3fs_l2_adapter)
        assert task_id in completed

        # Lookup - submit and wait for eventfd signal
        lookup_task_id = hf3fs_l2_adapter.submit_lookup_and_lock_task(
            keys, {0: _EMPTY_LAYOUT}
        )
        _wait_for_lookup(hf3fs_l2_adapter)
        result = hf3fs_l2_adapter.query_lookup_and_lock_result(lookup_task_id)
        assert result is not None

    def test_load_data(self, hf3fs_l2_adapter):
        """Test load operation."""
        key = create_object_key(10)
        memory_obj = create_memory_obj(1024)

        # First store some data
        _ = hf3fs_l2_adapter.submit_store_task([key], [memory_obj])
        _wait_for_store(hf3fs_l2_adapter)

        # Create a buffer for loading
        data_size = 1024
        load_memory_obj = create_memory_obj(data_size)
        load_task_id = hf3fs_l2_adapter.submit_load_task([key], [load_memory_obj])
        _wait_for_load(hf3fs_l2_adapter)
        result = hf3fs_l2_adapter.query_load_result(load_task_id)
        assert result is not None

    def test_delete_keys(self, hf3fs_l2_adapter):
        """Test batch delete operation."""
        keys = [create_object_key(20), create_object_key(21)]

        # Delete should not raise even if keys don't exist
        hf3fs_l2_adapter.delete(keys)

    def test_end_to_end_workflow(self, hf3fs_l2_adapter):
        """Test full store -> lookup -> load -> unlock workflow."""
        key = create_object_key(30)
        memory_obj = create_memory_obj(1024)

        # Store
        store_task_id = hf3fs_l2_adapter.submit_store_task([key], [memory_obj])
        completed = _wait_for_store(hf3fs_l2_adapter)
        assert store_task_id in completed

        # Lookup
        lookup_task_id = hf3fs_l2_adapter.submit_lookup_and_lock_task(
            [key], {0: _EMPTY_LAYOUT}
        )
        _wait_for_lookup(hf3fs_l2_adapter)
        result = hf3fs_l2_adapter.query_lookup_and_lock_result(lookup_task_id)
        assert result is not None

        # Unlock
        hf3fs_l2_adapter.submit_unlock([key])

    def test_multi_object_batch(self, hf3fs_l2_adapter):
        """Test multi-object batch operations."""
        keys = [create_object_key(i) for i in range(5)]
        memory_objs = [create_memory_obj(1024) for _ in keys]

        # Batch store
        store_task_id = hf3fs_l2_adapter.submit_store_task(keys, memory_objs)
        completed = _wait_for_store(hf3fs_l2_adapter)
        assert store_task_id in completed

        # Batch lookup
        lookup_task_id = hf3fs_l2_adapter.submit_lookup_and_lock_task(
            keys, {0: _EMPTY_LAYOUT}
        )
        _wait_for_lookup(hf3fs_l2_adapter)
        result = hf3fs_l2_adapter.query_lookup_and_lock_result(lookup_task_id)
        assert result is not None

        # Batch delete
        hf3fs_l2_adapter.delete(keys)
