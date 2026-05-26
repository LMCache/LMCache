# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import AsyncMock, MagicMock
import asyncio
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector.bigtable_connector import (
    BigtableConnector,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend
from tests.v1.utils import create_test_memory_obj


class MockDeadlineExceeded(Exception):
    pass


class MockPermissionDenied(Exception):
    pass


class MockNotFound(Exception):
    pass


class MockResourceExhausted(Exception):
    pass


# Setup mock google packages before any test runs
mock_exceptions = MagicMock()
mock_exceptions.DeadlineExceeded = MockDeadlineExceeded
mock_exceptions.Timeout = MockDeadlineExceeded
mock_exceptions.PermissionDenied = MockPermissionDenied
mock_exceptions.Unauthenticated = MockPermissionDenied
mock_exceptions.NotFound = MockNotFound
mock_exceptions.ResourceExhausted = MockResourceExhausted

mock_row_filters = MagicMock()
mock_row_filters.StripValueTransformerFilter.return_value = MagicMock()

mock_data = MagicMock()
mock_data.BigtableDataClientAsync = MagicMock()
mock_data.ReadRowsQuery = MagicMock()
mock_data.RowMutationEntry = MagicMock()
mock_data.row_filters = mock_row_filters

mock_bigtable = MagicMock(data=mock_data, row_filters=mock_row_filters)

mock_service_account = MagicMock()
mock_oauth2 = MagicMock(service_account=mock_service_account)

sys.modules["google.api_core"] = MagicMock(exceptions=mock_exceptions)
sys.modules["google.api_core.exceptions"] = mock_exceptions
sys.modules["google.cloud"] = MagicMock(bigtable=mock_bigtable)
sys.modules["google.cloud.bigtable"] = mock_bigtable
sys.modules["google.cloud.bigtable.row_filters"] = mock_row_filters
sys.modules["google.cloud.bigtable.data"] = mock_data
sys.modules["google.cloud.bigtable.data.row_filters"] = mock_row_filters
sys.modules["google.oauth2"] = mock_oauth2
sys.modules["google.oauth2.service_account"] = mock_service_account


async def mock_async_gen(items):
    for item in items:
        yield item


def create_test_config(extra_overrides=None):
    extras = {
        "bigtable_project_id": "test-project",
        "bigtable_instance_id": "test-instance",
        "bigtable_table_name": "test-table",
        "bigtable_max_chunk_size_mb": 90.0,
        "bigtable_max_retries": 2,
        "bigtable_exists_cache_ttl_seconds": 10.0,
    }
    if extra_overrides:
        extras.update(extra_overrides)

    return LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        remote_storage_plugins=["bigtable"],
        remote_serde="naive",
        lmcache_instance_id="test_instance",
        extra_config=extras,
    )


def create_test_metadata():
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
    return CacheEngineKey(
        model_name="test_model",
        world_size=3,
        worker_id=1,
        chunk_hash=hash(key_id),
        dtype=torch.bfloat16,
    )


@pytest.fixture
def async_loop():
    loop = asyncio.new_event_loop()
    # Standard
    import threading

    # First Party
    from lmcache.utils import start_loop_in_thread_with_exceptions

    thread = threading.Thread(
        target=start_loop_in_thread_with_exceptions,
        args=(loop,),
        name="test-async-loop",
    )
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5.0)


@pytest.fixture
def local_cpu_backend(memory_allocator):
    config = LMCacheEngineConfig.from_legacy(chunk_size=256)
    metadata = create_test_metadata()
    return LocalCPUBackend(config, metadata, memory_allocator=memory_allocator)


class TestBigtableConnector:
    def test_init_and_lazy_pool(self, async_loop, local_cpu_backend):
        """Verify independent lazy async client pool initialization."""
        config = create_test_config()
        metadata = create_test_metadata()

        mock_client_instance = MagicMock()
        mock_client_instance.read_row = AsyncMock(return_value=None)
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        connector = (
            backend.connection._connector
            if hasattr(backend.connection, "_connector")
            else backend.connection
        )
        assert isinstance(connector, BigtableConnector)
        assert connector.project_id == "test-project"
        assert connector.instance_id == "test-instance"
        assert connector.table_name == "test-table"

        # Client pool should not be initialized yet (lazy)
        assert connector._client is None

        # Trigger first operation
        key = create_test_key(1)

        assert not backend.contains(key)
        # Pool now initialized
        assert connector._client is not None
        mock_data.BigtableDataClientAsync.assert_called_once_with(
            project="test-project"
        )

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_out_of_tree_plugin_init(self, async_loop, local_cpu_backend):
        """Verify standalone out-of-tree plugin initialization via
        class_name/module_path.
        """
        config = create_test_config(
            {
                "remote_storage_plugin.bigtable.module_path": "lmc_bigtable_connector",
                "remote_storage_plugin.bigtable.class_name": "BigtableConnector",
                "bigtable_project_id": "ext-project",
                "bigtable_instance_id": "ext-instance",
                "bigtable_table_name": "ext-table",
            }
        )
        metadata = create_test_metadata()

        mock_plugin_module = MagicMock()
        mock_plugin_module.BigtableConnector = BigtableConnector
        sys.modules["lmc_bigtable_connector"] = mock_plugin_module

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        connector = (
            backend.connection._connector
            if hasattr(backend.connection, "_connector")
            else backend.connection
        )
        assert connector.__class__.__name__ == "BigtableConnector"
        assert connector.cfg.project_id == "ext-project"
        assert connector.cfg.instance_id == "ext-instance"
        assert connector.cfg.table_name == "ext-table"

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_credentials_path_init(self, async_loop, local_cpu_backend):
        """Verify passing credentials_path initializes client with
        explicit service account.
        """
        config = create_test_config(
            {"bigtable_credentials_path": "/path/to/creds.json"}
        )
        metadata = create_test_metadata()

        mock_creds = MagicMock()
        mock_service_account.Credentials.from_service_account_file.return_value = (
            mock_creds
        )

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        connector = backend.connection
        connector._get_client()

        mock_service_account.Credentials.from_service_account_file.assert_called_once_with(
            "/path/to/creds.json"
        )
        mock_data.BigtableDataClientAsync.assert_called_with(
            project="test-project", credentials=mock_creds
        )

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_custom_row_key_template(self, async_loop, local_cpu_backend):
        """Verify substituting {model} and {hash} placeholders flexibly."""
        config = create_test_config({"bigtable_row_key_template": "{model}#{hash}"})
        metadata = create_test_metadata()
        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key = create_test_key(123)
        row_key = backend.connection._get_row_key(key)
        row_key_str = row_key.decode("utf-8")

        assert row_key_str.startswith("test_model@3@1@bfloat16#")
        assert row_key_str.endswith(key.chunk_hash_hex)

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_local_ttl_cache_hits(self, async_loop, local_cpu_backend):
        """Verify that exists calls accelerate via local TTL cache
        without hitting Bigtable.
        """
        config = create_test_config()
        metadata = create_test_metadata()

        mock_client_instance = MagicMock()
        mock_client_instance.read_row = AsyncMock(return_value=MagicMock())
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key = create_test_key(1)

        assert backend.contains(key)
        assert mock_client_instance.read_row.call_count == 1

        assert backend.contains(key)
        assert mock_client_instance.read_row.call_count == 1

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_get_blocking_hit(self, async_loop, local_cpu_backend):
        config = create_test_config()
        metadata = create_test_metadata()

        memory_obj = create_test_memory_obj()
        expected_bytes = bytes(memory_obj.byte_array)

        mock_cell = MagicMock(value=expected_bytes)
        mock_row = MagicMock()
        mock_row.cells = {"cf": {b"data": [mock_cell]}}

        mock_client_instance = MagicMock()
        mock_client_instance.read_row = AsyncMock(return_value=mock_row)
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key = create_test_key(1)

        res = backend.get_blocking(key)
        assert res is not None
        assert isinstance(res, MemoryObj)
        assert bytes(res.byte_array)[: len(expected_bytes)] == expected_bytes

        assert backend.connection.exists_cache.get(key.to_string()) is True

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_cell_size_validation_skips_large_writes(
        self, async_loop, local_cpu_backend
    ):
        """Validate chunk size at write time. Skips > 90MB rather than failing."""
        config = create_test_config()
        metadata = create_test_metadata()

        mock_client_instance = MagicMock()
        mock_client_instance.mutate_row = AsyncMock()
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key = create_test_key(1)
        large_view = bytearray(95 * 1024 * 1024)
        mock_memory_obj = MagicMock()
        mock_memory_obj.byte_array = large_view

        asyncio.run_coroutine_threadsafe(
            backend.connection._put_internal(key, mock_memory_obj),
            async_loop,
        ).result()

        mock_client_instance.mutate_row.assert_not_called()

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_error_handling_timeouts_as_miss(self, async_loop, local_cpu_backend):
        """TimeoutError / DeadlineExceeded treated as cache miss and fall through."""
        config = create_test_config()
        metadata = create_test_metadata()

        mock_client_instance = MagicMock()
        mock_client_instance.read_row = AsyncMock(
            side_effect=MockDeadlineExceeded("timeout")
        )
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key = create_test_key(1)

        assert not backend.contains(key)

        res = backend.get_blocking(key)
        assert res is None

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_error_handling_auth_raises(self, async_loop, local_cpu_backend):
        """PermissionDenied / Unauthenticated propagate up."""
        config = create_test_config()
        metadata = create_test_metadata()

        mock_client_instance = MagicMock()
        mock_client_instance.read_row = AsyncMock(
            side_effect=MockPermissionDenied("auth error")
        )
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key = create_test_key(1)

        with pytest.raises(MockPermissionDenied):
            backend.contains(key)

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_batched_get(self, async_loop, local_cpu_backend):
        """Verify batched_get retrieves multiple memory objects cleanly."""
        config = create_test_config()
        metadata = create_test_metadata()

        memory_obj1 = create_test_memory_obj()
        memory_obj2 = create_test_memory_obj()
        bytes1 = bytes(memory_obj1.byte_array)
        bytes2 = bytes(memory_obj2.byte_array)

        mock_cell1 = MagicMock(value=bytes1)
        mock_cell2 = MagicMock(value=bytes2)
        mock_row1 = MagicMock(row_key=b"hash1#fingerprint")
        mock_row2 = MagicMock(row_key=b"hash2#fingerprint")
        mock_row1.cells = {"cf": {b"data": [mock_cell1]}}
        mock_row2.cells = {"cf": {b"data": [mock_cell2]}}

        mock_client_instance = MagicMock()
        mock_client_instance.read_rows = AsyncMock(return_value=[mock_row1, mock_row2])
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key1 = create_test_key(1)
        key2 = create_test_key(2)

        res = asyncio.run_coroutine_threadsafe(
            backend.connection.batched_get([key1, key2]),
            async_loop,
        ).result()

        assert len(res) == 2
        assert res[0] is not None and res[1] is not None
        assert bytes(res[0].byte_array)[: len(bytes1)] == bytes1
        assert bytes(res[1].byte_array)[: len(bytes2)] == bytes2

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_batched_put(self, async_loop, local_cpu_backend):
        """Verify batched_put packs mutations and sends bulk mutate rows cleanly."""
        config = create_test_config()
        metadata = create_test_metadata()

        mock_client_instance = MagicMock()
        mock_client_instance.bulk_mutate_rows = AsyncMock()
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key1 = create_test_key(1)
        key2 = create_test_key(2)
        memory_obj1 = create_test_memory_obj()
        memory_obj2 = create_test_memory_obj()

        asyncio.run_coroutine_threadsafe(
            backend.connection.batched_put([key1, key2], [memory_obj1, memory_obj2]),
            async_loop,
        ).result()

        mock_client_instance.bulk_mutate_rows.assert_called_once()

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_batched_async_contains(self, async_loop, local_cpu_backend):
        """Verify batched_async_contains returns correct match counts."""
        config = create_test_config()
        metadata = create_test_metadata()

        mock_row = MagicMock(row_key=b"hash1#fingerprint")
        mock_client_instance = MagicMock()
        mock_client_instance.read_rows = AsyncMock(return_value=[mock_row])
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key1 = create_test_key(1)
        key2 = create_test_key(2)

        res = asyncio.run_coroutine_threadsafe(
            backend.connection.batched_async_contains("test", [key1, key2]),
            async_loop,
        ).result()

        # Only key1 rowkey starts with matched row_key
        # from read_rows, so match count is 1
        assert res == 1

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_remove_sync(self, async_loop, local_cpu_backend):
        """Verify remove_sync issues DeleteAllFromRow mutation cleanly."""
        config = create_test_config()
        metadata = create_test_metadata()

        mock_client_instance = MagicMock()
        mock_client_instance.mutate_row = AsyncMock()
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key = create_test_key(1)
        res = backend.connection.remove_sync(key)

        assert res is True
        mock_client_instance.mutate_row.assert_called_once()

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_list(self, async_loop, local_cpu_backend):
        """Verify list() returns parsed standardized LMCache key metadata formats."""
        config = create_test_config()
        metadata = create_test_metadata()

        mock_row1 = MagicMock(row_key=b"hash1#test_model@3@1@bfloat16")
        mock_row2 = MagicMock(row_key=b"hash2#test_model@3@1@bfloat16")

        mock_client_instance = MagicMock()
        mock_client_instance.read_rows = AsyncMock(return_value=[mock_row1, mock_row2])
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        res = asyncio.run_coroutine_threadsafe(
            backend.connection.list(),
            async_loop,
        ).result()

        assert len(res) == 2
        assert "test_model@3@1@hash1@bfloat16" in res
        assert "test_model@3@1@hash2@bfloat16" in res

        backend.close()
        local_cpu_backend.memory_allocator.close()

    def test_ping(self, async_loop, local_cpu_backend):
        """Verify ping executes SELECT 1; health queries cleanly."""
        config = create_test_config()
        metadata = create_test_metadata()

        async def mock_query_iter():
            yield MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.execute_query = AsyncMock(return_value=mock_query_iter())
        mock_data.BigtableDataClientAsync.return_value = mock_client_instance

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        res = asyncio.run_coroutine_threadsafe(
            backend.connection.ping(),
            async_loop,
        ).result()

        assert res == 0
        mock_client_instance.execute_query.assert_called_once()

        backend.close()
        local_cpu_backend.memory_allocator.close()
