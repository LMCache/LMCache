# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from unittest.mock import MagicMock
import asyncio

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector import ConnectorContext
from lmcache.v1.storage_backend.connector.gcs_adapter import GCSConnectorAdapter
from lmcache.v1.storage_backend.connector.gcs_connector import (
    GCSConnector,
    GCSConnectorConfig,
    decode_gcs_object_name,
    encode_gcs_object_name,
    parse_gcs_bucket_handle,
    resolve_gcs_connector_config,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend


@dataclass(frozen=True)
class FakeBlobEntry:
    """Small stand-in for GCS blob metadata."""

    name: str
    size: int


class FakeGCSClient:
    """In-memory GCS client used to unit test the connector."""

    def __init__(
        self,
        *,
        project: str | None = None,
        credentials_path: str | None = None,
    ) -> None:
        del project
        del credentials_path
        self.storage: dict[str, bytes] = {}
        self.ensure_bucket_calls: list[str] = []
        self.bucket_exists_value = True
        self.deleted_blob_names: list[str] = []

    def ensure_bucket(self, bucket_name: str) -> None:
        """Record bucket-ensure requests."""
        self.ensure_bucket_calls.append(bucket_name)
        self.bucket_exists_value = True

    def bucket_exists(self, bucket_name: str) -> bool:
        """Return whether the bucket exists."""
        del bucket_name
        return self.bucket_exists_value

    def get_blob_size(self, bucket_name: str, blob_name: str) -> int:
        """Return the stored size or ``0`` for missing blobs."""
        del bucket_name
        payload = self.storage.get(blob_name)
        return 0 if payload is None else len(payload)

    def list_blobs(self, bucket_name: str, prefix: str) -> list[str]:
        """List stored blob names under the requested prefix."""
        del bucket_name
        prefix_with_separator = f"{prefix}/" if prefix else ""
        return [
            blob_name
            for blob_name in sorted(self.storage)
            if not prefix
            or blob_name == prefix
            or blob_name.startswith(prefix_with_separator)
        ]

    def upload_blob(self, bucket_name: str, blob_name: str, payload: bytes) -> None:
        """Store uploaded bytes under their blob names."""
        del bucket_name
        self.storage[blob_name] = payload

    def download_blob(self, bucket_name: str, blob_name: str) -> bytes | None:
        """Return stored blob bytes."""
        del bucket_name
        return self.storage.get(blob_name)

    def delete_blob(self, bucket_name: str, blob_name: str) -> bool:
        """Remove stored objects."""
        del bucket_name
        self.deleted_blob_names.append(blob_name)
        self.storage.pop(blob_name, None)
        return True


def create_test_metadata() -> LMCacheMetadata:
    """Create LMCache metadata with a deterministic full chunk layout."""
    return LMCacheMetadata(
        model_name="test-model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 256, 8, 128),
    )


def create_test_config(
    extra_config: dict[str, object] | None = None,
    *,
    plugin_name: str = "gcs",
    save_unfull_chunk: bool = False,
) -> LMCacheEngineConfig:
    """Create a plugin-based config for the GCS connector."""
    return LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=0.1,
        save_unfull_chunk=save_unfull_chunk,
        lmcache_instance_id="test-gcs",
        remote_storage_plugins=[plugin_name],
        extra_config=extra_config or {},
    )


def create_test_key(key_id: int) -> CacheEngineKey:
    """Create a deterministic cache key for unit tests."""
    return CacheEngineKey(
        model_name="test/model",
        world_size=1,
        worker_id=0,
        chunk_hash=key_id,
        dtype=torch.bfloat16,
    )


def create_connector(
    memory_allocator,
    *,
    plugin_name: str = "gcs",
    extra_config: dict[str, object] | None = None,
    save_unfull_chunk: bool = False,
    gcs_client: FakeGCSClient | None = None,
) -> tuple[GCSConnector, FakeGCSClient, LMCacheMetadata]:
    """Create a connector with an in-memory fake GCS client."""
    metadata = create_test_metadata()
    config_dict = {
        f"remote_storage_plugin.{plugin_name}.bucket_handle": "gs://test-bucket/prod",
        f"remote_storage_plugin.{plugin_name}.metadata_cache_ttl_secs": 30,
    }
    if extra_config is not None:
        config_dict.update(extra_config)

    config = create_test_config(
        config_dict,
        plugin_name=plugin_name,
        save_unfull_chunk=save_unfull_chunk,
    )
    local_cpu_backend = LocalCPUBackend(
        config,
        metadata,
        memory_allocator=memory_allocator,
    )
    client = gcs_client or FakeGCSClient()
    connector = GCSConnector(
        local_cpu_backend=local_cpu_backend,
        config=config,
        metadata=metadata,
        connector_config=resolve_gcs_connector_config(config, plugin_name),
        gcs_client=client,
    )
    return connector, client, metadata


def create_full_chunk_memory_obj(
    local_cpu_backend: LocalCPUBackend,
    metadata: LMCacheMetadata,
    fill_byte: int,
) -> tuple[MemoryObj, bytes]:
    """Allocate and initialize a full chunk memory object for upload tests."""
    memory_obj = local_cpu_backend.allocate(
        metadata.get_shapes(),
        metadata.get_dtypes(),
        MemoryFormat.KV_2LTD,
    )
    assert memory_obj is not None
    byte_buffer = memoryview(memory_obj.byte_array).cast("B")
    payload = bytes([fill_byte]) * len(byte_buffer)
    byte_buffer[:] = payload
    return memory_obj, payload


def memory_obj_to_bytes(memory_obj: MemoryObj) -> bytes:
    """Convert a test memory object to raw bytes."""
    return memoryview(memory_obj.byte_array).cast("B").tobytes()


@pytest.fixture
def async_loop():
    """Create an asyncio event loop running in a separate thread for testing."""
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
    """Create a LocalCPUBackend for testing."""
    config = LMCacheEngineConfig.from_legacy(chunk_size=256)
    metadata = create_test_metadata()
    return LocalCPUBackend(config, metadata, memory_allocator=memory_allocator)


@pytest.mark.parametrize(
    ("bucket_handle", "bucket_name", "object_prefix"),
    [
        ("gs://my-bucket", "my-bucket", ""),
        ("gs://my-bucket/prod/checkpoints", "my-bucket", "prod/checkpoints"),
    ],
)
def test_parse_gcs_bucket_handle(
    bucket_handle: str,
    bucket_name: str,
    object_prefix: str,
) -> None:
    """Bucket handles should split into bucket name and object prefix."""
    location = parse_gcs_bucket_handle(bucket_handle)
    assert location.bucket_name == bucket_name
    assert location.object_prefix == object_prefix


def test_gcs_object_name_encoding_is_reversible() -> None:
    """Encoded object names should round-trip for LMCache keys."""
    key_str = "model/name@1/2?x=y"
    assert decode_gcs_object_name(encode_gcs_object_name(key_str)) == key_str


def test_adapter_can_parse_plugin_urls() -> None:
    """The adapter should match plugin URLs with and without instances."""
    adapter = GCSConnectorAdapter()
    assert adapter.can_parse("plugin://gcs")
    assert adapter.can_parse("plugin://gcs.us")
    assert not adapter.can_parse("plugin://fs")


def test_resolve_gcs_connector_config_uses_full_plugin_name() -> None:
    """Instance-specific plugin config should be resolved from the full name."""
    config = create_test_config(
        {
            "remote_storage_plugin.gcs.us.bucket_handle": "gs://us-bucket/prod",
            "remote_storage_plugin.gcs.us.project": "test-project",
            "remote_storage_plugin.gcs.us.credentials_path": "/tmp/gcs-us.json",
            "remote_storage_plugin.gcs.us.create_bucket_if_missing": True,
            "remote_storage_plugin.gcs.us.metadata_cache_ttl_secs": 12,
        },
        plugin_name="gcs.us",
    )

    connector_config = resolve_gcs_connector_config(config, "gcs.us")

    assert connector_config.plugin_name == "gcs.us"
    assert connector_config.bucket_location.bucket_name == "us-bucket"
    assert connector_config.bucket_location.object_prefix == "prod"
    assert connector_config.project == "test-project"
    assert connector_config.credentials_path == "/tmp/gcs-us.json"
    assert connector_config.create_bucket_if_missing is True
    assert connector_config.metadata_cache_ttl_secs == 12.0


def test_adapter_create_connector_uses_plugin_scoped_config(monkeypatch) -> None:
    """Adapter connector creation should pass through the resolved plugin config."""
    created: dict[str, object] = {}

    class DummyConnector:
        """Capture adapter constructor arguments without invoking the real client."""

        def __init__(
            self,
            local_cpu_backend: object,
            config: LMCacheEngineConfig,
            metadata: LMCacheMetadata,
            connector_config: GCSConnectorConfig,
        ) -> None:
            created["local_cpu_backend"] = local_cpu_backend
            created["config"] = config
            created["metadata"] = metadata
            created["connector_config"] = connector_config

    config = create_test_config(
        {
            "remote_storage_plugin.gcs.prod.bucket_handle": "gs://test-bucket/prod",
        },
        plugin_name="gcs.prod",
    )
    metadata = create_test_metadata()
    loop = asyncio.new_event_loop()
    adapter = GCSConnectorAdapter()

    monkeypatch.setattr(
        "lmcache.v1.storage_backend.connector.gcs_adapter.GCSConnector",
        DummyConnector,
    )

    connector = adapter.create_connector(
        ConnectorContext(
            url="plugin://gcs.prod",
            loop=loop,
            local_cpu_backend=None,
            config=config,
            metadata=metadata,
            plugin_name="gcs.prod",
        )
    )

    assert isinstance(connector, DummyConnector)
    connector_config = created["connector_config"]
    assert isinstance(connector_config, GCSConnectorConfig)
    assert connector_config.plugin_name == "gcs.prod"
    assert connector_config.bucket_location.object_prefix == "prod"


def test_put_get_exists_list_and_remove_roundtrip(memory_allocator) -> None:
    """Single-object operations should round-trip against the fake client."""
    connector, fake_client, metadata = create_connector(memory_allocator)
    key = create_test_key(1)
    memory_obj, payload = create_full_chunk_memory_obj(
        connector.local_cpu_backend,
        metadata,
        fill_byte=17,
    )

    try:
        asyncio.run(connector.put(key, memory_obj))
        assert memory_obj.get_ref_count() == 0
        assert fake_client.ensure_bucket_calls == []

        assert connector.exists_sync(key) is True
        assert asyncio.run(connector.exists(key)) is True
        assert asyncio.run(connector.list()) == [key.to_string()]

        loaded = asyncio.run(connector.get(key))
        assert loaded is not None
        try:
            assert memory_obj_to_bytes(loaded) == payload
        finally:
            loaded.ref_count_down()

        assert connector.remove_sync(key) is True
        assert connector.exists_sync(key) is False
    finally:
        asyncio.run(connector.close())
        connector.local_cpu_backend.memory_allocator.close()


def test_batched_put_and_batched_get_preserve_order(memory_allocator) -> None:
    """Batched operations should preserve result order and prefix semantics."""
    connector, fake_client, metadata = create_connector(
        memory_allocator,
        extra_config={
            "remote_storage_plugin.gcs.create_bucket_if_missing": True,
        },
    )
    keys = [create_test_key(10), create_test_key(11)]
    memory_objs_and_payloads = [
        create_full_chunk_memory_obj(
            connector.local_cpu_backend, metadata, fill_byte=33
        ),
        create_full_chunk_memory_obj(
            connector.local_cpu_backend, metadata, fill_byte=44
        ),
    ]
    memory_objs = [item[0] for item in memory_objs_and_payloads]
    payloads = [item[1] for item in memory_objs_and_payloads]

    try:
        asyncio.run(connector.batched_put(keys, memory_objs))
        assert [memory_obj.get_ref_count() for memory_obj in memory_objs] == [0, 0]
        assert fake_client.ensure_bucket_calls == ["test-bucket"]

        missing_key = create_test_key(12)
        results = asyncio.run(connector.batched_get([keys[0], missing_key, keys[1]]))
        assert results[0] is not None
        assert results[1] is None
        assert results[2] is not None
        try:
            assert memory_obj_to_bytes(results[0]) == payloads[0]
            assert memory_obj_to_bytes(results[2]) == payloads[1]
        finally:
            for result in results:
                if result is not None:
                    result.ref_count_down()

        assert connector.batched_contains([keys[0], missing_key, keys[1]]) == 1

        prefix_hits = asyncio.run(
            connector.batched_get_non_blocking("lookup-1", [keys[0], missing_key])
        )
        try:
            assert len(prefix_hits) == 1
            assert memory_obj_to_bytes(prefix_hits[0]) == payloads[0]
        finally:
            for result in prefix_hits:
                result.ref_count_down()
    finally:
        asyncio.run(connector.close())
        connector.local_cpu_backend.memory_allocator.close()


def test_partial_chunk_upload_is_rejected(memory_allocator) -> None:
    """Partial chunks should be rejected by the conservative MVP."""
    connector, _, _ = create_connector(memory_allocator)
    key = create_test_key(20)
    try:
        memory_obj = MagicMock(spec=MemoryObj)
        memory_obj.get_physical_size.return_value = connector.full_chunk_size_bytes - 1
        with pytest.raises(ValueError, match="Partial/unfull chunks are not supported"):
            asyncio.run(connector.put(key, memory_obj))
    finally:
        asyncio.run(connector.close())
        connector.local_cpu_backend.memory_allocator.close()


def test_size_mismatch_get_returns_none(memory_allocator) -> None:
    """Loads should be rejected when the stored blob size is wrong."""
    connector, fake_client, _ = create_connector(memory_allocator)
    key = create_test_key(30)
    object_path = connector._key_string_to_object_path(key.to_string())
    fake_client.storage[object_path] = b"too-small"

    try:
        loaded = asyncio.run(connector.get(key))
        assert loaded is None
        assert connector.exists_sync(key) is False
    finally:
        asyncio.run(connector.close())
        connector.local_cpu_backend.memory_allocator.close()


def test_remote_backend_init_with_plugin_uses_builtin_gcs_adapter(
    monkeypatch,
    async_loop,
    local_cpu_backend,
) -> None:
    """RemoteBackend should create the built-in GCS connector via plugin_name."""
    fake_client = FakeGCSClient()
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.connector.gcs_connector.GCSClient",
        lambda project=None, credentials_path=None: fake_client,
    )

    config = create_test_config(
        {
            "remote_storage_plugin.gcs.bucket_handle": "gs://test-bucket/prod",
        }
    )
    metadata = create_test_metadata()
    backend = RemoteBackend(
        config=config,
        metadata=metadata,
        loop=async_loop,
        local_cpu_backend=local_cpu_backend,
        dst_device="cpu",
        plugin_name="gcs",
    )

    try:
        assert backend.plugin_name == "gcs"
        assert backend.remote_url == "plugin://gcs"
        assert backend.connection is not None
        assert backend.connection.__class__.__name__ == "InstrumentedRemoteConnector"
        wrapped_connector = backend.connection.getWrappedConnector()
        assert wrapped_connector.__class__.__name__ == "GCSConnector"
    finally:
        local_cpu_backend.memory_allocator.close()
        backend.close()
