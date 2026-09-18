# SPDX-License-Identifier: Apache-2.0
"""Exercise cloud key identity through public connectors and SDK I/O boundaries."""

# Standard
from concurrent.futures import Future
from types import SimpleNamespace
from urllib.parse import quote
import asyncio
import hashlib

# Third Party
import pytest
import pytest_asyncio
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector import CreateConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

pytestmark = [pytest.mark.no_shared_allocator, pytest.mark.asyncio]


def _key(model: str) -> CacheEngineKey:
    """Return a distinct model identity with the same chunk hash and topology."""
    return CacheEngineKey(model, 1, 0, 1001, torch.float32)


@pytest_asyncio.fixture(params=["s3", "azure"])
async def cloud(request, monkeypatch):
    """Construct real adapters, connectors and allocators with isolated I/O.

    Only SDK client/request operations are replaced. Key formatting, connector
    get/put, allocation, instrumentation and S3 scheduling run production code.
    """
    provider = request.param
    objects = {}
    calls = []
    connections = []

    if provider == "s3":
        # First Party
        from lmcache.v1.storage_backend.connector import s3_connector

        def http_request(method, path, headers, body_stream=None):
            """Keep the SDK request fields observable at the I/O boundary."""
            return SimpleNamespace(
                method=method, path=path, headers=headers, body_stream=body_stream
            )

        def s3_request(*, request, on_done, on_headers=None, on_body=None, **kwargs):
            """Execute one deterministic object-store operation without a network."""
            path = request.path.removeprefix("/")
            calls.append((request.method, path))
            result = Future()
            if request.method == "PUT":
                objects[path] = bytes(request.body_stream.read())
                on_done(status_code=200)
            elif request.method == "HEAD":
                present = path in objects
                on_headers(
                    200 if present else 404,
                    [("content-length", str(len(objects.get(path, b""))))],
                )
                on_done()
            elif request.method == "GET":
                on_body(objects[path], 0)
                on_done(status_code=200)
            else:
                raise AssertionError(request.method)
            result.set_result(None)
            return SimpleNamespace(finished_future=result)

        monkeypatch.setattr(s3_connector, "HttpRequest", http_request)
        monkeypatch.setattr(s3_connector.s3, "S3Client", lambda **kwargs: object())
        monkeypatch.setattr(s3_connector.s3, "S3Request", s3_request)
        extra = {
            "s3_region": "us-east-1",
            "s3_num_io_threads": 1,
            "s3_prefer_http2": False,
            "disable_tls": True,
            "aws_access_key_id": "test",
            "aws_secret_access_key": "test",
        }
    else:
        azure_blob = pytest.importorskip("azure.storage.blob.aio")
        azure_errors = pytest.importorskip("azure.core.exceptions")

        async def upload(blob, data, *, overwrite, length, **kwargs):
            """Capture bytes under the name passed to the real Azure BlobClient."""
            assert overwrite and length == len(data)
            calls.append(("PUT", blob.blob_name))
            objects[blob.blob_name] = bytes(data)

        async def properties(blob, **kwargs):
            """Return properties or the real SDK's missing-object exception."""
            calls.append(("HEAD", blob.blob_name))
            if blob.blob_name not in objects:
                raise azure_errors.ResourceNotFoundError("missing test object")
            return SimpleNamespace(size=len(objects[blob.blob_name]))

        async def download(blob, **kwargs):
            """Return the bytes addressed by the real Azure BlobClient."""
            calls.append(("GET", blob.blob_name))
            data = objects[blob.blob_name]

            async def readall():
                return data

            return SimpleNamespace(readall=readall)

        monkeypatch.setattr(azure_blob.BlobClient, "upload_blob", upload)
        monkeypatch.setattr(azure_blob.BlobClient, "get_blob_properties", properties)
        monkeypatch.setattr(azure_blob.BlobClient, "download_blob", download)
        extra = {
            "azure_account_url": "https://test.blob.core.windows.net",
            "azure_account_key": "dGVzdA==",
        }

    def connect(model):
        """Create a fresh public connector with model-specific metadata."""
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=2,
            remote_serde="naive",
            extra_config=extra,
        )
        metadata = LMCacheMetadata(
            model_name=model,
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.float32,
            kv_shape=(1, 2, 2, 1, 1),
            chunk_size=2,
        )
        allocator = TensorMemoryAllocator(
            torch.empty(64 * 1024, dtype=torch.uint8), align_bytes=1
        )
        local = LocalCPUBackend(config, metadata, memory_allocator=allocator)
        connector = CreateConnector(
            f"{provider}://identity-test",
            asyncio.get_running_loop(),
            local,
            config,
            metadata,
        )
        connections.append((connector, local))
        return connector

    try:
        yield SimpleNamespace(
            provider=provider, objects=objects, calls=calls, connect=connect
        )
    finally:
        for connector, local in reversed(connections):
            if provider == "s3":
                # The existing S3 close synchronously waits on its own loop.
                # This regression exercises key identity, so drain via the
                # executor's public async cleanup instead (shutdown is #4490).
                await connector._connector.pq_executor.shutdown_async(wait=False)
            else:
                await connector.close()
            local.close()


async def _put(connector, key, value):
    """Upload one full 16-byte chunk while preserving the caller reference."""
    inner = connector._connector
    memory_obj = inner.local_cpu_backend.allocate(
        inner.meta_shapes, inner.meta_dtypes, inner.meta_fmt
    )
    assert memory_obj is not None
    assert memory_obj.get_size() == memory_obj.get_physical_size() == 16
    memory_obj.tensor.fill_(value)
    expected = bytes(memory_obj.byte_array)
    memory_obj.ref_count_up()
    try:
        await connector.put(key, memory_obj)
        assert memory_obj.get_ref_count() == 1
    finally:
        memory_obj.ref_count_down()
    return expected


async def _read(connector, key):
    """Read and release a real tensor-backed object, returning its bytes."""
    result = await connector.get(key)
    assert result is not None
    try:
        return bytes(result.byte_array)
    finally:
        result.ref_count_down()


async def test_single_model_roundtrip(cloud):
    """A healthy control writes and reads through independent instances."""
    key = _key("one/model")
    expected = await _put(cloud.connect(key.model_name), key, 1.25)
    assert await _read(cloud.connect(key.model_name), key) == expected
    assert {method for method, _ in cloud.calls} == {"PUT", "HEAD", "GET"}
    assert len({name for _, name in cloud.calls}) == 1


async def test_distinct_models_keep_their_payloads(cloud):
    """Former slash/underscore aliases must not overwrite another model's KV."""
    first, second = _key("a/b_c"), _key("a_b/c")
    assert first.to_string() != second.to_string()
    expected_first = await _put(cloud.connect(first.model_name), first, 1.25)
    expected_second = await _put(cloud.connect(second.model_name), second, 9.5)
    # Fresh readers ensure the result is independent of cached object sizes.
    assert await _read(cloud.connect(first.model_name), first) == expected_first
    assert await _read(cloud.connect(second.model_name), second) == expected_second
    assert len(cloud.objects) == 2


async def test_ambiguous_legacy_object_is_not_read_or_modified(cloud):
    """The new namespace must never guess which identity owns a legacy object."""
    key = _key("a/b_c")
    legacy = quote(
        key.to_string().replace("/", "_"), safe="/" if cloud.provider == "s3" else ""
    )
    old_bytes = bytes(range(16))
    cloud.objects[legacy] = old_bytes
    connector = cloud.connect(key.model_name)
    result = await connector.get(key)
    try:
        assert result is None
    finally:
        if result is not None:
            result.ref_count_down()
    assert cloud.objects == {legacy: old_bytes}
    expected = await _put(connector, key, 5.0)
    assert await _read(cloud.connect(key.model_name), key) == expected
    assert cloud.objects[legacy] == old_bytes
    assert len(cloud.objects) == 2


@pytest.mark.parametrize(
    "model",
    ["模型/é_名字", "x" * 900 + "/y", "x%2Fy", "x_y"],
    ids=["unicode", "long", "percent", "underscore"],
)
async def test_object_names_are_bounded_and_consistent(cloud, model):
    """Long and Unicode identities retain a bounded, URL-safe object name."""
    key = _key(model)
    expected = await _put(cloud.connect(model), key, 2.5)
    assert await _read(cloud.connect(model), key) == expected
    names = {name for _, name in cloud.calls}
    assert len(names) == 1
    name = names.pop()
    assert name.startswith("lmcache-v2/")
    assert len(name.encode("ascii")) == 75
    assert set(name.removeprefix("lmcache-v2/")) <= set("0123456789abcdef")
    assert (
        name
        == "lmcache-v2/" + hashlib.sha256(key.to_string().encode("utf-8")).hexdigest()
    )
