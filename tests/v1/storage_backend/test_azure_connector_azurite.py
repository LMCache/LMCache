# SPDX-License-Identifier: Apache-2.0
"""Integration test for the AzureConnector against the Azurite emulator.

Uses the real ``azure-storage-blob`` client against the Azurite Blob emulator
(no cloud account, no GPU). Skips automatically if Azurite is not running on
127.0.0.1:10000.

Start Azurite first:
    azurite-blob --silent --location /tmp/azurite_data --blobPort 10000 \
        --blobHost 127.0.0.1 --skipApiVersionCheck

Run:
    PYTHONPATH=. python -m pytest \
        tests/v1/storage_backend/test_azure_connector_azurite.py -q
"""

# Standard
from collections.abc import Generator
import asyncio
import socket
import threading
import uuid

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector.azure_connector import AzureConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

azure_blob = pytest.importorskip("azure.storage.blob")
SyncBlobServiceClient = azure_blob.BlobServiceClient

AZURITE_CONN = (
    "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;"
    "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/"
    "K1SZFPTOtr/KBHBeksoGMGw==;"
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
)


def _azurite_up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 10000), timeout=0.5):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _azurite_up(), reason="Azurite blob emulator not reachable on 127.0.0.1:10000"
)


def create_test_metadata(kv_shape=(1, 2, 16, 8, 128), chunk_size=16) -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=kv_shape,
        chunk_size=chunk_size,
    )


def create_test_key(key_id: int) -> CacheEngineKey:
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
    # First Party
    from lmcache.utils import start_loop_in_thread_with_exceptions

    thread = threading.Thread(
        target=start_loop_in_thread_with_exceptions,
        args=(loop,),
        name="test-azurite-loop",
    )
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5.0)


@pytest.fixture
def local_cpu_backend(memory_allocator):
    config = LMCacheEngineConfig.from_legacy(chunk_size=16)
    return LocalCPUBackend(
        config, create_test_metadata(), memory_allocator=memory_allocator
    )


@pytest.fixture
def azurite_container() -> Generator[str, None, None]:
    sync = SyncBlobServiceClient.from_connection_string(AZURITE_CONN)
    container = f"kvtest{uuid.uuid4().hex}"
    sync.create_container(container)
    try:
        yield container
    finally:
        sync.delete_container(container)
        sync.close()


def run(loop, coro):
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


def _make_connector(container, async_loop, local_cpu_backend):
    return AzureConnector(
        container=container,
        loop=async_loop,
        local_cpu_backend=local_cpu_backend,
        connection_string=AZURITE_CONN,
    )


def test_real_blob_roundtrip_against_azurite(
    async_loop, local_cpu_backend, azurite_container
):
    """Full write->read round-trip of AzureConnector against real Blob (Azurite).

    Steps: exists (miss) -> put -> exists (hit) -> get (byte-identical) ->
    list -> close.
    """
    connector = _make_connector(azurite_container, async_loop, local_cpu_backend)
    key = create_test_key(1)

    memory_obj = local_cpu_backend.allocate(
        connector.meta_shapes, connector.meta_dtypes, connector.meta_fmt
    )
    buf = memoryview(memory_obj.byte_array).cast("B")
    pattern = (bytes(range(256)) * ((len(buf) // 256) + 1))[: len(buf)]
    buf[:] = pattern
    original = bytes(buf)

    assert run(async_loop, connector.exists(key)) is False

    run(async_loop, connector.put(key, memory_obj))
    memory_obj.ref_count_down()

    # fresh connector to bypass the in-process size cache
    reader = _make_connector(azurite_container, async_loop, local_cpu_backend)
    assert run(async_loop, reader.exists(key)) is True

    res = run(async_loop, reader.get(key))
    assert res is not None
    assert bytes(memoryview(res.byte_array).cast("B")) == original
    res.ref_count_down()

    assert len(run(async_loop, reader.list())) == 1

    run(async_loop, reader.close())
    run(async_loop, connector.close())
