# SPDX-License-Identifier: Apache-2.0
"""Integration test for the AzureConnector against the Azurite emulator.

TEST TYPE: integration against a REAL azure-storage-blob client + the official
Azurite Blob emulator over a real socket — but NO cloud account and NO GPU. It
uses the real SDK (not mocks), so it exercises the true serialization / API
contract. Skips automatically if Azurite is not running on 127.0.0.1:10000.
This is the level that actually catches real-SDK bugs the unit mocks cannot.

Doing so surfaced a real bug the mock-only tests hide (see
``test_put_memoryview_bug_and_read_path``): ``AzureConnector.put`` hands a raw
``memoryview`` (``memory_obj.byte_array``) to ``upload_blob``, which the real
``azure-storage-blob`` SDK rejects — so every write silently fails (best-effort)
and the cache never populates on real Blob. The fix is one line: upload
``bytes(memory_obj.byte_array)`` (or ``.cast('B')`` first).

Start Azurite first:
    azurite-blob --silent --location /tmp/azurite_data --blobPort 10000 --blobHost 127.0.0.1

Run:
    PYTHONPATH=. python -m pytest tests/v1/storage_backend/test_azure_connector_azurite.py -q
"""

# Standard
import asyncio
import socket
import threading
import time

# Third Party
import pytest
import torch
from azure.storage.blob import BlobServiceClient as SyncBlobServiceClient

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector.azure_connector import AzureConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

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


def run(loop, coro):
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


def _make_connector(container, async_loop, local_cpu_backend):
    return AzureConnector(
        container=container,
        loop=async_loop,
        local_cpu_backend=local_cpu_backend,
        connection_string=AZURITE_CONN,
    )


def test_real_blob_roundtrip_against_azurite(async_loop, local_cpu_backend):
    """Full write->read round-trip of AzureConnector against REAL Blob (Azurite).

    This is the regression test for the two bugs that mock-only tests hid, which
    were found by running against Azurite:
      * put() used to hand a raw ``memoryview`` to ``upload_blob`` -> the SDK
        raised "memoryview: unsupported format ...", and because writes are
        best-effort the error was swallowed, so nothing was ever stored.
      * get() used to call ``downloader.readinto(byte_array)`` -> the aio SDK's
        ``readinto`` writes to a *stream* (``.write``), not a buffer, so it raised
        AttributeError and returned None even when the blob was present.
    The connector now uploads ``bytes(...)`` and reads via ``readall()`` + copy.

    What each step verifies against the real Blob API:
      1. exists() before write  -> False   (real HEAD / get_blob_properties)
      2. put()                  -> actually stores the chunk (fix #1)
      3. exists() after write   -> True     (positive HEAD)
      4. get()                  -> byte-identical MemoryObj (fix #2, real download)
      5. list()                 -> the one blob we wrote (real list_blobs)
      6. close()                -> clean client shutdown
    Auth is exercised too: the connector authenticates via the Azurite connection
    string to reach any of these endpoints.
    """
    sync = SyncBlobServiceClient.from_connection_string(AZURITE_CONN)
    container = f"kvtest{int(time.time())}"
    sync.create_container(container)  # connector assumes the container pre-exists

    connector = _make_connector(container, async_loop, local_cpu_backend)
    key = create_test_key(1)

    # Allocate a full KV chunk and fill it with a known byte pattern.
    memory_obj = local_cpu_backend.allocate(
        connector.meta_shapes, connector.meta_dtypes, connector.meta_fmt
    )
    buf = memoryview(memory_obj.byte_array).cast("B")
    pattern = (bytes(range(256)) * ((len(buf) // 256) + 1))[: len(buf)]
    buf[:] = pattern
    original = bytes(buf)

    # 1. miss before any write
    assert run(async_loop, connector.exists(key)) is False

    # 2. put -> real upload (previously a silent no-op; now actually stores bytes)
    run(async_loop, connector.put(key, memory_obj))
    memory_obj.ref_count_down()
    # Confirm at the raw-SDK level that the blob really exists now.
    blob_name = connector._blob_name(key.to_string())
    assert sync.get_container_client(container).get_blob_client(blob_name).exists()

    # 3. exists after write (use a fresh connector so we test a real HEAD, not the
    #    in-process size cache populated by put()).
    reader = _make_connector(container, async_loop, local_cpu_backend)
    assert run(async_loop, reader.exists(key)) is True

    # 4. get -> real download, byte-identical to what we wrote (previously None)
    res = run(async_loop, reader.get(key))
    assert res is not None
    assert bytes(memoryview(res.byte_array).cast("B")) == original
    res.ref_count_down()

    # 5. list -> exactly the one blob we stored
    assert len(run(async_loop, reader.list())) == 1

    # 6. close both clients cleanly
    run(async_loop, reader.close())
    run(async_loop, connector.close())
