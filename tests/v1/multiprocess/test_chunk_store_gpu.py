# SPDX-License-Identifier: Apache-2.0
"""Real CUDA/ROCm chunk-store IPC lifetime and KV round trips."""

# Standard
from collections.abc import Iterator
from multiprocessing.synchronize import Event as ProcessEvent
import multiprocessing as mp
import os
import socket
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.utils import EngineType
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.mp_observability.config import DEFAULT_OBSERVABILITY_CONFIG
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.chunk_event_future import ChunkEventDeviceMessagingFuture
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.server import run_cache_server
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from tests.v1.multiprocess.transport_test_utils import (
    REQUEST_TRANSPORTS,
    RequestTransport,
    request_server_url,
)

pytestmark = [pytest.mark.cuda, pytest.mark.no_shared_allocator]
if not (torch_device_type == "cuda" and torch_dev.is_available()):
    pytest.skip("requires CUDA or ROCm", allow_module_level=True)

# First Party
from lmcache.v1.platform.devices.cuda.ipc_wrapper import CudaIPCWrapper  # noqa: E402

CHUNK = 16
TIMEOUT = 120.0


def _serve(transport: RequestTransport, port: int, stop: ProcessEvent) -> None:
    runtime = run_cache_server(
        mp_config=MPServerConfig(
            transport=transport,
            host="127.0.0.1",
            port=port,
            chunk_size=CHUNK,
            enable=["chunk_store"],
            null_block_id=-1,
        ),
        storage_manager_config=StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=256 * 1024**2,
                    init_size_in_bytes=64 * 1024**2,
                    use_lazy=True,
                )
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
        ),
        obs_config=DEFAULT_OBSERVABILITY_CONFIG,
        return_engine=True,
        start_prometheus_http_server=False,
    )
    assert runtime is not None
    server, engine = runtime
    try:
        stop.wait(TIMEOUT * 4)
    finally:
        server.close()
        engine.close()


@pytest.fixture(params=REQUEST_TRANSPORTS)
def gpu_client(request: pytest.FixtureRequest) -> Iterator[RequestClient]:
    transport = request.param
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    process = ctx.Process(target=_serve, args=(transport, port, stop))
    process.start()
    client = RequestClientFactory.create(request_server_url(transport, port))
    try:
        deadline = time.monotonic() + TIMEOUT
        while time.monotonic() < deadline:
            assert process.is_alive(), f"server exited: {process.exitcode}"
            try:
                assert client.get_chunk_size().result(0.5) == CHUNK
                break
            except LMCacheTimeoutError:
                continue
        else:
            pytest.fail("chunk store server did not become ready")
        yield client
    finally:
        client.close()
        stop.set()
        process.join(20)
        if process.is_alive():
            process.terminate()
            process.join(10)
        if process.is_alive():
            process.kill()
            process.join()


def test_delayed_chunk_import_after_2100_exports_and_kv_round_trip(
    gpu_client: RequestClient,
) -> None:
    client = gpu_client
    backend = get_event_ipc_backend(0)
    tensors = [
        torch.rand((2, 4224, 8, 4, 64), device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    wrappers: list[DeviceIPCWrapper] = [CudaIPCWrapper(tensor) for tensor in tensors]
    instance = os.getpid()
    client.register_kv_cache(
        instance, wrappers, "chunk-test", 1, EngineType.VLLM, {}, []
    ).result(TIMEOUT)
    producer = backend.create_event(0)
    backend.record_event(producer, torch_dev.current_stream())
    producer_handle = backend.export_event(producer, 0)
    key = IPCCacheServerKey.from_token_ids(
        "chunk-test",
        1,
        0,
        list(range(64)),
        start=0,
        end=64,
        request_id="first",
    )
    first = client.store_with_chunk_events(
        key, instance, [list(range(1, 9))], producer_handle
    )
    first_response = first.result(TIMEOUT)
    assert first_response[2]
    assert [(start, end) for _, start, end in first_response[1]] == [
        (0, 16),
        (16, 32),
        (32, 48),
        (48, 64),
    ]

    churn_key = IPCCacheServerKey.from_token_ids(
        "chunk-test",
        1,
        0,
        list(range(100000, 100000 + 2100 * CHUNK)),
        start=0,
        end=2100 * CHUNK,
        request_id="churn",
    )
    churn = client.store_with_chunk_events(
        churn_key, instance, [list(range(1, 4201))], producer_handle
    )
    assert len(churn.result(TIMEOUT)[1]) == 2100

    def release(lease: str) -> MessagingFuture[None]:
        return client.release_chunk_store_events(instance, lease)

    churn_future = ChunkEventDeviceMessagingFuture(churn, 0, backend, release)
    assert churn_future.result(TIMEOUT)
    churn_future.wait_for_release(TIMEOUT)

    # Import the first store's handles only after the backend's 2048-event ring
    # has rolled over. The service's lease must be their remaining owner.
    first_future = ChunkEventDeviceMessagingFuture(first, 0, backend, release)
    assert first_future.result(TIMEOUT)
    first_future.wait_for_release(TIMEOUT)
    assert first_future.take_completed_ranges() == (
        (0, 16),
        (16, 32),
        (32, 48),
        (48, 64),
    )
    expected = [tensor[:, 1:9].clone() for tensor in tensors]
    for tensor in tensors:
        tensor[:, 1:9].zero_()
    lookup = key.no_worker_id_version()
    client.lookup(lookup, 1).result(TIMEOUT)
    hits = client.wait_prefetch_status(lookup.request_id, TIMEOUT).result(TIMEOUT)
    assert hits == 4
    consumed = backend.create_event(0)
    backend.record_event(consumed, torch_dev.current_stream())
    restored = client.retrieve(
        key, instance, [list(range(4208, 4216))], backend.export_event(consumed, 0), 0
    ).to_device_future(
        device=0,
        event_backend=backend,
    )
    assert restored.result(TIMEOUT)
    for tensor, original in zip(tensors, expected, strict=True):
        torch.testing.assert_close(tensor[:, 4208:4216], original, rtol=0, atol=0)
    assert first_future.query()
    assert first_future.take_completed_ranges() == ()
    client.unregister_kv_cache(instance).result(TIMEOUT)
