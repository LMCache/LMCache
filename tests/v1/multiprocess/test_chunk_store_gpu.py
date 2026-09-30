# SPDX-License-Identifier: Apache-2.0
"""Cross-process GPU tests for the dynamically loaded chunk-store plugin."""

# Standard
from collections.abc import Iterator
from dataclasses import replace
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
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.server import run_cache_server
from lmcache.v1.multiprocess.server_module import ServerModuleSpec
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
            separate_object_groups=True,
            server_modules=[
                ServerModuleSpec("lmcache.v1.multiprocess.modules.chunk_store")
            ],
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


@pytest.mark.parametrize("start", [0, 16])
def test_chunk_completion_allows_source_reuse_and_exact_kv_retrieval(
    gpu_client: RequestClient,
    start: int,
) -> None:
    client = gpu_client
    backend = get_event_ipc_backend(0)
    tensors = [
        torch.rand((2, 32, 8, 4, 64), device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    wrappers: list[DeviceIPCWrapper] = [CudaIPCWrapper(tensor) for tensor in tensors]
    instance = os.getpid()
    client.register_kv_cache(
        instance, wrappers, "chunk-test", 1, EngineType.VLLM, {}, []
    ).result(TIMEOUT)
    prefix_blocks = start // 8
    expected = [
        torch.cat((tensor[:, 10 : 10 + prefix_blocks], tensor[:, 1:9]), dim=1)
        for tensor in tensors
    ]
    producer = backend.create_event(0)
    backend.record_event(producer, torch_dev.current_stream())
    key = IPCCacheServerKey.from_token_ids(
        "chunk-test",
        1,
        0,
        list(range(start + 64)),
        start=start,
        end=start + 64,
        request_id=f"chunk-{start}",
    )
    producer_handle = backend.export_event(producer, 0)
    if start:
        prefix = client.store(
            replace(key, start=0, end=start),
            instance,
            [list(range(10, 10 + prefix_blocks))],
            producer_handle,
        ).to_device_future(device=0, event_backend=backend)
        assert prefix.result(TIMEOUT)
    terminal, chunks, success = client.store_with_chunk_events(
        key, instance, [list(range(1, 9))], producer_handle
    ).result(TIMEOUT)
    assert success
    assert [(left, right) for _, left, right in chunks] == [
        (left, left + CHUNK) for left in range(start, start + 64, CHUNK)
    ]
    assert terminal == chunks[-1][0]
    events = [backend.import_event(handle, 0) for handle, _, _ in chunks]
    for chunk, event in enumerate(events):
        backend.synchronize_event(event, 0)
        assert backend.query_event(event)
        # Recycle each chunk's source as soon as its own event completes.
        for tensor in tensors:
            tensor[:, 1 + chunk * 2 : 3 + chunk * 2].zero_()

    lookup = replace(key, start=0, worker_id=None, request_id=f"lookup-{start}")
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        client.lookup(lookup, 1).result(TIMEOUT)
        hits = client.wait_prefetch_status(lookup.request_id, TIMEOUT).result(TIMEOUT)
        if hits == (start + 64) // CHUNK:
            break
        # Source completion can precede finish_write publication. Release this
        # attempt's read locks before retrying the prefix lookup.
        client.free_lookup_locks(lookup, 1).result(TIMEOUT)
        client.end_session(lookup.request_id).result(TIMEOUT)
        time.sleep(0.01)
    else:
        pytest.fail("stored chunks were not published before the deadline")
    consumed = backend.create_event(0)
    backend.record_event(consumed, torch_dev.current_stream())
    restored = client.retrieve(
        replace(lookup, worker_id=0),
        instance,
        [list(range(16, 24 + prefix_blocks))],
        backend.export_event(consumed, 0),
        0,
    ).to_device_future(device=0, event_backend=backend)
    assert restored.result(TIMEOUT)
    for tensor, original in zip(tensors, expected, strict=True):
        torch.testing.assert_close(
            tensor[:, 16 : 24 + prefix_blocks], original, rtol=0, atol=0
        )
    client.end_session(lookup.request_id).result(TIMEOUT)
    client.unregister_kv_cache(instance).result(TIMEOUT)
