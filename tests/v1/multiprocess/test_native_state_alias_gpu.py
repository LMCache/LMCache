# SPDX-License-Identifier: Apache-2.0
"""Real MP round trip of strided byte PAGE and checkpoint aliases on CUDA/ROCm.

Run with a GPU-enabled LMCache build::

    pytest -xvs tests/v1/multiprocess/test_native_state_alias_gpu.py

The test spawns its own small MP server on a free local port and registers the
original pool aliases over IPC. LMCache owns any internal transfer buffers.
"""

# Standard
from collections.abc import Generator
from dataclasses import replace
import multiprocessing as mp
import os
import socket
import time

# Third Party
import pytest
import torch
import zmq

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
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.server import run_cache_server
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend

CHUNK = 16
TIMEOUT = 30.0
pytestmark = [pytest.mark.cuda, pytest.mark.no_shared_allocator]

if not (torch_dev.is_available() and torch_device_type == "cuda"):
    pytest.skip("requires a CUDA or ROCm runtime", allow_module_level=True)

# First Party
from lmcache.v1.platform.devices.cuda.ipc_wrapper import CudaIPCWrapper  # noqa: E402


def _serve(port: int) -> None:
    """Run an isolated MP server with separate PAGE and STATE objects."""
    run_cache_server(
        mp_config=MPServerConfig(
            host="127.0.0.1",
            port=port,
            chunk_size=CHUNK,
            null_block_id=-1,
            separate_object_groups=True,
        ),
        storage_manager_config=StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=64 * 1024**2,
                    use_lazy=True,
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
        ),
        obs_config=DEFAULT_OBSERVABILITY_CONFIG,
    )


@pytest.fixture
def native_client() -> Generator[RequestClient, None, None]:
    """Start a separate server process and stop it after the test."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    process = mp.get_context("spawn").Process(target=_serve, args=(port,), daemon=True)
    process.start()
    context = zmq.Context()
    client = RequestClientFactory.create(f"tcp://127.0.0.1:{port}", context=context)
    try:
        deadline = time.monotonic() + TIMEOUT
        while time.monotonic() < deadline:
            assert process.is_alive(), f"MP server exited: {process.exitcode}"
            try:
                if client.ping(None).result(timeout=0.5):
                    break
            except LMCacheTimeoutError:
                continue
        else:
            pytest.fail("MP server did not become ready")
        yield client
    finally:
        client.close()
        context.destroy(linger=0)
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


def _transfer(
    client: RequestClient,
    operation: str,
    key: IPCCacheServerKey,
    instance_id: int,
    block_ids: list[list[int]],
    device: torch.device,
) -> bool:
    """Wait for both the MQ response and its backend-specific device event."""
    backend = get_event_ipc_backend(device)
    event = backend.create_event(device)
    backend.record_event(event, None)
    handle = backend.export_event(event, device)
    if operation == "store":
        future = client.store(key, instance_id, block_ids, handle)
    else:
        future = client.retrieve(key, instance_id, block_ids, handle, 0)
    # Keep the producer event alive until the remote copy has completed.
    future.retain_reference(event)
    return future.to_device_future(device, backend).result(TIMEOUT)


def _lookup(client: RequestClient, key: IPCCacheServerKey) -> int:
    """Submit a real prefetch and wait for its final common prefix length."""
    client.lookup(key, 1).result(TIMEOUT)
    result = client.wait_prefetch_status(key.request_id, TIMEOUT).result(TIMEOUT + 1)
    assert result is not None
    return int(result)


def _lookup_when_visible(
    client: RequestClient,
    key: IPCCacheServerKey,
    expected_hit: int,
) -> IPCCacheServerKey:
    """Wait until an asynchronous store is visible to prefix lookup.

    The store device event covers the D2H copy, while ``finish_write`` runs in
    a later stream-ordered host callback. A lookup submitted immediately after
    the device event may therefore observe the previous committed prefix. Each
    unsuccessful probe releases its lookup locks and session before retrying.
    """
    deadline = time.monotonic() + TIMEOUT
    attempt = 0
    last_hit = -1
    while time.monotonic() < deadline:
        attempt_key = replace(key, request_id=f"{key.request_id}-{attempt}")
        last_hit = _lookup(client, attempt_key)
        if last_hit == expected_hit:
            return attempt_key
        client.free_lookup_locks(attempt_key, 1).result(TIMEOUT)
        client.end_session(attempt_key.request_id).result(TIMEOUT)
        if last_hit > expected_hit:
            break
        attempt += 1
        time.sleep(0.01)
    raise AssertionError(
        f"store did not become visible before timeout: "
        f"expected {expected_hit} chunks, last lookup returned {last_hit}"
    )


def test_native_alias_sparse_checkpoint_roundtrip(native_client: RequestClient) -> None:
    """Restore exact page/checkpoint payloads without modifying adjacent bytes."""
    device = torch.device("cuda", 0)
    page_pool = torch.randint(0, 255, (32, 1, 96), dtype=torch.uint8, device=device)
    state_pool = torch.full((8, 1, 128), 199, dtype=torch.uint8, device=device)
    # Singleton BS still needs a canonical inner stride for IPC validation.
    aliases = [
        page_pool.as_strided((32, 1, 64), (96, 64, 1)),
        state_pool.as_strided((8, 1, 32), (128, 32, 1)),
        state_pool.as_strided((8, 1, 48), (128, 48, 1), storage_offset=48),
    ]
    aliases[1][0].fill_(11)
    aliases[2][0].fill_(23)
    aliases[1][1].fill_(55)
    aliases[2][1].fill_(77)
    expected_pages = aliases[0][:16].cpu().clone()
    expected_state = [alias[:2].cpu().clone() for alias in aliases[1:]]
    groups = [
        EngineGroupInfo(0, (0,), tokens_per_block=4),
        EngineGroupInfo(
            1,
            (1,),
            tokens_per_block=CHUNK,
            sw_size_tokens=CHUNK,
            recurrent_state=True,
        ),
        EngineGroupInfo(
            2,
            (2,),
            tokens_per_block=CHUNK,
            sw_size_tokens=CHUNK,
            recurrent_state=True,
        ),
    ]
    instance_id = os.getpid()
    tokens = tuple(range(4 * CHUNK + 1))
    key = IPCCacheServerKey(
        model_name="native-alias-roundtrip",
        world_size=1,
        worker_id=0,
        token_ids=tokens,
        start=0,
        end=4 * CHUNK,
        request_id="native-save",
        num_kv_readers=1,
    )
    native_client.register_kv_cache(
        instance_id,
        [CudaIPCWrapper(alias) for alias in aliases],
        key.model_name,
        1,
        EngineType.ATOM,
        {},
        groups,
    ).result(TIMEOUT)
    try:
        assert _transfer(
            native_client,
            "store",
            replace(key, end=2 * CHUNK),
            instance_id,
            [list(range(8)), [-1, 0], [-1, 0]],
            device,
        )
        assert _transfer(
            native_client,
            "store",
            replace(key, start=2 * CHUNK),
            instance_id,
            [list(range(8, 16)), [-1, 1], [-1, 1]],
            device,
        )
        # Destroy the original payloads: restoring must read the server's L1.
        page_pool.fill_(199)
        state_pool.fill_(199)
        for limit_chunks, expected_hit, source_unit, destination_unit in (
            (4, 4, 1, 3),
            (3, 2, 0, 4),
        ):
            lookup_key = replace(
                key,
                worker_id=None,
                token_ids=tokens[: limit_chunks * CHUNK],
                end=limit_chunks * CHUNK,
                request_id=f"native-load-{limit_chunks}",
            )
            lookup_key = _lookup_when_visible(native_client, lookup_key, expected_hit)
            retrieve_key = replace(lookup_key, worker_id=0, end=expected_hit * CHUNK)
            target_pages = list(range(16, 16 + expected_hit * 4))
            target_state = [-1] * (expected_hit - 1) + [destination_unit]
            assert _transfer(
                native_client,
                "retrieve",
                retrieve_key,
                instance_id,
                [target_pages, target_state, target_state],
                device,
            )
            torch.testing.assert_close(
                aliases[0][target_pages].cpu(), expected_pages[: expected_hit * 4]
            )
            for alias, expected in zip(aliases[1:], expected_state, strict=True):
                torch.testing.assert_close(
                    alias[destination_unit].cpu(), expected[source_unit]
                )
            native_client.end_session(lookup_key.request_id).result(TIMEOUT)
        # Aliases expose payload bytes only: padding/gaps and other checkpoint
        # units must remain untouched by both restore operations.
        assert torch.all(page_pool[:, :, 64:] == 199)
        assert torch.all(state_pool[:, :, 32:48] == 199)
        assert torch.all(state_pool[:, :, 96:] == 199)
        assert torch.all(state_pool[[0, 1, 2, 5, 6, 7]] == 199)
    finally:
        native_client.unregister_kv_cache(instance_id).result(TIMEOUT)


def test_state_ordinals_alias_the_page_allocation(native_client: RequestClient) -> None:
    """Multiple engine groups can register and copy the same native PAGE pool."""
    device = torch.device("cuda", 0)
    pool = torch.randint(0, 199, (40, 1, 96), dtype=torch.uint8, device=device)
    aliases = [pool, pool, pool.as_strided((40, 1, 32), (96, 32, 1))]
    expected_pages = pool[:8].cpu().clone()
    expected_state = [pool[12].cpu().clone(), pool[15, :, :32].cpu().clone()]
    groups = [
        EngineGroupInfo(0, (0,), tokens_per_block=4),
        EngineGroupInfo(
            1,
            (1,),
            tokens_per_block=CHUNK,
            sw_size_tokens=CHUNK,
            recurrent_state=True,
        ),
        EngineGroupInfo(
            2,
            (2,),
            tokens_per_block=CHUNK,
            sw_size_tokens=CHUNK,
            recurrent_state=True,
        ),
    ]
    instance_id = os.getpid()
    key = IPCCacheServerKey(
        model_name="native-shared-page-pool",
        world_size=1,
        worker_id=0,
        token_ids=tuple(range(2 * CHUNK)),
        start=0,
        end=2 * CHUNK,
        request_id="same-pool-save",
        num_kv_readers=1,
    )
    native_client.register_kv_cache(
        instance_id,
        [CudaIPCWrapper(alias) for alias in aliases],
        key.model_name,
        1,
        EngineType.ATOM,
        {},
        groups,
    ).result(TIMEOUT)
    try:
        assert _transfer(
            native_client,
            "store",
            key,
            instance_id,
            [list(range(8)), [-1, 12], [-1, 15]],
            device,
        )
        pool.fill_(199)
        lookup_key = replace(key, worker_id=None, request_id="same-pool-load")
        lookup_key = _lookup_when_visible(native_client, lookup_key, 2)
        assert _transfer(
            native_client,
            "retrieve",
            replace(lookup_key, worker_id=0),
            instance_id,
            [list(range(16, 24)), [-1, 30], [-1, 31]],
            device,
        )
        torch.testing.assert_close(pool[16:24].cpu(), expected_pages)
        torch.testing.assert_close(pool[30].cpu(), expected_state[0])
        torch.testing.assert_close(pool[31, :, :32].cpu(), expected_state[1])
        assert torch.all(pool[31, :, 32:] == 199)
        untouched = [i for i in range(40) if i not in {*range(16, 24), 30, 31}]
        assert torch.all(pool[untouched] == 199)
        native_client.end_session(lookup_key.request_id).result(TIMEOUT)
    finally:
        native_client.unregister_kv_cache(instance_id).result(TIMEOUT)
