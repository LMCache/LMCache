# SPDX-License-Identifier: Apache-2.0
"""Concurrent multi-peer integration tests for the P2P L2 adapter.

The test starts three independent in-process peers. Each peer owns a disjoint
set of objects in its L1 cache, exposes the P2P lookup handlers over LMCache's
message queue, and serves its L1 buffer through a real NIXL transfer channel.
The local side runs one adapter per peer and drives those adapters concurrently
from separate workers through lookup, load, and unlock.

The test requires CUDA and a working NIXL runtime. It is intentionally kept in
the distributed integration-test tree, which is not part of the default
unit-test command.
"""

# Standard
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar, cast
import itertools
import os
import threading
import time

# Third Party
import pytest
import torch
import zmq

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.l2_adapters.p2p_l2_adapter import (
    P2PL2Adapter,
    P2PL2AdapterConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.distributed.transfer_channel import (
    delete_transfer_channel_context,
    initialize_transfer_channel_context,
)
from lmcache.v1.distributed.transfer_channel.impl.nixl_impl import (
    NixlTransferChannelContext,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.config import CoordinatorConfig, P2PConfig
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.p2p_controller import P2PController
from lmcache.v1.multiprocess.mq import MessageQueueServer
from lmcache.v1.multiprocess.protocol import get_payload_classes

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )

pytest.importorskip(
    "nixl",
    reason="NIXL runtime is required for the multi-peer P2P integration test",
)

_PAGE = 4096
_NUM_PEERS = 3
_NUM_KEYS_PER_PEER = 4
_PEER_L1_SIZE = 16 * 1024 * 1024
_POLL_TIMEOUT_S = 15.0
_port_counter = itertools.count(18500)
_T = TypeVar("_T")


def _next_url() -> str:
    """Return a unique loopback host/port endpoint for this test module."""
    return f"127.0.0.1:{next(_port_counter)}"


def _poll_result(
    query: Callable[[], _T | None],
    timeout_s: float = _POLL_TIMEOUT_S,
) -> _T:
    """Poll a public asynchronous result API until it completes.

    Args:
        query: Callable returning ``None`` while the operation is in progress.
        timeout_s: Maximum time to wait for completion.

    Returns:
        The completed operation result.

    Raises:
        TimeoutError: If the operation does not complete before ``timeout_s``.
    """
    deadline = time.monotonic() + timeout_s
    result = query()
    while result is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"operation did not complete in {timeout_s}s")
        time.sleep(min(0.01, remaining))
        result = query()
    return result


def _make_storage_manager(size_bytes: int, shm_name: str) -> StorageManager:
    """Create a small pinned-DRAM L1 storage manager for a test peer."""
    memory_config = L1MemoryManagerConfig(
        size_in_bytes=size_bytes,
        use_lazy=False,
        init_size_in_bytes=size_bytes,
        align_bytes=_PAGE,
        shm_name=shm_name,
    )
    l1_config = L1ManagerConfig(
        memory_config=memory_config,
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )
    config = StorageManagerConfig(
        l1_manager_config=l1_config,
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )
    return StorageManager(config)


class _PeerContext:
    """Minimal public context required by ``P2PController``."""

    def __init__(self, storage_manager: StorageManager) -> None:
        self.storage_manager = storage_manager


@dataclass
class _Peer:
    """Resources and data owned by one in-process P2P peer."""

    storage_manager: StorageManager
    transfer_context: NixlTransferChannelContext
    controller: P2PController
    mq_server: MessageQueueServer
    mq_url: str
    keys: list[ObjectKey]
    values: dict[ObjectKey, int]


class _LocalObject:
    """Public address-shaped test object accepted by ``submit_load_task``."""

    def __init__(self, offset: int, size: int) -> None:
        self.shm_offset = offset
        self.shm_byte_length = size


def _make_peer(peer_index: int, layout: MemoryLayoutDesc) -> _Peer:
    """Create one populated peer with real MQ and NIXL endpoints."""
    keys = [
        ObjectKey(
            chunk_hash=f"peer-{peer_index}-key-{key_index}".encode(),
            model_name="test_model",
            kv_rank=0,
        )
        for key_index in range(_NUM_KEYS_PER_PEER)
    ]
    values = {
        key: peer_index * _NUM_KEYS_PER_PEER + key_index + 1
        for key_index, key in enumerate(keys)
    }

    storage_manager = _make_storage_manager(
        _PEER_L1_SIZE,
        shm_name=f"lmcache_p2p_stress_{os.getpid()}_{peer_index}",
    )
    transfer_context: NixlTransferChannelContext | None = None
    controller: P2PController | None = None
    mq_server: MessageQueueServer | None = None
    try:
        reserved = storage_manager.reserve_write(keys, layout, mode="new")
        assert set(reserved) == set(keys)
        for key, value in values.items():
            tensor = reserved[key].tensor
            assert tensor is not None
            tensor.fill_(value)
        storage_manager.finish_write(keys)

        transfer_url = _next_url()
        transfer_context = NixlTransferChannelContext(
            storage_manager.l1_memory_desc,
            listen_url=transfer_url,
            advertise_url=transfer_url,
        )

        controller = P2PController(
            cast(MPCacheServerContext, _PeerContext(storage_manager)),
            P2PConfig(),
            CoordinatorConfig(),
            instance_id=f"peer-{peer_index}",
        )
        mq_url = f"tcp://{_next_url()}"
        mq_server = MessageQueueServer(mq_url, zmq.Context.instance())
        specs = controller.get_handlers()
        for spec in specs:
            mq_server.add_blocking_handler(
                spec.request_type,
                get_payload_classes(spec.request_type),
                spec.handler,
            )
        mq_server.add_normal_thread_pool(
            [spec.request_type for spec in specs], max_workers=4
        )
        mq_server.start()

        return _Peer(
            storage_manager=storage_manager,
            transfer_context=transfer_context,
            controller=controller,
            mq_server=mq_server,
            mq_url=mq_url,
            keys=keys,
            values=values,
        )
    except Exception:
        if mq_server is not None:
            mq_server.close()
        if controller is not None:
            controller.close()
        if transfer_context is not None:
            transfer_context.close()
        storage_manager.close()
        raise


def _run_peer_workload(
    peer: _Peer,
    adapter: P2PL2Adapter,
    all_keys: list[ObjectKey],
    layout: MemoryLayoutDesc,
    local_objects: list[MemoryObj],
    local_buffer: torch.Tensor,
    start_barrier: threading.Barrier,
) -> None:
    """Run one peer's lookup/isolation/load workload from one worker."""
    start_barrier.wait(timeout=_POLL_TIMEOUT_S)

    lookup_submitted = False
    lookup_succeeded = False
    try:
        lookup_id = adapter.submit_lookup_and_lock_task(all_keys, {0: layout})
        lookup_submitted = True
        bitmap = _poll_result(lambda: adapter.query_lookup_and_lock_result(lookup_id))

        expected_hit_keys = set(peer.keys)
        assert bitmap.popcount() == len(expected_hit_keys)
        for index, key in enumerate(all_keys):
            assert bitmap.test(index) is (key in expected_hit_keys)
        lookup_succeeded = True

        load_id = adapter.submit_load_task(peer.keys, local_objects)
        load_bitmap = _poll_result(lambda: adapter.query_load_result(load_id))
        assert load_bitmap.popcount() == len(peer.keys)
        for key_index, key in enumerate(peer.keys):
            assert load_bitmap.test(key_index) is True
            page_start = local_objects[key_index].shm_offset
            page = local_buffer[page_start : page_start + _PAGE]
            assert torch.all(page == peer.values[key]).item()

        status = adapter.report_status()
        assert status["in_flight_lookups"] == 0
        assert status["in_flight_loads"] == 0
        assert peer.controller.report_status()["active_p2p_lookup_jobs"] == 0
    finally:
        if lookup_submitted:
            if lookup_succeeded:
                # Only the peer's keys were locked; the other keys were misses.
                adapter.submit_unlock(peer.keys)
            else:
                # Best-effort cleanup also covers unexpected partial lookup hits.
                adapter.submit_unlock(all_keys)


@pytest.mark.cuda
@pytest.mark.integration
def test_multi_peer_concurrent_p2p_lookup_and_load() -> None:
    """Verify concurrent lookup and load correctness across three peers."""
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([_PAGE])],
        dtypes=[torch.uint8],
    )
    peers: list[_Peer] = []
    adapters: list[P2PL2Adapter] = []
    local_buffer = torch.zeros(
        _NUM_PEERS * _NUM_KEYS_PER_PEER * _PAGE,
        dtype=torch.uint8,
    )

    try:
        for peer_index in range(_NUM_PEERS):
            peers.append(_make_peer(peer_index, layout))
        local_url = _next_url()
        initialize_transfer_channel_context(
            "nixl",
            L1MemoryDesc(
                ptr=local_buffer.data_ptr(),
                size=local_buffer.numel(),
                align_bytes=_PAGE,
            ),
            local_url,
            local_url,
        )

        for peer in peers:
            adapters.append(
                P2PL2Adapter(
                    P2PL2AdapterConfig(
                        peer_mq_server_url=peer.mq_url,
                        peer_transfer_channel_server_url=(
                            peer.transfer_context.advertise_url
                        ),
                        lookup_timeout_s=_POLL_TIMEOUT_S,
                        load_timeout_s=_POLL_TIMEOUT_S,
                    )
                )
            )
        all_keys = [key for peer in peers for key in peer.keys]
        start_barrier = threading.Barrier(_NUM_PEERS)

        with ThreadPoolExecutor(max_workers=_NUM_PEERS) as executor:
            futures = [
                executor.submit(
                    _run_peer_workload,
                    peer,
                    adapters[peer_index],
                    all_keys,
                    layout,
                    cast(
                        list[MemoryObj],
                        [
                            _LocalObject(
                                offset=(peer_index * _NUM_KEYS_PER_PEER + key_index)
                                * _PAGE,
                                size=_PAGE,
                            )
                            for key_index in range(_NUM_KEYS_PER_PEER)
                        ],
                    ),
                    local_buffer,
                    start_barrier,
                )
                for peer_index, peer in enumerate(peers)
            ]
            for future in futures:
                future.result()
    finally:
        for adapter in adapters:
            adapter.close()
        delete_transfer_channel_context()
        for peer in peers:
            peer.controller.close()
            peer.mq_server.close()
            peer.transfer_context.close()
            peer.storage_manager.close()
