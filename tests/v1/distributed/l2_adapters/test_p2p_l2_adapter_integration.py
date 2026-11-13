# SPDX-License-Identifier: Apache-2.0
"""In-process, real-NIXL integration tests for the P2P L2 adapter.

Stands up a peer side (a real ``StorageManager``, a NIXL transfer-channel
context registered against its L1, and a request server hosting a
``P2PController``) and a local side (the global NIXL context over a local
buffer + a ``P2PL2Adapter``). The tests drive the adapter through the read
lifecycle (lookup -> loopback RDMA read -> unlock) and the write lifecycle
(reserve -> loopback RDMA write -> commit) and compare the bytes on both
sides.

Requires a working NIXL runtime and CUDA (the L1 pool is pinned DRAM); skipped
otherwise.
"""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, cast
import itertools
import time

# Third Party
import pytest
import torch
import zmq

# First Party
from lmcache import torch_dev, torch_device_type

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )

nixl = pytest.importorskip("nixl")

# First Party
from lmcache.v1.client.transport.factory import RequestClientFactory  # noqa: E402
from lmcache.v1.distributed.api import (  # noqa: E402
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.config import (  # noqa: E402
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.error import L1Error  # noqa: E402
from lmcache.v1.distributed.internal_api import (  # noqa: E402
    L1ManagerInterface,
    L1MemoryDesc,
    L2StoreResult,
)
from lmcache.v1.distributed.l2_adapters.p2p_l2_adapter import (  # noqa: E402
    P2PL2Adapter,
    P2PL2AdapterConfig,
)
from lmcache.v1.distributed.storage_manager import (  # noqa: E402
    StorageManager,
    l1_owners_of,
)
from lmcache.v1.distributed.transfer_channel import (  # noqa: E402
    delete_transfer_channel_context,
    initialize_transfer_channel_context,
)
from lmcache.v1.distributed.transfer_channel.impl.nixl_impl import (  # noqa: E402
    NixlTransferChannelContext,
)
from lmcache.v1.memory_management import (  # noqa: E402
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.multiprocess.config import (  # noqa: E402
    CoordinatorConfig,
    MPServerConfig,
    P2PConfig,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext  # noqa: E402
from lmcache.v1.multiprocess.modules.p2p_controller import P2PController  # noqa: E402
from lmcache.v1.multiprocess.transport.base import RequestServer  # noqa: E402
from lmcache.v1.multiprocess.transport.server_factory import (  # noqa: E402
    create_request_server,
)
from tests.v1.distributed.utils import single_row_spec  # noqa: E402

_PAGE = 4096
_NUM_KEYS = 3
_port_counter = itertools.count(18300)
RequestTransport = Literal["zmq", "grpc"]


def _next_url() -> str:
    return f"127.0.0.1:{next(_port_counter)}"


def _start_p2p_request_server(
    transport: RequestTransport,
    controller: P2PController,
) -> tuple[str, RequestServer]:
    """Start a P2P request server for the selected transport.

    Args:
        transport: Request transport to exercise.
        controller: P2P controller that handles peer requests.

    Returns:
        The client URL and the started request server.
    """
    target = _next_url()
    host, port_str = target.rsplit(":", maxsplit=1)
    mp_config = MPServerConfig(
        transport=transport,
        host=host,
        port=int(port_str),
        max_cpu_workers=4,
        max_gpu_workers=4,
    )
    request_server = create_request_server([controller], mp_config)
    request_server.start()
    scheme = "tcp" if transport == "zmq" else "grpc"
    return f"{scheme}://{target}", request_server


def _local_memory_obj(buffer: torch.Tensor, offset: int) -> MemoryObj:
    """Wrap one destination page in the production memory-object type."""
    metadata = MemoryObjMetadata(
        shape=torch.Size([_PAGE]),
        dtype=torch.uint8,
        address=offset,
        phy_size=_PAGE,
        ref_count=1,
        shapes=[torch.Size([_PAGE])],
        dtypes=[torch.uint8],
    )
    return TensorMemoryObj(
        raw_data=buffer[offset : offset + _PAGE],
        metadata=metadata,
        parent_allocator=None,
    )


def _make_storage_manager(
    size_bytes: int, write_ttl_seconds: int = 600, store_policy: str = "default"
) -> StorageManager:
    memory_config = L1MemoryManagerConfig(
        size_in_bytes=size_bytes,
        use_lazy=False,
        init_size_in_bytes=size_bytes,
        align_bytes=_PAGE,
    )
    l1_config = L1ManagerConfig(
        memory_config=memory_config,
        write_ttl_seconds=write_ttl_seconds,
        read_ttl_seconds=300,
    )
    config = StorageManagerConfig(
        l1_manager_config=l1_config,
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        store_policy=store_policy,
    )
    return StorageManager(config)


class _PeerContext:
    """Minimal P2PController context -- only ``storage_manager`` is used."""

    def __init__(self, storage_manager: StorageManager) -> None:
        self.storage_manager = storage_manager


def _key(i: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(i),
        model_name="test_model",
        kv_rank=0,
    )


def _poll(fn, timeout_s: float = 10.0):
    deadline = time.monotonic() + timeout_s
    result = fn()
    while result is None and time.monotonic() < deadline:
        time.sleep(0.02)
        result = fn()
    return result


_LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([_PAGE])], dtypes=[torch.uint8])


def _l1_values(l1: L1ManagerInterface, keys: list[ObjectKey]) -> list[int | None]:
    """Return the fill value of each key resident in ``l1``, None if absent."""
    results = l1.reserve_read(keys)
    values: list[int | None] = []
    held: list[ObjectKey] = []
    for key in keys:
        err, obj = results[key]
        if err != L1Error.SUCCESS or obj is None:
            values.append(None)
            continue
        held.append(key)
        tensor = obj.tensor
        assert tensor is not None
        assert torch.all(tensor == tensor.flatten()[0])
        values.append(int(tensor.flatten()[0]))
    l1.finish_read(held)
    return values


@dataclass
class _Peer:
    """A peer: its storage manager, the L1 peers transfer into, its URLs."""

    sm: StorageManager
    l1: L1ManagerInterface
    mq_url: str
    tc_url: str


@contextmanager
def _peer_side(
    request_transport: RequestTransport,
    *,
    l1_pages: int = 16384,
    write_ttl_seconds: int = 600,
) -> Iterator[_Peer]:
    """Stand up a peer: storage manager, NIXL context over its L1, and a
    request server hosting its P2P controller.

    Args:
        request_transport: Request transport the peer serves.
        l1_pages: Size of the peer's L1 in pages.
        write_ttl_seconds: Write TTL of the peer's L1.
    """
    peer_sm = _make_storage_manager(l1_pages * _PAGE, write_ttl_seconds)
    peer_tc_ctx = None
    request_server: RequestServer | None = None
    try:
        peer_l1 = peer_sm.get_l1_manager_by_tags(
            MemoryTags(MemorySegment.DRAM, MemoryResidency.LOCAL)
        )
        peer_l1_desc = peer_l1.get_l1_memory_desc()
        assert peer_l1_desc is not None
        peer_tc_url = _next_url()
        peer_tc_ctx = NixlTransferChannelContext(
            peer_l1_desc, listen_url=peer_tc_url, advertise_url=peer_tc_url
        )
        controller = P2PController(
            cast(MPCacheServerContext, _PeerContext(peer_sm)),
            P2PConfig(),
            CoordinatorConfig(),
            instance_id="peer",
        )
        peer_mq_url, request_server = _start_p2p_request_server(
            request_transport,
            controller,
        )
        yield _Peer(peer_sm, peer_l1, peer_mq_url, peer_tc_url)
    finally:
        if request_server is not None:
            request_server.close()
        if peer_tc_ctx is not None:
            peer_tc_ctx.close()
        peer_sm.close()


@dataclass
class _Pair:
    """A peer storage manager and a local adapter connected to it."""

    peer_sm: StorageManager
    peer_l1: L1ManagerInterface
    peer_mq_url: str
    local_buf: torch.Tensor
    adapter: P2PL2Adapter

    def local_page(self, index: int) -> MemoryObj:
        """Return page ``index`` of the local buffer as a memory object."""
        return _local_memory_obj(self.local_buf, index * _PAGE)

    def store(self, keys: list[ObjectKey], objects: list[MemoryObj]) -> L2StoreResult:
        """Store ``objects`` into the peer and wait for the task's result."""
        task_id = self.adapter.submit_store_task(keys, objects)
        result = _poll(lambda: self.adapter.pop_completed_store_tasks().get(task_id))
        assert result is not None, "the store task did not complete"
        return result

    def peer_values(self, keys: list[ObjectKey]) -> list[int | None]:
        """Return the fill value of each key in the peer's L1, None if absent."""
        return _l1_values(self.peer_l1, keys)


@contextmanager
def _p2p_pair(
    request_transport: RequestTransport,
    *,
    write_enabled: bool = False,
    peer_l1_pages: int = 16384,
    peer_write_ttl_seconds: int = 600,
    local_pages: int = 2 * _NUM_KEYS + 1,
) -> Iterator[_Pair]:
    """Stand up a peer and a local adapter over loopback NIXL.

    Args:
        request_transport: Request transport between the adapter and the peer.
        write_enabled: Whether the adapter stores into the peer.
        peer_l1_pages: Size of the peer's L1 in pages.
        peer_write_ttl_seconds: Write TTL of the peer's L1.
        local_pages: Size of the local registered buffer in pages.
    """
    local_buf = torch.zeros(local_pages * _PAGE, dtype=torch.uint8)
    with _peer_side(
        request_transport,
        l1_pages=peer_l1_pages,
        write_ttl_seconds=peer_write_ttl_seconds,
    ) as peer:
        adapter = None
        try:
            # --- Local side: global NIXL context over the local buffer ---
            local_tc_url = _next_url()
            local_l1_desc = L1MemoryDesc(
                ptr=local_buf.data_ptr(),
                size=local_buf.numel(),
                align_bytes=_PAGE,
            )
            initialize_transfer_channel_context(
                "nixl", local_l1_desc, local_tc_url, local_tc_url
            )
            adapter = P2PL2Adapter(
                P2PL2AdapterConfig(
                    peer.mq_url,
                    peer.tc_url,
                    lookup_timeout_s=10.0,
                    write_enabled=write_enabled,
                )
            )
            yield _Pair(peer.sm, peer.l1, peer.mq_url, local_buf, adapter)
        finally:
            if adapter is not None:
                adapter.close()
            delete_transfer_channel_context()


@pytest.mark.parametrize("request_transport", ["zmq", "grpc"])
def test_p2p_adapter_end_to_end(request_transport: RequestTransport) -> None:
    keys = [_key(i) for i in range(_NUM_KEYS)]

    with _p2p_pair(request_transport) as pair:
        peer_sm, adapter, local_buf = pair.peer_sm, pair.adapter, pair.local_buf

        # --- Peer side: store known objects in L1 ---
        reserved = peer_sm.reserve_write(keys, _LAYOUT)
        peer_objects: dict[ObjectKey, MemoryObj] = {}
        expected_values: dict[ObjectKey, int] = {}
        for i, key in enumerate(keys):
            peer_obj = reserved[key]
            assert peer_obj is not None
            peer_tensor = peer_obj.tensor
            assert peer_tensor is not None
            value = i + 1
            peer_tensor.fill_(value)
            peer_objects[key] = peer_obj
            expected_values[key] = value
        peer_sm.finish_write(list(reserved), l1_owners_of(reserved))

        # Lookup: every key is resident on the peer.
        lookup_id = adapter.submit_lookup_and_lock_task(keys, {0: _LAYOUT})
        bitmap = _poll(lambda: adapter.query_lookup_and_lock_result(lookup_id))
        assert bitmap is not None
        for i in range(_NUM_KEYS):
            assert bitmap.test(i) is True
        # Stashed remote addresses match the peer objects' real offsets.
        for key in keys:
            assert adapter._remote_addresses[key].offset == peer_objects[key].shm_offset

        # Load: pull each key into a distinct page of the local buffer.
        local_objs = [pair.local_page(i) for i in range(_NUM_KEYS)]
        load_id = adapter.submit_load_task(keys, local_objs)
        load_bitmap = _poll(lambda: adapter.query_load_result(load_id))
        assert load_bitmap is not None
        for i, key in enumerate(keys):
            assert load_bitmap.test(i) is True
            page = local_buf[i * _PAGE : (i + 1) * _PAGE]
            assert torch.all(page == expected_values[key]), (
                f"page {i} did not receive the peer's bytes"
            )

        # Unlock: releases the peer locks and clears the stashed addresses.
        adapter.submit_unlock(keys)
        for key in keys:
            assert key not in adapter._remote_addresses


@pytest.mark.parametrize("request_transport", ["zmq", "grpc"])
def test_p2p_store_end_to_end(request_transport: RequestTransport) -> None:
    keys = [_key(i) for i in range(_NUM_KEYS)]

    with _p2p_pair(request_transport, write_enabled=True) as pair:
        adapter, local_buf = pair.adapter, pair.local_buf

        # Store: local pages 0..N-1, each filled with its own value.
        sources = [pair.local_page(i) for i in range(_NUM_KEYS)]
        for i in range(_NUM_KEYS):
            local_buf[i * _PAGE : (i + 1) * _PAGE] = i + 1
        result = pair.store(keys, sources)
        assert result.is_successful() is True
        assert result.bytes_transferred() == _NUM_KEYS * _PAGE
        assert pair.peer_values(keys) == [1, 2, 3]

        # The stored objects are ordinary peer objects: read them back
        # through the lookup -> load path into other local pages.
        lookup_id = adapter.submit_lookup_and_lock_task(keys, {0: _LAYOUT})
        bitmap = _poll(lambda: adapter.query_lookup_and_lock_result(lookup_id))
        assert bitmap is not None
        assert bitmap.popcount() == _NUM_KEYS
        destinations = [pair.local_page(_NUM_KEYS + i) for i in range(_NUM_KEYS)]
        load_id = adapter.submit_load_task(keys, destinations)
        load_bitmap = _poll(lambda: adapter.query_load_result(load_id))
        assert load_bitmap is not None
        assert load_bitmap.popcount() == _NUM_KEYS
        for i in range(_NUM_KEYS):
            page = local_buf[(_NUM_KEYS + i) * _PAGE : (_NUM_KEYS + i + 1) * _PAGE]
            assert torch.all(page == i + 1)
        adapter.submit_unlock(keys)

        # Storing keys the peer already holds fails and leaves its bytes alone.
        local_buf[: _NUM_KEYS * _PAGE] = 99
        assert pair.store(keys, sources).is_successful() is False
        assert pair.peer_values(keys) == [1, 2, 3]


def test_p2p_store_into_a_full_peer_fails_and_leaves_it_unchanged() -> None:
    peer_pages = 8
    with _p2p_pair(
        "zmq",
        write_enabled=True,
        peer_l1_pages=peer_pages,
        local_pages=2 * peer_pages + 1,
    ) as pair:
        usage_before = pair.peer_l1.get_memory_usage()

        # More objects than the peer's L1 can hold: nothing is stored.
        too_many = [_key(i) for i in range(2 * peer_pages)]
        sources = [pair.local_page(i) for i in range(len(too_many))]
        assert pair.store(too_many, sources).is_successful() is False
        assert pair.peer_l1.get_memory_usage() == usage_before
        assert pair.peer_values(too_many) == [None] * len(too_many)

        # A store that fits still works afterwards.
        pair.local_buf[:_PAGE] = 7
        assert pair.store(too_many[:1], sources[:1]).is_successful() is True
        assert pair.peer_values(too_many[:1]) == [7]


def test_p2p_store_takes_over_an_abandoned_reservation_after_the_write_ttl() -> None:
    key = _key(0)
    with _p2p_pair("zmq", write_enabled=True, peer_write_ttl_seconds=1) as pair:
        # Another writer reserves the key on the peer and never finishes.
        other_writer = RequestClientFactory.create(
            pair.peer_mq_url, context=zmq.Context.instance()
        )
        try:
            (address,) = other_writer.p2p_reserve_write([key], _LAYOUT).result(5)
        finally:
            other_writer.close()
        assert address.is_valid()

        # While that reservation is live the key cannot be stored.
        pair.local_buf[:_PAGE] = 5
        source = [pair.local_page(0)]
        assert pair.store([key], source).is_successful() is False
        assert pair.peer_values([key]) == [None]

        # Once it expired, a new store takes the buffer over.
        time.sleep(1.2)
        assert pair.store([key], source).is_successful() is True
        assert pair.peer_values([key]) == [5]


def _write_objects(sm: StorageManager, keys: list[ObjectKey], first_value: int) -> None:
    """Write one object per key in a single store, filled with consecutive
    values starting at ``first_value``."""
    reserved = sm.reserve_write(keys, _LAYOUT)
    assert len(reserved) == len(keys)
    for i, key in enumerate(keys):
        tensor = reserved[key].tensor
        assert tensor is not None
        tensor.fill_(first_value + i)
    sm.finish_write(keys, l1_owners_of(reserved))


def test_p2p_store_policy_keeps_objects_local_until_the_usage_threshold() -> None:
    main_pages = 16
    kept = [_key(i) for i in range(4)]
    spilled = [_key(100 + i) for i in range(8)]
    spilled_values = list(range(11, 19))

    with _peer_side("zmq") as peer:
        main_sm = _make_storage_manager(main_pages * _PAGE, store_policy="p2p")
        adapter_id = None
        try:
            main_l1 = main_sm.get_l1_manager_by_tags(
                MemoryTags(MemorySegment.DRAM, MemoryResidency.LOCAL)
            )
            main_l1_desc = main_l1.get_l1_memory_desc()
            assert main_l1_desc is not None
            main_tc_url = _next_url()
            initialize_transfer_channel_context(
                "nixl", main_l1_desc, main_tc_url, main_tc_url
            )
            adapter_id = main_sm.add_l2_adapter(
                P2PL2AdapterConfig(
                    peer.mq_url,
                    peer.tc_url,
                    write_enabled=True,
                    write_l1_usage_threshold=0.5,
                )
            )

            # A quarter of the main L1: below the threshold, so these stay.
            _write_objects(main_sm, kept, first_value=1)
            assert _poll(
                lambda: (
                    main_sm.report_status()["store_controllers"]["_default"][
                        "pending_keys_count"
                    ]
                    == 0
                    or None
                )
            )
            time.sleep(0.05)

            # Three quarters used once these are written: they go to the peer
            # and leave the main L1.
            _write_objects(main_sm, spilled, first_value=spilled_values[0])
            assert _poll(
                lambda: (
                    (
                        _l1_values(peer.l1, spilled) == spilled_values
                        and _l1_values(main_l1, spilled) == [None] * len(spilled)
                    )
                    or None
                )
            ), "objects written above the threshold should move to the peer"
            assert _l1_values(main_l1, kept) == [1, 2, 3, 4]
            assert _l1_values(peer.l1, kept) == [None] * len(kept)

            # The moved objects are served back from the peer.
            handle = main_sm.submit_prefetch_task(
                single_row_spec(spilled, _LAYOUT, fetching_policy="full")
            )
            assert main_sm.wait_prefetch_status(handle, timeout=10.0)
            result = main_sm.query_prefetch_status(handle)
            assert result is not None
            assert result.l2_hit_cells[0].popcount() == len(spilled)
            with main_sm.read_prefetched_results(spilled) as objects:
                assert objects is not None
                for obj, value in zip(objects, spilled_values, strict=True):
                    tensor = obj.tensor
                    assert tensor is not None
                    assert torch.all(tensor == value)
            del objects
            main_sm.finish_read_prefetched(spilled)
        finally:
            # As the P2P controller does: the peer adapter goes away before
            # the storage manager closes.
            if adapter_id is not None:
                main_sm.delete_l2_adapter(adapter_id)
            main_sm.close()
            delete_transfer_channel_context()
