# SPDX-License-Identifier: Apache-2.0
"""In-process, real-NIXL integration test for GPU-direct borrowing.

Stands up an owner (a real ``StorageManager`` with objects in its pinned DRAM
L1, a NIXL transfer-channel context over that L1, and a request server hosting
a ``P2PController``) and a borrower (the global NIXL context, a GPU landing
ring registered as a device region, a shadow L1 paired with a DRAM L1, and a
``GdrPeerL2Adapter``). It drives lookup -> borrow -> admit -> read straight
into GPU memory -> release -> unlock and verifies the GPU bytes match the
owner's. Requires a working NIXL runtime and CUDA; skipped otherwise.
"""

# Standard
from typing import Literal, cast
import itertools
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )

nixl = pytest.importorskip("nixl")

# First Party
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
from lmcache.v1.distributed.internal_api import L1MemoryDesc  # noqa: E402
from lmcache.v1.distributed.l1_manager import ShadowL1Manager  # noqa: E402
from lmcache.v1.distributed.l1_manager.base import BaseL1Manager  # noqa: E402
from lmcache.v1.distributed.l2_adapters.gdr_peer_l2_adapter import (  # noqa: E402
    GdrPeerL2Adapter,
    GdrPeerL2AdapterConfig,
    TransferChannelRegionRegistrar,
)
from lmcache.v1.distributed.storage_manager import (  # noqa: E402
    StorageManager,
    l1_owners_of,
)
from lmcache.v1.distributed.transfer_channel import (  # noqa: E402
    delete_transfer_channel_context,
    get_transfer_channel_context,
    initialize_transfer_channel_context,
)
from lmcache.v1.distributed.transfer_channel.impl.nixl_impl import (  # noqa: E402
    NixlTransferChannelContext,
)
from lmcache.v1.gpu_connector.gdr_context import GdrLandingRing  # noqa: E402
from lmcache.v1.memory_management import (  # noqa: E402
    MemoryObj,
    RemoteBorrowedMemoryObj,
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

_PAGE = 4096
_NUM_KEYS = 3
_READ_TTL = 300
_port_counter = itertools.count(18400)
RequestTransport = Literal["zmq", "grpc"]


def _next_url() -> str:
    return f"127.0.0.1:{next(_port_counter)}"


def _start_p2p_request_server(
    transport: RequestTransport, controller: P2PController
) -> tuple[str, RequestServer]:
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


def _make_storage_manager(size_bytes: int) -> StorageManager:
    memory_config = L1MemoryManagerConfig(
        size_in_bytes=size_bytes,
        use_lazy=False,
        init_size_in_bytes=size_bytes,
        align_bytes=_PAGE,
    )
    l1_config = L1ManagerConfig(
        memory_config=memory_config,
        write_ttl_seconds=600,
        read_ttl_seconds=_READ_TTL,
    )
    return StorageManager(
        StorageManagerConfig(
            l1_manager_config=l1_config,
            eviction_config=EvictionConfig(eviction_policy="LRU"),
        )
    )


class _PeerContext:
    """Minimal P2PController context -- only ``storage_manager`` is used."""

    def __init__(self, storage_manager: StorageManager) -> None:
        self.storage_manager = storage_manager


def _key(i: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(i), model_name="test_model", kv_rank=0
    )


def _poll(fn, timeout_s: float = 10.0):
    deadline = time.monotonic() + timeout_s
    result = fn()
    while result is None and time.monotonic() < deadline:
        time.sleep(0.02)
        result = fn()
    return result


def _wait_until(predicate, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.02)


@pytest.mark.parametrize("request_transport", ["zmq", "grpc"])
def test_gdr_borrow_reads_owner_bytes_straight_into_gpu(
    request_transport: RequestTransport,
) -> None:
    keys = [_key(i) for i in range(_NUM_KEYS)]
    layout = MemoryLayoutDesc(shapes=[torch.Size([_PAGE])], dtypes=[torch.uint8])

    owner_sm = _make_storage_manager(64 * 1024 * 1024)
    owner_tc_ctx = None
    request_server: RequestServer | None = None
    adapter = None
    ring = None
    shadow = None
    borrower_sm = _make_storage_manager(4 * _PAGE)
    device = torch.device(f"{torch_device_type}:0")

    try:
        # --- Owner: store known objects in L1 ---
        reserved = owner_sm.reserve_write(keys, layout)
        expected_values: dict[ObjectKey, int] = {}
        owner_objects: dict[ObjectKey, MemoryObj] = {}
        for i, key in enumerate(keys):
            obj = reserved[key]
            assert obj is not None and obj.tensor is not None
            obj.tensor.fill_(i + 1)
            expected_values[key] = i + 1
            owner_objects[key] = obj
        owner_sm.finish_write(list(reserved), l1_owners_of(reserved))

        owner_l1 = owner_sm.get_l1_manager_by_tags(
            MemoryTags(MemorySegment.DRAM, MemoryResidency.LOCAL)
        )
        owner_desc = owner_l1.get_l1_memory_desc()
        assert owner_desc is not None
        owner_tc_url = _next_url()
        owner_tc_ctx = NixlTransferChannelContext(
            owner_desc, listen_url=owner_tc_url, advertise_url=owner_tc_url
        )
        controller = P2PController(
            cast(MPCacheServerContext, _PeerContext(owner_sm)),
            P2PConfig(),
            CoordinatorConfig(),
            instance_id="owner",
        )
        owner_rpc_url, request_server = _start_p2p_request_server(
            request_transport, controller
        )

        # --- Borrower: NIXL over its own DRAM L1, a GPU landing ring, a shadow ---
        borrower_l1 = borrower_sm.get_l1_manager_by_tags(
            MemoryTags(MemorySegment.DRAM, MemoryResidency.LOCAL)
        )
        borrower_desc = borrower_l1.get_l1_memory_desc()
        assert borrower_desc is not None
        local_tc_url = _next_url()
        initialize_transfer_channel_context(
            "nixl",
            cast(L1MemoryDesc, borrower_desc),
            local_tc_url,
            local_tc_url,
        )
        registrar = TransferChannelRegionRegistrar(
            get_transfer_channel_context(), _PAGE
        )
        ring = GdrLandingRing(device, _PAGE, slots_per_bank=2, registrar=registrar)
        shadow = ShadowL1Manager.next_to(cast(BaseL1Manager, borrower_l1))

        adapter = GdrPeerL2Adapter(
            GdrPeerL2AdapterConfig(owner_rpc_url, owner_tc_url, lookup_timeout_s=10.0)
        )

        # Lookup: every key is resident on the owner and reports its TTL.
        lookup_id = adapter.submit_lookup_and_lock_task(keys, {0: layout})
        bitmap = _poll(lambda: adapter.query_lookup_and_lock_result(lookup_id))
        assert bitmap is not None
        assert all(bitmap.test(i) for i in range(_NUM_KEYS))

        # Borrow + admit: the shadow indexes remote handles, no L1 bytes move.
        handles = adapter.take_borrowed_objects(lookup_id, keys, {0: layout})
        adapter.release_lookup(lookup_id, keys)
        assert set(handles) == set(keys)
        results = shadow.admit(handles, read_locks=1)
        assert all(results[key][0] == L1Error.SUCCESS for key in keys)
        assert borrower_sm.get_l1_usage()[0] == 0
        assert adapter.get_active_borrow_count() == _NUM_KEYS

        # Read each borrowed object straight into a GPU landing slot.
        ring.acquire(0)
        for i, key in enumerate(keys):
            handle = results[key][1]
            assert isinstance(handle, RemoteBorrowedMemoryObj)
            slot = ring.slot(i % 2, i // 2)
            read = handle.read_into(slot.region_id, slot.offset)
            _wait_until(read.is_finished)
            assert read.succeeded()
            assert slot.buffer.device.type == torch_device_type
            assert torch.all(slot.buffer.cpu() == expected_values[key]), (
                f"GPU slot for key {i} did not receive the owner's bytes"
            )

        # Release: the last finish_read frees the handle, which unlocks the owner.
        shadow.finish_read(keys)
        _wait_until(lambda: adapter.get_active_borrow_count() == 0)
        _wait_until(lambda: controller.report_status()["active_p2p_lookup_jobs"] == 0)
        # The owner's read locks are gone: its L1 can delete the objects again.
        assert all(err == L1Error.SUCCESS for err in owner_l1.delete(keys).values())
    finally:
        if shadow is not None:
            shadow.close()
        if adapter is not None:
            adapter.close()
        if ring is not None:
            ring.close()
        if request_server is not None:
            request_server.close()
        delete_transfer_channel_context()
        if owner_tc_ctx is not None:
            owner_tc_ctx.close()
        borrower_sm.close()
        owner_sm.close()
