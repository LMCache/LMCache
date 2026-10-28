# SPDX-License-Identifier: Apache-2.0
"""CPU reproduction of mixed-L1 P2P using real RPC and socket byte transfers.

GDS driver initialization, transfer engine and notifier registration are substituted.
The GDS allocator, L1 locks, storage controllers, P2P adapter and RPC server are real.
This does not exercise GDS DMA, NIXL, Mooncake or a model's GPU KV layout.
"""

# Standard
from contextlib import ExitStack
from pathlib import Path
from typing import Callable, Literal, cast
from unittest.mock import MagicMock
import ctypes
import json
import socket
import threading
import time

# Third Party
import pytest
import torch
import zmq

# First Party
from lmcache.lmcache_native import PeriodicEventNotifier
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.config import parse_args
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ManagerInterface, L1MemoryDesc
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.l2_adapters.p2p_l2_adapter import P2PL2AdapterConfig
from lmcache.v1.distributed.storage_controllers.utils import resolve_l1_affinity
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.distributed.transfer_channel import (
    TransferChannelAddress,
    TransferChannelClient,
    TransferChannelContext,
    TransferChannelReadResult,
    TransferChannelServer,
    get_transfer_channel_context,
)
from lmcache.v1.distributed.transfer_channel.factory import (
    register_transfer_channel_factory,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.config import CoordinatorConfig, MPServerConfig, P2PConfig
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.p2p_controller import P2PController
from lmcache.v1.multiprocess.transport.server_factory import create_request_server
from lmcache.v1.platform import EventNotifier
from tests.v1.distributed.utils import single_row_spec
import lmcache.v1.distributed.l1_manager as l1_module
import lmcache.v1.distributed.l2_adapters.p2p_l2_adapter as adapter_module
import lmcache.v1.multiprocess.modules.p2p_controller as controller_module

_PAGE = 4096
_LAYOUT = MemoryLayoutDesc([torch.Size([_PAGE])], [torch.uint8])
_DRAM = MemoryTags(MemorySegment.DRAM, MemoryResidency.LOCAL)
_GDS = MemoryTags(MemorySegment.GDS, MemoryResidency.LOCAL)


class SocketServer(TransferChannelServer):
    """Test-only byte server exposing exactly the registered DRAM region."""

    def __init__(
        self, listen_url: str, advertise_url: str, l1_memory_desc: L1MemoryDesc
    ) -> None:
        self.memory = l1_memory_desc
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        port = self.socket.bind_to_random_port("tcp://127.0.0.1")
        self.url = f"127.0.0.1:{port}"
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def serve(self) -> None:
        """Reply to bounded offset/size reads over a real loopback socket."""
        while not self.stop.is_set():
            if not self.socket.poll(50):
                continue
            addresses = self.socket.recv_json()
            payloads = []
            for offset, size in addresses:
                assert 0 <= offset < offset + size <= self.memory.size
                payloads.append(ctypes.string_at(self.memory.ptr + offset, size))
            self.socket.send_multipart(payloads)

    def close(self) -> None:
        """Join the server before freeing its socket and registered region."""
        self.stop.set()
        self.thread.join(5)
        assert not self.thread.is_alive()
        self.socket.close(linger=0)
        self.context.term()


class SocketClient(TransferChannelClient):
    """Test-only reader that copies received bytes into registered local DRAM."""

    def __init__(self, memory: L1MemoryDesc, peer_url: str) -> None:
        self.memory = memory
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)
        self.socket.connect(f"tcp://{peer_url}")
        self.tasks: dict[int, TransferChannelReadResult] = {}
        self.next_id = 0

    def submit_read(
        self,
        local_addresses: list[TransferChannelAddress],
        remote_addresses: list[TransferChannelAddress],
    ) -> int:
        """Read bytes through TCP and return a completed task ID."""
        self.socket.send_json([(a.offset, a.size) for a in remote_addresses])
        payloads = self.socket.recv_multipart()
        for address, payload in zip(local_addresses, payloads, strict=True):
            assert (
                0 <= address.offset < address.offset + address.size <= self.memory.size
            )
            assert len(payload) == address.size
            ctypes.memmove(self.memory.ptr + address.offset, payload, address.size)
        task_id = self.next_id
        self.next_id += 1
        self.tasks[task_id] = TransferChannelReadResult(True, [True] * len(payloads))
        return task_id

    def query_read_status(self, task_id: int) -> TransferChannelReadResult:
        """Return the completed socket read's per-object status."""
        return self.tasks.pop(task_id)

    def close(self) -> None:
        """Release the requesting socket."""
        self.socket.close(linger=0)
        self.context.term()


class SocketContext(TransferChannelContext):
    """A test transport implementing the production transfer-channel contract."""

    def __init__(
        self, l1_memory_desc: L1MemoryDesc, listen_url: str, advertise_url: str
    ) -> None:
        self.memory = l1_memory_desc
        self.server = SocketServer(listen_url, advertise_url, l1_memory_desc)
        self.clients: dict[str, SocketClient] = {}

    def get_transfer_channel_server(self) -> TransferChannelServer:
        """Return the live byte server."""
        return self.server

    def get_transfer_channel_client(
        self, peer_advertise_url: str
    ) -> TransferChannelClient:
        """Connect a byte reader to a peer."""
        if peer_advertise_url not in self.clients:
            self.clients[peer_advertise_url] = SocketClient(
                self.memory, peer_advertise_url
            )
        return self.clients[peer_advertise_url]

    def remove_transfer_channel_client(self, peer_advertise_url: str) -> None:
        """Close a peer reader if connected."""
        client = self.clients.pop(peer_advertise_url, None)
        if client is not None:
            client.close()

    def get_transfer_channel_address(
        self, lmcache_addresses: list[tuple[int, int]]
    ) -> list[TransferChannelAddress]:
        """Validate destination offsets against the registered DRAM pool."""
        for offset, size in lmcache_addresses:
            assert 0 <= offset < offset + size <= self.memory.size
        return [
            TransferChannelAddress(offset, size) for offset, size in lmcache_addresses
        ]

    def get_num_connected_clients(self) -> int:
        """Return the number of socket readers."""
        return len(self.clients)

    def close(self) -> None:
        """Close all readers and the byte server."""
        for client in self.clients.values():
            client.close()
        self.clients.clear()
        self.server.close()


register_transfer_channel_factory("test-mixed-l1-socket", SocketContext)


def _storage(tmp_path: Path, gds_first: bool) -> StorageManager:
    """Construct real DRAM and GDS L1 managers without touching a GDS device."""
    specs = [
        {
            "type": "DRAM",
            "tag": "dram",
            "size_gb": 16 * _PAGE / (1 << 30),
            "use_lazy": False,
        },
        {
            "type": "GDS",
            "tag": "gds",
            "size_gb": 16 * _PAGE / (1 << 30),
            "path": str(tmp_path / "unused-gds"),
        },
    ]
    if gds_first:
        specs.reverse()
    args = ["--eviction-policy", "noop"]
    for spec in specs:
        args.extend(["--l1-manager", json.dumps(spec)])
    return StorageManager(parse_args(args))


def _store(manager: L1ManagerInterface, key: ObjectKey, value: int) -> MemoryObj:
    """Publish one object through the manager's public reservation API."""
    error, obj = manager.reserve_write([key], [False], _LAYOUT)[key]
    assert error == L1Error.SUCCESS and obj is not None
    if obj.tensor is not None:
        obj.tensor.fill_(value)
    assert manager.finish_write([key])[key] == L1Error.SUCCESS
    return obj


def _wait(check: Callable[[], bool]) -> None:
    """Wait at most five seconds for an asynchronous lock release."""
    deadline = time.monotonic() + 5
    while not check():
        assert time.monotonic() < deadline, "P2P did not release its read locks"
        time.sleep(0.01)


@pytest.mark.parametrize("gds_first", [False, True])
@pytest.mark.parametrize("request_transport", ["zmq", "grpc"])
def test_mixed_l1_p2p_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gds_first: bool,
    request_transport: Literal["zmq", "grpc"],
) -> None:
    """Initialize P2P, pull only DRAM bytes, and release only DRAM read locks."""
    monkeypatch.setattr(l1_module, "initialize_l1_gds_context", lambda *_: None)
    # Membership discovery is independent of the lookup/load/unlock data path.
    monkeypatch.setattr(controller_module, "create_periodic_thread", MagicMock())
    with ExitStack() as cleanup:
        source = _storage(tmp_path, gds_first)
        cleanup.callback(source.close)
        target = _storage(tmp_path, not gds_first)
        cleanup.callback(target.close)
        dram = source.get_l1_manager_by_memory_tags(_DRAM)
        gds = source.get_l1_manager_by_memory_tags(_GDS)
        assert dram is not None and gds is not None
        keys = [ObjectKey(ObjectKey.IntHash2Bytes(i), "mixed-p2p", 0) for i in range(4)]
        first = _store(dram, keys[0], 17)
        _store(gds, keys[1], 0)
        _store(dram, keys[2], 93)
        # Same key and same slab offset in another L1 must never be exported.
        _store(gds, keys[0], 0)
        assert gds.reserve_read([keys[0]])[keys[0]][0] == L1Error.SUCCESS

        ctx = MagicMock(spec=MPCacheServerContext)
        ctx.storage_manager = source
        controller = P2PController(
            ctx,
            P2PConfig(
                advertise_url="127.0.0.1:0", transfer_engine="test-mixed-l1-socket"
            ),
            CoordinatorConfig(),
            instance_id="source",
            request_transport=request_transport,
        )
        cleanup.callback(controller.close)
        channel = cast(SocketContext, get_transfer_channel_context())
        assert channel.memory == dram.get_l1_memory_desc()
        assert first.tensor is not None
        assert channel.memory.ptr + first.shm_offset == first.tensor.data_ptr()

        with socket.socket() as port_socket:
            port_socket.bind(("127.0.0.1", 0))
            port = port_socket.getsockname()[1]
        server = create_request_server(
            [controller],
            MPServerConfig(
                host="127.0.0.1",
                port=port,
                transport=request_transport,
                max_cpu_workers=2,
                max_gpu_workers=1,
            ),
        )
        cleanup.callback(server.close)
        server.start()
        target_dram = target.get_l1_manager_by_memory_tags(_DRAM)
        assert target_dram is not None
        target_desc = target_dram.get_l1_memory_desc()
        assert target_desc is not None
        target_channel = SocketContext(target_desc, "", "")
        cleanup.callback(target_channel.close)
        monkeypatch.setattr(
            adapter_module, "get_transfer_channel_context", lambda: target_channel
        )
        # Isolate L1 routing from the independent pipe-notification bug: the
        # notifier boundary translates poll endpoints to writable endpoints.
        event_notifiers: dict[int, EventNotifier] = {}
        create_notifier = adapter_module.create_event_notifier

        def capture_notifier() -> EventNotifier:
            notifier = create_notifier()
            event_notifiers[notifier.fileno()] = notifier
            return notifier

        native_notifier = PeriodicEventNotifier.get()
        assert native_notifier is not None
        notifier_proxy = MagicMock()
        notifier_proxy.register_fd.side_effect = lambda fd: native_notifier.register_fd(
            event_notifiers[fd].notify_fileno()
        )
        notifier_proxy.unregister_fd.side_effect = lambda fd: (
            native_notifier.unregister_fd(event_notifiers[fd].notify_fileno())
        )
        periodic = MagicMock()
        periodic.get.return_value = notifier_proxy
        monkeypatch.setattr(adapter_module, "create_event_notifier", capture_notifier)
        monkeypatch.setattr(adapter_module, "PeriodicEventNotifier", periodic)
        scheme = "tcp" if request_transport == "zmq" else "grpc"
        adapter_id = target.add_l2_adapter(
            P2PL2AdapterConfig(
                f"{scheme}://127.0.0.1:{port}",
                channel.server.url,
            )
        )
        cleanup.callback(target.delete_l2_adapter, adapter_id)
        handle = target.submit_prefetch_task(
            single_row_spec(
                keys,
                _LAYOUT,
                fetching_policy="full",
            )
        )
        assert target.wait_prefetch_status(handle, 5)
        result = target.query_prefetch_status(handle)
        assert result is not None
        assert result.hit_cells[0].get_indices_list() == [0, 2]
        assert result.l2_hit_cells[0].get_indices_list() == [0, 2]
        assert set(result.l1_owners.values()) == {target_dram.l1_manager_id}
        hit_keys = [keys[0], keys[2]]
        with target.read_prefetched_results(hit_keys, result.l1_owners) as objects:
            assert objects is not None
            for obj, value in zip(objects, [17, 93], strict=True):
                assert obj.tensor is not None and bool(torch.all(obj.tensor == value))
        target.finish_read_prefetched(hit_keys, l1_owners=result.l1_owners)
        _wait(lambda: dram.report_status()["read_locked_count"] == 0)
        # The independent GDS reader was not released by the peer's unlock.
        assert gds.report_status()["read_locked_count"] == 1
        assert gds.finish_read([keys[0]])[keys[0]] == L1Error.SUCCESS
        assert target_dram.report_status()["read_locked_count"] == 0
        assert source.memcheck() and target.memcheck()
        print(
            f"PASS {request_transport}: GDS-first={gds_first}; DRAM bytes=17,93; "
            "GDS-only/missing keys=miss; DRAM locks released; GDS lock preserved"
        )


def test_selected_l1_lookup_uses_exact_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same-tag L1 managers are distinguishable; default lookups search both."""
    specs = [
        {
            "type": "DRAM",
            "tag": tag,
            "size_gb": 16 * _PAGE / (1 << 30),
            "use_lazy": False,
        }
        for tag in ("first", "second")
    ]
    args = ["--eviction-policy", "noop"]
    for manager_spec in specs:
        args.extend(["--l1-manager", json.dumps(manager_spec)])
    config = parse_args(args)
    managers = tuple(L1Manager(c) for c in config.l1_manager_configs)
    storage = StorageManager(config, _l1_managers=managers)
    try:
        first, second = managers
        assert storage.get_l1_manager_by_memory_tags(_DRAM) is first
        assert storage.get_l1_manager_by_memory_tags(_GDS) is None
        keys = [
            ObjectKey(ObjectKey.IntHash2Bytes(i), "selected-l1", 0) for i in range(2)
        ]
        _store(first, keys[0], 12)
        _store(second, keys[1], 24)
        spec = single_row_spec(
            keys, _LAYOUT, fetching_policy="full", allowed_l2_memory_tags=frozenset()
        )
        for manager in managers:
            handle = storage.submit_prefetch_task(
                spec,
                l1_manager_id=manager.l1_manager_id,
            )
            result = storage.query_prefetch_status(handle)
            assert result is not None
            hit_keys = result.hit_cells[0].gather(keys)
            assert len(hit_keys) == 1
            assert result.l1_owners == {hit_keys[0]: manager.l1_manager_id}
            storage.finish_read_prefetched(hit_keys, l1_owners=result.l1_owners)
        handle = storage.submit_prefetch_task(spec)
        result = storage.query_prefetch_status(handle)
        assert result is not None and result.hit_cells[0].popcount() == 2
        storage.finish_read_prefetched(keys, l1_owners=result.l1_owners)
        filtered_handle = storage.submit_prefetch_task(
            single_row_spec(
                keys,
                _LAYOUT,
                fetching_policy="full",
                allowed_l1_memory_tags=frozenset({_GDS}),
                allowed_l2_memory_tags=frozenset(),
            ),
            l1_manager_id=first.l1_manager_id,
        )
        filtered_result = storage.query_prefetch_status(filtered_handle)
        assert filtered_result is not None
        assert filtered_result.hit_cells[0].popcount() == 0
        assert filtered_result.l1_owners == {}
        with pytest.raises(ValueError, match="requires empty allowed_l2_memory_tags"):
            storage.submit_prefetch_task(
                single_row_spec(keys, _LAYOUT), l1_manager_id=first.l1_manager_id
            )
        with pytest.raises(ValueError, match="Unknown L1 manager"):
            storage.submit_prefetch_task(spec, l1_manager_id=-1)
        assert all(m.report_status()["read_locked_count"] == 0 for m in managers)
        monkeypatch.setattr(first, "memory_region_count", lambda: 2)
        assert storage.get_l1_manager_by_memory_tags(_DRAM) is first
        assert (
            storage.get_l1_manager_by_memory_tags(_DRAM, memory_region_count=1)
            is second
        )
        manager_map = {manager.l1_manager_id: manager for manager in managers}
        assert (
            resolve_l1_affinity(P2PL2AdapterConfig("peer", "transfer"), manager_map)
            == second.l1_manager_id
        )
        assert (
            resolve_l1_affinity(
                MockL2AdapterConfig(max_size_gb=1, mock_bandwidth_gb=1), manager_map
            )
            == first.l1_manager_id
        )
        ctx = MagicMock()
        ctx.storage_manager = storage
        controller = P2PController(
            ctx, P2PConfig(), CoordinatorConfig(), instance_id="self"
        )
        task_id = controller.p2p_lookup_and_lock(keys, {0: _LAYOUT})
        addresses = controller.p2p_query_lookup_results(task_id)
        assert addresses is not None
        assert [address.is_valid() for address in addresses] == [False, True]
        controller.p2p_unlock_objects(keys)
        monkeypatch.setattr(second, "memory_region_count", lambda: 2)
        assert (
            storage.get_l1_manager_by_memory_tags(_DRAM, memory_region_count=1) is None
        )
        with pytest.raises(ValueError, match="No L1 has"):
            resolve_l1_affinity(P2PL2AdapterConfig("peer", "transfer"), manager_map)
    finally:
        storage.close()
