# SPDX-License-Identifier: Apache-2.0
"""CPU tests for P2P pool selection, data placement and lock ownership.

Storage managers, P2P controllers and adapters are real. RPC calls dispatch
in-process and the transfer boundary copies bytes between registered DRAM
buffers. GDS initialization and peer discovery are stubbed; network protocols
and native transfer engines are covered by their own integration tests.
"""

# Standard
from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import ExitStack
from pathlib import Path
from typing import TypeVar
from unittest.mock import MagicMock
import ctypes
import json

# Third Party
import pytest
import torch

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
from lmcache.v1.distributed.internal_api import L1ManagerInterface
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.l2_adapters.p2p_l2_adapter import P2PL2AdapterConfig
from lmcache.v1.distributed.storage_controllers.utils import resolve_l1_affinity
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.distributed.transfer_channel import (
    TransferChannelAddress,
    TransferChannelContext,
    TransferChannelReadResult,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.config import CoordinatorConfig, P2PConfig
from lmcache.v1.multiprocess.modules.p2p_controller import P2PController
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.platform import EventNotifier
from tests.v1.distributed.utils import single_row_spec
import lmcache.v1.distributed.l1_manager as l1_module
import lmcache.v1.distributed.l2_adapters.p2p_l2_adapter as adapter_module
import lmcache.v1.multiprocess.modules.p2p_controller as controller_module

_PAGE = 4096
_LAYOUT = MemoryLayoutDesc([torch.Size([_PAGE])], [torch.uint8])
_DRAM = MemoryTags(MemorySegment.DRAM, MemoryResidency.LOCAL)
_GDS = MemoryTags(MemorySegment.GDS, MemoryResidency.LOCAL)
_T = TypeVar("_T")


def _completed(value: _T) -> Future[_T]:
    """Wrap a synchronous controller result in the RPC client's future shape."""
    future: Future[_T] = Future()
    future.set_result(value)
    return future


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


@pytest.mark.parametrize("gds_first", [False, True])
def test_mixed_l1_p2p_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gds_first: bool,
) -> None:
    """Initialize P2P, pull only DRAM bytes, and release only DRAM read locks."""
    monkeypatch.setattr(l1_module, "initialize_l1_gds_context", lambda *_: None)
    # Membership discovery is independent of the lookup/load/unlock data path.
    monkeypatch.setattr(controller_module, "create_periodic_thread", MagicMock())
    register = MagicMock()
    monkeypatch.setattr(
        controller_module, "initialize_transfer_channel_context", register
    )
    monkeypatch.setattr(
        controller_module, "delete_transfer_channel_context", MagicMock()
    )
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
        # Another reader of the same key in GDS must keep its own read lock.
        _store(gds, keys[0], 0)
        assert gds.reserve_read([keys[0]])[keys[0]][0] == L1Error.SUCCESS

        ctx = MagicMock()
        ctx.storage_manager = source
        controller = P2PController(
            ctx,
            P2PConfig(advertise_url="127.0.0.1:7600"),
            CoordinatorConfig(),
            instance_id="source",
        )
        cleanup.callback(controller.close)
        source_desc = register.call_args.args[1]
        assert source_desc == dram.get_l1_memory_desc()
        assert first.tensor is not None
        assert source_desc.ptr + first.shm_offset == first.tensor.data_ptr()
        target_dram = target.get_l1_manager_by_memory_tags(_DRAM)
        assert target_dram is not None
        target_desc = target_dram.get_l1_memory_desc()
        assert target_desc is not None

        client = MagicMock(spec=RequestClient)
        client.p2p_lookup_and_lock.side_effect = lambda keys, layouts: _completed(
            controller.p2p_lookup_and_lock(keys, layouts)
        )
        client.p2p_query_lookup_results.side_effect = lambda task: _completed(
            controller.p2p_query_lookup_results(task)
        )
        client.p2p_unlock_objects.side_effect = lambda keys: _completed(
            controller.p2p_unlock_objects(keys)
        )
        monkeypatch.setattr(
            adapter_module.RequestClientFactory, "create", lambda *a, **kw: client
        )

        def copy_bytes(
            local: list[TransferChannelAddress], remote: list[TransferChannelAddress]
        ) -> int:
            for dst, src in zip(local, remote, strict=True):
                assert dst.size == src.size
                assert 0 <= src.offset < src.offset + src.size <= source_desc.size
                assert 0 <= dst.offset < dst.offset + dst.size <= target_desc.size
                ctypes.memmove(
                    target_desc.ptr + dst.offset, source_desc.ptr + src.offset, src.size
                )
            transfer.query_read_status.return_value = TransferChannelReadResult(
                True, [True] * len(local)
            )
            return 0

        channel = MagicMock(spec=TransferChannelContext)
        channel.get_transfer_channel_address.side_effect = lambda pairs: [
            TransferChannelAddress(offset, size) for offset, size in pairs
        ]
        transfer = channel.get_transfer_channel_client.return_value
        transfer.submit_read.side_effect = copy_bytes
        monkeypatch.setattr(
            adapter_module, "get_transfer_channel_context", lambda: channel
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
        adapter_id = target.add_l2_adapter(
            P2PL2AdapterConfig("tcp://peer:5555", "peer:7600")
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
        assert dram.report_status()["read_locked_count"] == 0
        # The independent GDS reader was not released by the peer's unlock.
        assert gds.report_status()["read_locked_count"] == 1
        assert gds.finish_read([keys[0]])[keys[0]] == L1Error.SUCCESS
        assert target_dram.report_status()["read_locked_count"] == 0
        assert source.memcheck() and target.memcheck()


@pytest.fixture
def dram_storage() -> Iterator[tuple[StorageManager, tuple[L1Manager, ...]]]:
    """Provide two real DRAM pools with one distinct key in each."""
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
        for i, manager in enumerate(managers):
            _store(
                manager, ObjectKey(ObjectKey.IntHash2Bytes(i), "selected-l1", 0), i + 12
            )
        yield storage, managers
    finally:
        storage.close()


def test_selected_l1_lookup_uses_exact_manager(
    dram_storage: tuple[StorageManager, tuple[L1Manager, ...]],
) -> None:
    """Manager selection restricts hits while preserving filters and lock ownership."""
    storage, managers = dram_storage
    first = managers[0]
    keys = [ObjectKey(ObjectKey.IntHash2Bytes(i), "selected-l1", 0) for i in range(2)]
    spec = single_row_spec(
        keys, _LAYOUT, fetching_policy="full", allowed_l2_memory_tags=frozenset()
    )
    for key, manager in zip(keys, managers, strict=True):
        handle = storage.submit_prefetch_task(
            spec,
            l1_manager_id=manager.l1_manager_id,
        )
        result = storage.query_prefetch_status(handle)
        assert result is not None
        hit_keys = result.hit_cells[0].gather(keys)
        assert hit_keys == [key]
        assert result.l1_owners == {key: manager.l1_manager_id}
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


def test_p2p_selection_skips_multi_region_dram(
    dram_storage: tuple[StorageManager, tuple[L1Manager, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pool selection, lookup and adapter affinity agree on eligible DRAM."""
    storage, managers = dram_storage
    first, second = managers
    keys = [ObjectKey(ObjectKey.IntHash2Bytes(i), "selected-l1", 0) for i in range(2)]
    assert storage.get_l1_manager_by_memory_tags(_DRAM) is first
    assert storage.get_l1_manager_by_memory_tags(_GDS) is None
    monkeypatch.setattr(first, "memory_region_count", lambda: 2)
    assert storage.get_l1_manager_by_memory_tags(_DRAM) is first
    assert storage.get_l1_manager_by_memory_tags(_DRAM, memory_region_count=1) is second
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
    assert storage.get_l1_manager_by_memory_tags(_DRAM, memory_region_count=1) is None
    with pytest.raises(ValueError, match="No L1 has"):
        resolve_l1_affinity(P2PL2AdapterConfig("peer", "transfer"), manager_map)
