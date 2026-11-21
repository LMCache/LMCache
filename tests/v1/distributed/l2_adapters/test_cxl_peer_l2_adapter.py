# SPDX-License-Identifier: Apache-2.0
"""CXL peer adapter: borrow a peer's objects as views of its slab in a shared
pool. Two storage managers own disjoint slabs of one file-backed pool; the
owner's real P2P controller answers the borrower's lookups in process."""

# Standard
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import DaxDeviceGeometry
from lmcache.v1.common.transfer_channel.api import (
    CxlArenaDescriptor,
    TransferChannelAddress,
)
from lmcache.v1.distributed.api import (
    GroupedObjectKeys,
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
    with_shadow_l1,
)
from lmcache.v1.distributed.l2_adapters import p2p_l2_adapter as p2p_mod
from lmcache.v1.distributed.l2_adapters.cxl_peer_l2_adapter import (
    CxlPeerL2Adapter,
    CxlPeerL2AdapterConfig,
)
from lmcache.v1.distributed.memory_manager import ShadowL1MemoryManager
from lmcache.v1.distributed.storage_manager import StorageManager, l1_owners_of
from lmcache.v1.memory_allocators import devdax_memory_allocator as devdax_mod
from lmcache.v1.memory_allocators.devdax_memory_allocator import CxlPeerMapping
from lmcache.v1.memory_management import BorrowedMemoryObj
from lmcache.v1.multiprocess.config import CoordinatorConfig, P2PConfig
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules import p2p_controller
from lmcache.v1.multiprocess.modules.p2p_controller import P2PController
from lmcache.v1.multiprocess.object_group_transfer.engines import (
    TransferEngine,
    select_engine,
)

pytestmark = pytest.mark.no_shared_allocator
PAGE = 4096
LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([PAGE])], dtypes=[torch.uint8])
KEY = ObjectKey(chunk_hash=b"shared-chunk", model_name="cxl", kv_rank=0)
TTL = 300
URL = "grpc://owner:5555"


def _wait(check: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 10
    while not check():
        if time.monotonic() >= deadline:
            pytest.fail("CXL operation did not complete")
        time.sleep(0.01)


def _manager(path: Path, offset: int, *, shadow: bool = False) -> StorageManager:
    """A storage manager owning the 4-page slab at ``offset`` of the pool,
    with the CXL shadow L1 a borrower needs when ``shadow`` is set."""
    config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=PAGE * 4,
                use_lazy=False,
                shm_name="",
                devdax_path=str(path),
                cxl_pool_id="test-pool",
                devdax_offset_in_bytes=offset,
            ),
            read_ttl_seconds=TTL,
        ),
        eviction_config=EvictionConfig(eviction_policy="noop"),
    )
    if shadow:
        config = with_shadow_l1(config, MemorySegment.CXL)
    return StorageManager(config)


def _store(manager: StorageManager, key: ObjectKey = KEY, value: int = 37) -> None:
    obj = manager.reserve_write([key], LAYOUT)[key]
    assert obj is not None and obj.tensor is not None
    obj.tensor.fill_(value)
    manager.finish_write([key], l1_owners_of({key: obj}))


def _owner_holds_lock(owner: StorageManager, key: ObjectKey = KEY) -> bool:
    """Whether a read lock keeps the owner from deleting ``key``; an unlocked
    key is deleted by the probe."""
    return owner.delete_l1_keys([key]) == (0, 1)


class _OwnerClient:
    """The borrower's RPC client, answered in process by the owner's real
    P2P controller."""

    def __init__(self, controller: P2PController) -> None:
        self._controller = controller
        self.unlocked: list[ObjectKey] = []

    @staticmethod
    def done(value: object) -> Future[object]:
        future: Future[object] = Future()
        future.set_result(value)
        return future

    def p2p_lookup_and_lock(
        self, keys: list[ObjectKey], group_layout_descs: dict[int, MemoryLayoutDesc]
    ) -> Future[object]:
        return self.done(self._controller.p2p_lookup_and_lock(keys, group_layout_descs))

    def p2p_query_lookup_results(self, task_id: int) -> Future[object]:
        return self.done(self._controller.p2p_query_lookup_results(task_id))

    def p2p_unlock_objects(self, keys: list[ObjectKey]) -> Future[object]:
        self.unlocked.extend(keys)
        return self.done(self._controller.p2p_unlock_objects(keys))

    def close(self) -> None:
        pass


class _Peers(SimpleNamespace):
    borrower: StorageManager
    owner: StorageManager
    client: _OwnerClient
    path: Path


@pytest.fixture
def peers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Peers]:
    """A borrower and an owner on disjoint slabs of one pool file; every peer
    adapter created in the test talks to the owner's controller."""
    path = tmp_path / "pool.bin"
    with path.open("wb") as stream:
        stream.truncate(PAGE * 16)
    borrower = _manager(path, 0, shadow=True)
    owner = _manager(path, PAGE * 8)
    # Discovery is tested separately; replace its timer, not the handlers.
    monkeypatch.setattr(
        p2p_controller, "create_periodic_thread", lambda **_: MagicMock()
    )
    controller = P2PController(
        cast(MPCacheServerContext, SimpleNamespace(storage_manager=owner)),
        P2PConfig(transfer_engine="cxl"),
        CoordinatorConfig(url="http://coordinator"),
        instance_id="owner",
    )
    client = _OwnerClient(controller)
    monkeypatch.setattr(p2p_mod, "P2PClient", lambda *_, **__: client)
    try:
        yield _Peers(borrower=borrower, owner=owner, client=client, path=path)
    finally:
        borrower.close()
        controller.close()
        owner.close()


def _adapter(peers: _Peers, read_ttl_s: float = TTL) -> CxlPeerL2Adapter:
    local, remote = peers.borrower.cxl_arena, peers.owner.cxl_arena
    assert local is not None and remote is not None
    return CxlPeerL2Adapter(
        CxlPeerL2AdapterConfig.from_dict(
            {
                "peer_mq_server_url": URL,
                "local_device_path": str(peers.path),
                "local_arena": asdict(local),
                "peer_arena": asdict(remote),
                "read_ttl_s": read_ttl_s,
            }
        )
    )


def _lookup(adapter: CxlPeerL2Adapter, keys: list[ObjectKey]) -> list[bool]:
    """Run one lookup to completion; whether each key was a borrowable hit."""
    task_id = adapter.submit_lookup_and_lock_task(keys, {0: LAYOUT})
    deadline = time.monotonic() + 10
    while (bitmap := adapter.query_lookup_and_lock_result(task_id)) is None:
        assert time.monotonic() < deadline, "lookup did not complete"
        time.sleep(0.01)
    return [bitmap.test(i) for i in range(len(keys))]


def _load(
    adapter: CxlPeerL2Adapter, keys: list[ObjectKey]
) -> dict[ObjectKey, BorrowedMemoryObj]:
    """Run the borrower's load over fresh shadow placeholders; the bound ones."""
    _, objs = ShadowL1MemoryManager().allocate(LAYOUT, len(keys))
    bitmap = adapter.query_load_result(adapter.submit_load_task(keys, objs))
    assert bitmap is not None
    return {
        key: cast(BorrowedMemoryObj, obj)
        for i, (key, obj) in enumerate(zip(keys, objs, strict=True))
        if bitmap.test(i)
    }


def _borrow(adapter: CxlPeerL2Adapter, key: ObjectKey = KEY) -> BorrowedMemoryObj:
    assert _lookup(adapter, [key]) == [True]
    return _load(adapter, [key])[key]


def test_hit_is_bound_to_a_view_of_the_owners_bytes(peers: _Peers) -> None:
    """A lookup hit is borrowed as a local view of the owner's bytes through
    this host's own mapping, retrieved like a host-memory object; releasing
    it returns the owner's read lock."""
    _store(peers.owner)
    adapter = _adapter(peers)
    missing = ObjectKey(chunk_hash=b"missing", model_name="cxl", kv_rank=0)
    assert _lookup(adapter, [KEY, missing]) == [True, False]
    objects = _load(adapter, [KEY, missing])
    assert list(objects) == [KEY]
    obj = objects[KEY]
    assert obj.is_local and not obj.is_remote and obj.is_valid()
    assert obj.get_size() == PAGE
    assert select_engine(obj) is TransferEngine.GPU_DMA
    assert obj.tensor is not None and torch.all(obj.tensor == 37)
    assert obj.raw_tensor is not None and obj.raw_tensor.numel() == PAGE
    assert bytes(obj.byte_array) == bytes([37]) * PAGE
    _, (owned,) = peers.owner.unsafe_read([KEY])
    assert obj.data_ptr != owned.data_ptr
    assert adapter.get_local_borrow_count() == 1
    assert _owner_holds_lock(peers.owner)

    obj.release()
    obj.release()
    assert not obj.is_valid() and obj.raw_tensor is None
    assert adapter.get_local_borrow_count() == 0
    _wait(lambda: not _owner_holds_lock(peers.owner))
    assert peers.client.unlocked == [KEY]
    adapter.close()


def test_borrowing_uses_no_local_capacity(peers: _Peers) -> None:
    """A prefetch borrows through the shadow L1 with the borrower's own slab
    full, and the last reader's finish returns the owner's lock."""
    borrower, owner = peers.borrower, peers.owner
    for i in range(4):
        _store(borrower, ObjectKey(f"local-{i}".encode(), "cxl", 0))
    used_before = borrower.get_l1_usage()
    assert used_before == (4 * PAGE, 4 * PAGE)
    _store(owner)
    assert owner.cxl_arena is not None
    borrower.add_cxl_peer(owner.cxl_arena, URL, 10)
    handle = borrower.submit_prefetch_task(
        PrefetchTaskSpec(
            key_groups=[
                GroupedObjectKeys(
                    keys=[KEY], object_group_id=0, layout_desc=LAYOUT, num_kv_readers=2
                )
            ],
        )
    )
    assert borrower.wait_prefetch_status(handle, timeout=10)
    result = borrower.query_prefetch_status(handle)
    assert result is not None and result.hit_cells[0].test(0)
    assert borrower.get_l1_usage() == used_before
    assert KEY in result.l1_owners
    with borrower.read_prefetched_results([KEY], result.l1_owners) as objects:
        assert objects is not None
        view = objects[0]
        assert isinstance(view, BorrowedMemoryObj) and view.is_local
        assert view.tensor is not None and torch.all(view.tensor == 37)
    borrower.finish_read_prefetched([KEY], l1_owners=result.l1_owners)
    assert _owner_holds_lock(owner)
    borrower.finish_read_prefetched([KEY], l1_owners=result.l1_owners)
    _wait(lambda: not _owner_holds_lock(owner))
    assert borrower.get_l1_usage() == used_before


def test_each_borrow_holds_its_own_owner_lock(peers: _Peers) -> None:
    """Two lookups of a key hold two owner locks; each view returns one."""
    _store(peers.owner)
    adapter = _adapter(peers)
    first, second = _borrow(adapter), _borrow(adapter)
    assert first.same_bytes(second)
    first.release()
    _wait(lambda: len(peers.client.unlocked) == 1)
    assert second.is_valid() and _owner_holds_lock(peers.owner)
    second.release()
    _wait(lambda: not _owner_holds_lock(peers.owner))
    adapter.close()


def test_unborrowed_hit_is_returned_by_unlock(peers: _Peers) -> None:
    """A hit the prefetch controller does not load goes back at its unlock."""
    _store(peers.owner)
    adapter = _adapter(peers)
    assert _lookup(adapter, [KEY]) == [True]
    adapter.submit_unlock([KEY])
    _wait(lambda: not _owner_holds_lock(peers.owner))
    assert _load(adapter, [KEY]) == {}
    adapter.close()


def test_hit_outside_the_mapped_slab_is_refused_and_unlocked(
    peers: _Peers, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An address in another slab is never borrowed: the lookup reports a
    miss and the owner's lock is returned at once."""
    _store(peers.owner)
    adapter = _adapter(peers)
    remote = peers.owner.cxl_arena
    assert remote is not None
    foreign = replace(remote, session_id="another-incarnation")
    answer = peers.client.p2p_query_lookup_results

    def answer_from_another_slab(task_id: int) -> Future[object]:
        addresses = cast(list[TransferChannelAddress] | None, answer(task_id).result())
        if addresses is None:
            return peers.client.done(None)
        return peers.client.done(
            [replace(address, cxl_arena=foreign) for address in addresses]
        )

    monkeypatch.setattr(
        peers.client, "p2p_query_lookup_results", answer_from_another_slab
    )
    assert _lookup(adapter, [KEY]) == [False]
    _wait(lambda: not _owner_holds_lock(peers.owner))
    assert _load(adapter, [KEY]) == {}
    adapter.close()


def test_view_ends_before_the_owners_ttl(
    peers: _Peers, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A view stops being valid once less than the read window is left
    before the owner's read lock can lapse (TTL minus a margin), so a GPU
    copy queued from it has that long to finish. Released after the lapse,
    it sends no unlock: the owner's TTL covers it, and a late unlock could
    release a newer borrow."""
    _store(peers.owner)
    # The lock may lapse at 30 - 1 s margin; a copy needs 5 s of it left.
    adapter = _adapter(peers, read_ttl_s=30)
    obj = _borrow(adapter)
    now = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: now + 23)
    assert obj.is_valid()
    monkeypatch.setattr(time, "monotonic", lambda: now + 24.5)
    assert not obj.is_valid()
    monkeypatch.setattr(time, "monotonic", lambda: now + 29.5)
    obj.release()
    assert peers.client.unlocked == []
    adapter.close()


def test_owner_restart_ends_every_lease(peers: _Peers) -> None:
    """An owner that closes its slab clears the header, so views of the old
    incarnation are invalid at once and its hits are no longer borrowed."""
    _store(peers.owner)
    adapter = _adapter(peers)
    obj = _borrow(adapter)
    assert adapter.report_status()["is_healthy"]
    # What the owner's close does to its slab header.
    with peers.path.open("r+b") as stream:
        stream.seek(PAGE * 8)
        stream.write(bytes(32))
    assert not obj.is_valid()
    assert not adapter.report_status()["is_healthy"]
    assert _lookup(adapter, [KEY]) == [False]
    obj.release()
    adapter.close()


def test_close_refuses_while_views_are_live_then_drains(peers: _Peers) -> None:
    _store(peers.owner)
    adapter = _adapter(peers)
    obj = _borrow(adapter)
    # A second hit is reserved but never loaded: close returns it.
    assert _lookup(adapter, [KEY]) == [True]
    with pytest.raises(RuntimeError, match="live borrowed objects"):
        adapter.close()
    assert adapter.report_status()["local_borrows"] == 1
    assert obj.is_valid() and torch.all(cast(torch.Tensor, obj.tensor) == 37)
    obj.release()
    adapter.close()
    adapter.close()
    assert peers.client.unlocked == [KEY, KEY]
    assert not _owner_holds_lock(peers.owner)


def test_close_is_retried_after_a_lingering_tensor_is_dropped(peers: _Peers) -> None:
    """A tensor a reader still holds keeps the slab mapped; the refused close
    succeeds once it is dropped."""
    _store(peers.owner)
    adapter = _adapter(peers)
    obj = _borrow(adapter)
    lingering = obj.tensor
    obj.release()
    with pytest.raises(BufferError):
        adapter.close()
    del lingering
    adapter.close()
    assert adapter.get_local_borrow_count() == 0


def test_config_validates_peer_identity(peers: _Peers) -> None:
    """Pool identity, slab overlap, timeouts and the mapped incarnation are
    all checked before a peer is served."""
    local, remote = peers.borrower.cxl_arena, peers.owner.cxl_arena
    assert local is not None and remote is not None
    path = str(peers.path)
    assert CxlArenaDescriptor.from_json(remote.to_json()) == remote
    with pytest.raises(ValueError, match="same pool"):
        CxlPeerL2AdapterConfig(URL, path, local, replace(remote, pool_id="other"))
    with pytest.raises(ValueError, match="disjoint"):
        CxlPeerL2AdapterConfig(URL, path, local, replace(remote, offset=0))
    with pytest.raises(ValueError, match="read_ttl_s"):
        CxlPeerL2AdapterConfig(URL, path, local, remote, read_ttl_s=0)
    with pytest.raises(ValueError, match="read_window_s"):
        CxlPeerL2AdapterConfig(URL, path, local, remote, read_window_s=0)
    with pytest.raises(ValueError, match="shorter than read_ttl_s"):
        CxlPeerL2AdapterConfig(
            URL, path, local, remote, read_ttl_s=10, read_window_s=10
        )
    with pytest.raises(ValueError, match="peer_mq_server_url"):
        CxlPeerL2AdapterConfig.from_dict({"local_device_path": path})
    config = CxlPeerL2AdapterConfig(URL, path, local, remote)
    assert config.get_wanted_l1_memory_tags() == (
        MemoryTags(MemorySegment.CXL, MemoryResidency.SHADOW),
    )
    with pytest.raises(ValueError, match="header"):
        CxlPeerMapping(path, replace(remote, session_id="stale"))
    mapping = CxlPeerMapping(path, remote)
    try:
        with pytest.raises(ValueError, match="range"):
            mapping.view(remote.size, PAGE)
        with pytest.raises(ValueError, match="range"):
            mapping.view(1, PAGE)
    finally:
        mapping.close()


def test_owner_close_clears_slab_identity(tmp_path: Path) -> None:
    """Borrowers stop trusting a slab as soon as its owner closes it."""
    path = tmp_path / "pool.bin"
    path.write_bytes(bytes(PAGE * 8))
    owner = _manager(path, 0)
    assert owner.cxl_arena is not None
    mapping = CxlPeerMapping(str(path), owner.cxl_arena)
    try:
        assert mapping.is_current()
        owner.close()
        assert not mapping.is_current()
    finally:
        mapping.close()


def test_slab_header_must_match_device_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A header shorter than the device alignment fails with the fix to apply."""
    path = tmp_path / "pool.bin"
    path.write_bytes(bytes(PAGE * 8))
    owner = _manager(path, 0)
    try:
        assert owner.cxl_arena is not None
        monkeypatch.setattr(
            devdax_mod,
            "get_dax_device_geometry",
            lambda _: DaxDeviceGeometry(capacity=PAGE * 8, align=2 << 20),
        )
        with pytest.raises(RuntimeError, match="--l1-align-bytes 2097152"):
            CxlPeerMapping(str(path), owner.cxl_arena)
    finally:
        owner.close()
