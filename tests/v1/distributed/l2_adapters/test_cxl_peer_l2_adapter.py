# SPDX-License-Identifier: Apache-2.0
"""Shared-file CXL tests using real allocators, prefetch, and peer RPCs."""

# Standard
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast
from unittest.mock import MagicMock
import socket
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import (
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.cxl_types import CxlArenaDescriptor
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.cxl_peer_l2_adapter import (
    CxlPeerL2Adapter,
    CxlPeerL2AdapterConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.memory_allocators.devdax_memory_allocator import CxlPeerMapping
from lmcache.v1.memory_management import CXLMemoryObj
from lmcache.v1.multiprocess.config import CoordinatorConfig, MPServerConfig, P2PConfig
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules import p2p_controller
from lmcache.v1.multiprocess.modules.p2p_controller import P2PController
from lmcache.v1.multiprocess.transport.server_factory import create_request_server

pytestmark = pytest.mark.no_shared_allocator
PAGE = 4096
LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([PAGE])], dtypes=[torch.uint8])
KEY = ObjectKey(chunk_hash=b"shared-chunk", model_name="cxl", kv_rank=0)


def _wait(check: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 10
    while not check():
        if time.monotonic() >= deadline:
            pytest.fail("CXL operation did not complete")
        time.sleep(0.01)


def _manager(path: Path, offset: int, ttl: int = 300) -> StorageManager:
    return StorageManager(
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=PAGE * 4,
                    use_lazy=False,
                    shm_name="",
                    devdax_path=str(path),
                    cxl_pool_id="test-pool",
                    cxl_pool_offset=offset,
                ),
                read_ttl_seconds=ttl,
            ),
            eviction_config=EvictionConfig(eviction_policy="noop"),
        )
    )


def _store(manager: StorageManager, key: ObjectKey = KEY) -> None:
    obj = manager.reserve_write([key], LAYOUT)[key]
    assert obj is not None and obj.tensor is not None
    obj.tensor.fill_(37)
    manager.finish_write([key])


@pytest.fixture
def owner_ttl() -> int:
    """Owner TTL shared by the real RPC fixtures."""
    return 300


@pytest.fixture(params=["zmq", "grpc"])
def peers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    owner_ttl: int,
) -> Iterator[tuple[StorageManager, StorageManager, str, Path]]:
    """Run two disjoint slabs and the owner's real P2P request server."""
    path = tmp_path / "pool.bin"
    with path.open("wb") as stream:
        stream.truncate(PAGE * 16)
    borrower = _manager(path, 0)
    owner = _manager(path, PAGE * 8, ttl=owner_ttl)
    # Discovery is tested separately; replace its timer, not the lookup handler.
    monkeypatch.setattr(
        p2p_controller, "create_periodic_thread", lambda **_: MagicMock()
    )
    controller = P2PController(
        cast(MPCacheServerContext, SimpleNamespace(storage_manager=owner)),
        P2PConfig(transfer_engine="cxl"),
        CoordinatorConfig(url="http://coordinator"),
        instance_id="owner",
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    transport = cast(Literal["zmq", "grpc"], request.param)
    server = create_request_server(
        [controller],
        MPServerConfig(
            transport=transport,
            host="127.0.0.1",
            port=port,
            max_cpu_workers=2,
            max_gpu_workers=1,
        ),
    )
    server.start()
    scheme = "tcp" if transport == "zmq" else "grpc"
    try:
        yield borrower, owner, f"{scheme}://127.0.0.1:{port}", path
    finally:
        borrower.close()
        server.close()
        controller.close()
        owner.close()


def _adapter(
    borrower: StorageManager,
    owner: StorageManager,
    url: str,
    path: Path,
) -> CxlPeerL2Adapter:
    assert borrower.cxl_arena is not None and owner.cxl_arena is not None
    return CxlPeerL2Adapter(
        CxlPeerL2AdapterConfig(
            url,
            str(path),
            borrower.cxl_arena,
            owner.cxl_arena,
        )
    )


def _borrow(adapter: CxlPeerL2Adapter, key: ObjectKey = KEY) -> CXLMemoryObj:
    task = adapter.submit_lookup_and_lock_task([key], {0: LAYOUT})
    result = adapter.query_lookup_and_lock_result(task)
    assert result is not None and result.test(0)
    objects = adapter.take_borrowed_objects(task, [key], {0: LAYOUT})
    obj = objects[key]
    assert isinstance(obj, CXLMemoryObj)
    return obj


def test_prefetch_borrows_with_full_local_slab(peers: tuple) -> None:
    """Borrowing succeeds at zero free payload capacity and retains owner data."""
    borrower, owner, url, _ = peers
    for i in range(4):
        _store(borrower, replace(KEY, chunk_hash=f"local-{i}".encode()))
    used_before = borrower.get_l1_usage()
    assert used_before == (4 * PAGE, 4 * PAGE)
    _store(owner)
    assert owner.cxl_arena is not None
    borrower.add_cxl_peer(owner.cxl_arena, url, 10)
    task = borrower.submit_prefetch_task(
        PrefetchTaskSpec(
            key_groups=[
                GroupedObjectKeys(keys=[KEY], object_group_id=0, layout_desc=LAYOUT)
            ],
            num_kv_readers=2,
        )
    )
    assert borrower.wait_prefetch_status(task, timeout=10)
    result = borrower.query_prefetch_status(task)
    assert result is not None and result.hit_cells[0].test(0)
    assert borrower.get_l1_usage() == used_before
    assert owner.delete_l1_keys([KEY]) == (0, 1)
    with borrower.read_prefetched_results([KEY]) as objects:
        shadow = objects[0]
        assert isinstance(shadow, CXLMemoryObj)
        assert shadow.tensor is not None
        owner_keys, owner_objects = owner.unsafe_read([KEY])
        assert owner_keys == [KEY]
        assert shadow.data_ptr != owner_objects[0].data_ptr
        assert torch.all(shadow.tensor == 37)
        if torch_dev.is_available():
            gpu_copy = shadow.tensor.to(torch_device_type, non_blocking=True)
            torch_dev.synchronize()
            assert torch.all(gpu_copy.cpu() == 37)
    borrower.finish_read_prefetched([KEY])
    assert owner.delete_l1_keys([KEY]) == (0, 1)
    borrower.finish_read_prefetched([KEY])
    _wait(lambda: owner.delete_l1_keys([KEY]) == (1, 0))
    assert borrower.get_l1_usage() == used_before


def test_independent_borrows_and_idempotent_release(peers: tuple) -> None:
    """Overlapping lookups for the same key each own one remote reservation."""
    borrower, owner, url, path = peers
    _store(owner)
    adapter = _adapter(borrower, owner, url, path)
    try:
        first, second = _borrow(adapter), _borrow(adapter)
        assert adapter.get_active_borrow_count() == 2
        first.release()
        first.release()
        _wait(lambda: adapter.get_active_borrow_count() == 1)
        assert owner.delete_l1_keys([KEY]) == (0, 1)
        second.release()
        _wait(lambda: adapter.get_active_borrow_count() == 0)
        assert owner.delete_l1_keys([KEY]) == (1, 0)
    finally:
        adapter.close()


def test_unused_lookup_and_invalid_layout_release(peers: tuple) -> None:
    """An unused hit can be unlocked without constructing a shadow."""
    borrower, owner, url, path = peers
    _store(owner)
    adapter = _adapter(borrower, owner, url, path)
    try:
        task = adapter.submit_lookup_and_lock_task([KEY], {0: LAYOUT})
        found = adapter.query_lookup_and_lock_result(task)
        assert found is not None and found.test(0)
        bad_layout = MemoryLayoutDesc([torch.Size([PAGE // 2])], [torch.uint8])
        assert adapter.take_borrowed_objects(task, [KEY], {0: bad_layout}) == {}
        adapter.release_lookup(task, [KEY])
        adapter.release_lookup(task, [KEY])
        _wait(lambda: adapter.get_active_borrow_count() == 0)
        assert owner.delete_l1_keys([KEY]) == (1, 0)
    finally:
        adapter.close()


def test_peer_close_waits_for_shadow(peers: tuple) -> None:
    """Adapter teardown cannot unmap a view still held by L1/GPU readers."""
    borrower, owner, url, path = peers
    _store(owner)
    adapter = _adapter(borrower, owner, url, path)
    view = _borrow(adapter)
    with pytest.raises(RuntimeError, match="live shadows"):
        adapter.close()
    view.release()
    adapter.close()
    assert owner.delete_l1_keys([KEY]) == (1, 0)


def test_peer_identity_and_range_validation(peers: tuple) -> None:
    """Pool identity, overlap, and the actual mapped incarnation are checked."""
    borrower, owner, url, path = peers
    local, remote = borrower.cxl_arena, owner.cxl_arena
    assert local is not None and remote is not None
    assert CxlArenaDescriptor.from_json(remote.to_json()) == remote
    with pytest.raises(ValueError, match="same pool"):
        CxlPeerL2AdapterConfig(url, str(path), local, replace(remote, pool_id="other"))
    with pytest.raises(ValueError, match="disjoint"):
        CxlPeerL2AdapterConfig(url, str(path), local, replace(remote, offset=0))
    with pytest.raises(ValueError, match="header"):
        CxlPeerMapping(str(path), replace(remote, session_id="stale"))
    mapping = CxlPeerMapping(str(path), remote)
    try:
        with pytest.raises(ValueError, match="range"):
            mapping.view(remote.size, PAGE)
        with pytest.raises(ValueError, match="range"):
            mapping.view(1, PAGE)
    finally:
        mapping.close()


def test_adapter_draining_retains_mapping_until_gpu_completion(peers: tuple) -> None:
    """Runtime removal waits for completed prefetches' outstanding readers."""
    borrower, owner, url, _ = peers
    _store(owner)
    adapter_id = borrower.add_cxl_peer(owner.cxl_arena, url, 10)
    task = borrower.submit_prefetch_task(
        PrefetchTaskSpec(
            key_groups=[GroupedObjectKeys([KEY], 0, LAYOUT)],
            num_kv_readers=1,
        )
    )
    assert borrower.wait_prefetch_status(task, timeout=10)
    with pytest.raises(TimeoutError, match="draining"):
        borrower.delete_l2_adapter(adapter_id, timeout=0.1)
    assert len(borrower.l2_adapters()) == 1
    with borrower.read_prefetched_results([KEY]) as objects:
        assert torch.all(objects[0].tensor == 37)
    borrower.finish_read_prefetched([KEY])
    borrower.delete_l2_adapter(adapter_id, timeout=10)
    assert borrower.l2_adapters() == []
    assert owner.delete_l1_keys([KEY]) == (1, 0)


def test_shadow_cannot_be_exported_to_another_peer(peers: tuple) -> None:
    """A second node cannot borrow a view that is already borrowed."""
    borrower, owner, url, path = peers
    _store(owner)
    adapter = _adapter(borrower, owner, url, path)
    view = _borrow(adapter)
    try:
        assert borrower.get_cxl_address(view) is None
        owner_keys, objects = owner.unsafe_read([KEY])
        assert owner_keys == [KEY]
        assert owner.get_cxl_address(objects[0]) is not None
    finally:
        view.release()
        adapter.close()


def test_shadow_admission_discards_duplicate_without_freeing_payload(
    peers: tuple,
    tmp_path: Path,
) -> None:
    """Concurrent admissions preserve the resident view and balance both borrows."""
    borrower, owner, url, path = peers
    _store(owner)
    adapter = _adapter(borrower, owner, url, path)
    local_path = tmp_path / "local.bin"
    local_path.write_bytes(bytes(PAGE * 2))
    manager = L1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=PAGE,
                use_lazy=False,
                shm_name="",
                devdax_path=str(local_path),
                cxl_pool_id="test-pool",
            )
        )
    )
    try:
        first, second = _borrow(adapter), _borrow(adapter)
        assert manager.register_shadow(KEY, first, "a") == L1Error.SUCCESS
        assert manager.register_shadow(KEY, second, "b") == L1Error.SUCCESS
        assert manager.get_staging_memory_usage() == 0
        assert manager.finish_write_and_reserve_read([KEY], tag="a")[KEY][1] is first
        assert manager.finish_write_and_reserve_read([KEY], tag="b")[KEY][1] is first
        _wait(lambda: adapter.get_active_borrow_count() == 1)
        assert not second.is_valid()
        assert manager.get_memory_usage() == (0, PAGE)
        assert manager.finish_read([KEY])[KEY] == L1Error.SUCCESS
        assert owner.delete_l1_keys([KEY]) == (0, 1)
        assert manager.finish_read([KEY])[KEY] == L1Error.SUCCESS
        _wait(lambda: adapter.get_active_borrow_count() == 0)
        assert owner.delete_l1_keys([KEY]) == (1, 0)
    finally:
        manager.close()
        adapter.close()


def test_abandoned_owner_reservation_expires(tmp_path: Path) -> None:
    """CXL ownership uses the existing TTL read-lock reclamation contract."""
    path = tmp_path / "pool.bin"
    path.write_bytes(bytes(PAGE * 8))
    owner = _manager(path, 0, ttl=1)
    controller = P2PController(
        cast(MPCacheServerContext, SimpleNamespace(storage_manager=owner)),
        P2PConfig(),
        CoordinatorConfig(),
        instance_id="owner",
    )
    try:
        _store(owner)
        task = controller.p2p_lookup_and_lock([KEY], {0: LAYOUT})
        _wait(lambda: controller.p2p_query_lookup_results(task) is not None)
        assert owner.delete_l1_keys([KEY]) == (0, 1)
        _wait(lambda: owner.delete_l1_keys([KEY]) == (1, 0))
    finally:
        controller.close()
        owner.close()


@pytest.mark.parametrize("owner_ttl", [1])
def test_owner_expiry_cannot_be_extended_by_local_readers(peers: tuple) -> None:
    """Local reuse cannot extend the original remote reservation's lifetime."""
    borrower, owner, url, _ = peers
    _store(owner)
    borrower.add_cxl_peer(owner.cxl_arena, url, 10)
    spec = PrefetchTaskSpec(
        key_groups=[GroupedObjectKeys([KEY], 0, LAYOUT)],
        num_kv_readers=1,
    )
    first = borrower.submit_prefetch_task(spec)
    assert borrower.wait_prefetch_status(first, timeout=10)
    assert borrower.query_prefetch_status(first).hit_cells[0].test(0)
    time.sleep(0.55)
    second = borrower.submit_prefetch_task(spec)
    assert borrower.wait_prefetch_status(second, timeout=10)
    assert borrower.query_prefetch_status(second).hit_cells[0].test(0)
    adapter = borrower.l2_adapters()[0][1]
    _wait(lambda: adapter.get_active_borrow_count() == 0)
    assert borrower.unsafe_read([KEY])[0] == []
    _wait(lambda: owner.delete_l1_keys([KEY]) == (1, 0))
