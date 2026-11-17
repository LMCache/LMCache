# SPDX-License-Identifier: Apache-2.0
"""Owner-qualified capacity contracts, using controlled native completions."""

# Standard
from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from pathlib import Path
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import MemoryResidency, MemorySegment, MemoryTags
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    FSNativeL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    NativeConnectorL2Adapter,
)
from lmcache.v1.memory_management import BytesBufferMemoryObj
from lmcache.v1.platform import create_event_notifier
from tests.v1.distributed.test_native_connector_l2_adapter import (
    create_object_key,
    wait_for_event_fd,
)

LOCAL = MemoryTags(MemorySegment.STORAGE, MemoryResidency.LOCAL)


class ControlledConnector:
    """Hold native operations until the test supplies completion evidence."""

    def __init__(self) -> None:
        self.notifier = create_event_notifier()
        self.requests: dict[int, tuple[str, list[str]]] = {}
        self.completions: list[tuple[int, bool, str, list[bool] | None]] = []
        self.next_id = 0
        self.lock = threading.Lock()

    def event_fd(self) -> int:
        return self.notifier.fileno()

    def submit_batch_set(self, keys: list[str], buffers: list[memoryview]) -> int:
        return self.submit("store", keys)

    def submit_batch_exists(self, keys: list[str]) -> int:
        return self.submit("lookup", keys)

    def submit_batch_delete(self, keys: list[str]) -> int:
        return self.submit("delete", keys)

    def submit(self, operation: str, keys: list[str]) -> int:
        with self.lock:
            self.next_id += 1
            self.requests[self.next_id] = (operation, keys)
            return self.next_id

    def complete(self, fid: int, ok: bool, results: list[bool] | None) -> None:
        with self.lock:
            self.completions.append((fid, ok, "" if ok else "injected", results))
        self.notifier.notify()

    def drain_completions(self) -> list[tuple[int, bool, str, list[bool] | None]]:
        self.notifier.consume()
        with self.lock:
            result, self.completions = self.completions, []
            return result

    def close(self) -> None:
        self.notifier.close()


OwnerFixture = tuple[NativeConnectorL2Adapter, ControlledConnector]


@pytest.fixture
def owner() -> Iterator[OwnerFixture]:
    client = ControlledConnector()
    adapter = NativeConnectorL2Adapter(
        client, LOCAL, max_capacity_gb=1024 / 1024**3, value_capacity=True
    )
    try:
        yield adapter, client
    finally:
        adapter.close()


def test_default_interface_is_not_qualified() -> None:
    client = ControlledConnector()
    adapter = NativeConnectorL2Adapter(client, LOCAL, max_capacity_gb=1)
    try:
        assert not adapter.supports_value_capacity
        with pytest.raises(NotImplementedError):
            adapter.get_capacity_snapshot()
        with pytest.raises(NotImplementedError):
            adapter.delete_unlocked([])
    finally:
        adapter.close()


def test_snapshot_is_frozen_detached_and_pending_is_unique(owner: OwnerFixture) -> None:
    adapter, client = owner
    a, b = create_object_key(1), create_object_key(2)
    obj = BytesBufferMemoryObj(b"a" * 64)
    empty = adapter.get_capacity_snapshot()
    adapter.submit_store_task([a, a, b], [obj, obj, obj])
    adapter.submit_store_task([a], [obj])
    snapshot = adapter.get_capacity_snapshot()
    assert snapshot.capacity_bytes == 1024
    assert snapshot.pending_bytes == 128
    assert snapshot.unavailable == frozenset([a, b])
    assert not snapshot.resident_sizes
    with pytest.raises(FrozenInstanceError):
        snapshot.pending_bytes = 0  # type: ignore[misc]
    with pytest.raises(TypeError):
        snapshot.resident_sizes[a] = 1  # type: ignore[index]
    client.complete(1, True, None)
    assert wait_for_event_fd(adapter.get_store_event_fd())
    current = adapter.get_capacity_snapshot()
    assert current.resident_sizes == {a: 64, b: 64}
    assert current.pending_bytes == 0
    assert current.unavailable == frozenset([a])
    assert not empty.resident_sizes and not snapshot.resident_sizes


@pytest.mark.parametrize("found", [[False, False], [True, False]])
def test_failed_store_reconciles_native_partial_writes(
    owner: OwnerFixture, found: list[bool]
) -> None:
    adapter, client = owner
    keys = [create_object_key(1), create_object_key(2)]
    obj = BytesBufferMemoryObj(b"x" * 64)
    task = adapter.submit_store_task(keys, [obj, obj])
    client.complete(1, False, None)
    # Wait for the reconciliation submit without reaching into adapter state.
    deadline = threading.Event()
    for _ in range(500):
        if client.next_id == 2:
            break
        deadline.wait(0.01)
    assert client.requests[2][0] == "lookup"
    assert adapter.get_capacity_snapshot().pending_bytes == 128
    client.complete(2, True, found)
    assert wait_for_event_fd(adapter.get_store_event_fd())
    assert not adapter.pop_completed_store_tasks()[task].is_successful()
    snapshot = adapter.get_capacity_snapshot()
    assert snapshot.resident_sizes == ({keys[0]: 64} if found[0] else {})
    assert snapshot.pending_bytes == 0
    assert not snapshot.unavailable


def test_delete_skips_pending_store_lookup_and_locks(owner: OwnerFixture) -> None:
    adapter, client = owner
    a, b = create_object_key(1), create_object_key(2)
    adapter.submit_store_task([a, b], [BytesBufferMemoryObj(b"x")] * 2)
    adapter.delete_unlocked([a, b])
    assert client.next_id == 1
    client.complete(1, True, None)
    assert wait_for_event_fd(adapter.get_store_event_fd())
    task = adapter.submit_lookup_and_lock_task([a], {})
    adapter.delete_unlocked([a])
    assert client.next_id == 2
    assert a in adapter.get_capacity_snapshot().unavailable
    client.complete(2, True, [True])
    assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
    bitmap = adapter.query_lookup_and_lock_result(task)
    assert bitmap is not None and bitmap.test(0)
    adapter.delete_unlocked([a])
    assert client.next_id == 2
    adapter.submit_unlock([a])
    finished = threading.Event()

    def delete_and_signal() -> None:
        adapter.delete_unlocked([a, b])
        finished.set()

    thread = threading.Thread(target=delete_and_signal)
    thread.start()
    try:
        for _ in range(500):
            if client.next_id == 3:
                break
            finished.wait(0.01)
        assert client.requests[3][0] == "delete"
        assert not finished.is_set()
        assert adapter.get_capacity_snapshot().unavailable == frozenset([a, b])
        adapter.delete_unlocked([a, b])
        assert client.next_id == 3
        client.complete(3, True, [True, False])
        assert finished.wait(5)
        assert adapter.get_capacity_snapshot().resident_sizes == {b: 1}
    finally:
        thread.join(5)
    assert not thread.is_alive()


def test_delete_timeout_retains_pending_and_late_completion(
    owner: OwnerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, client = owner
    key = create_object_key(1)
    adapter.submit_store_task([key], [BytesBufferMemoryObj(b"x")])
    client.complete(1, True, None)
    assert wait_for_event_fd(adapter.get_store_event_fd())
    original_wait = threading.Event.wait

    def wait(event: threading.Event, timeout: float | None = None) -> bool:
        return False if timeout == 30.0 else original_wait(event, timeout)

    with monkeypatch.context() as patcher:
        patcher.setattr(threading.Event, "wait", wait)
        with pytest.raises(TimeoutError):
            adapter.delete_unlocked([key])
    assert adapter.get_capacity_snapshot().unavailable == frozenset([key])
    adapter.delete_unlocked([key])
    assert client.next_id == 2
    client.complete(2, True, [True])
    for _ in range(500):
        if not adapter.get_capacity_snapshot().unavailable:
            break
        original_wait(threading.Event(), 0.01)
    assert not adapter.get_capacity_snapshot().resident_sizes


@pytest.mark.parametrize(
    "ok,results", [(False, [False, False]), (True, None), (True, [False])]
)
def test_missing_reconciliation_evidence_never_releases_reservations(
    owner: OwnerFixture, ok: bool, results: list[bool] | None
) -> None:
    adapter, client = owner
    keys = [create_object_key(1), create_object_key(2)]
    task = adapter.submit_store_task(keys, [BytesBufferMemoryObj(b"x" * 64)] * 2)
    client.complete(1, False, None)
    for _ in range(500):
        if client.next_id == 2:
            break
        threading.Event().wait(0.01)
    assert client.next_id == 2
    client.complete(2, ok, results)
    assert wait_for_event_fd(adapter.get_store_event_fd())
    assert not adapter.pop_completed_store_tasks()[task].is_successful()
    snapshot = adapter.get_capacity_snapshot()
    assert snapshot.pending_bytes == 128
    assert snapshot.unavailable == frozenset(keys)
    adapter.delete_unlocked(keys)
    assert client.next_id == 2


@pytest.mark.parametrize("results", [None, [False]])
def test_delete_without_positive_evidence_reclaims_nothing(
    owner: OwnerFixture, results: list[bool] | None
) -> None:
    adapter, client = owner
    key = create_object_key(1)
    adapter.submit_store_task([key], [BytesBufferMemoryObj(b"x")])
    client.complete(1, True, None)
    assert wait_for_event_fd(adapter.get_store_event_fd())
    thread = threading.Thread(target=adapter.delete_unlocked, args=([key],))
    thread.start()
    try:
        for _ in range(500):
            if client.next_id == 2:
                break
            threading.Event().wait(0.01)
        assert client.next_id == 2
        with pytest.raises(RuntimeError, match="pending deletion"):
            adapter.submit_store_task([key], [BytesBufferMemoryObj(b"x")])
        task = adapter.submit_lookup_and_lock_task([key], {})
        bitmap = adapter.query_lookup_and_lock_result(task)
        assert bitmap is not None and bitmap.test(0) is False
        client.complete(2, False, results)
    finally:
        thread.join(5)
    assert not thread.is_alive()
    assert adapter.get_capacity_snapshot().resident_sizes == {key: 1}


def test_real_fs_owner_roundtrip_and_cold_restart_rejection(tmp_path: Path) -> None:
    pytest.importorskip("lmcache.lmcache_fs")
    config = FSNativeL2AdapterConfig(
        str(tmp_path),
        max_capacity_gb=1024 / 1024**3,
        require_empty=True,
        num_read_workers=0,
    )
    adapter = create_l2_adapter(config)
    keys = [create_object_key(1), create_object_key(2)]
    try:
        assert adapter.supports_value_capacity
        task = adapter.submit_store_task(keys, [BytesBufferMemoryObj(b"x" * 64)] * 2)
        assert wait_for_event_fd(adapter.get_store_event_fd())
        assert adapter.pop_completed_store_tasks()[task].is_successful()
        assert adapter.get_capacity_snapshot().resident_sizes == dict.fromkeys(keys, 64)
        lookup = adapter.submit_lookup_and_lock_task([keys[0]], {})
        assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        bitmap = adapter.query_lookup_and_lock_result(lookup)
        assert bitmap is not None and bitmap.test(0)
        adapter.delete_unlocked(keys)
        assert adapter.get_capacity_snapshot().resident_sizes == {keys[0]: 64}
        with pytest.raises(ValueError, match="empty"):
            create_l2_adapter(config)
        adapter.submit_unlock([keys[0]])
        adapter.delete_unlocked(keys)
        assert not adapter.get_capacity_snapshot().resident_sizes
    finally:
        adapter.close()
