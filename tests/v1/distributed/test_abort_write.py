# SPDX-License-Identifier: Apache-2.0
"""
Tests for abort_write: releasing write reservations without publishing them.
"""

# Standard
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import Mock

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.cli.commands.trace._dispatch import ReplayContext, build_default_dispatcher
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_controllers.write_policy import OrderedWritePolicy
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.trace import codecs
from lmcache.v1.mp_observability.trace import decorator as trace_decorator
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.server_transfer import InlineTransferStrategy
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContextMetadata
from lmcache.v1.multiprocess.transfer_context.tensor_wire import encode_tensors
import lmcache.v1.distributed.storage_manager as storage_manager_module
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator

LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([1024])], dtypes=[torch.float32])
# The same byte size as LAYOUT but a different shape: a LAYOUT-shaped chunk
# does not fit an object reserved with it, the worker/server layout drift the
# inline store guards against.
DRIFTED_LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([2, 512])], dtypes=[torch.float32])
OBJECT_BYTES = 4096

Storage = tuple[StorageManager, tuple[L1Manager, ...], list[Event]]
StorageFactory = Callable[[tuple[int, ...]], Storage]


def key(index: int) -> ObjectKey:
    """Return a deterministic test key."""
    return ObjectKey(ObjectKey.IntHash2Bytes(index), "abort-test", 0)


class _RecordingListener:
    """L1 listener that records write-finished and deleted notifications."""

    def __init__(self) -> None:
        self.written: list[ObjectKey] = []
        self.deleted: list[ObjectKey] = []

    def on_l1_keys_write_finished(self, keys: list[ObjectKey]) -> None:
        self.written.extend(keys)

    def on_l1_keys_finish_write_and_reserve_read(self, keys: list[ObjectKey]) -> None:
        self.written.extend(keys)

    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]) -> None:
        self.deleted.extend(keys)

    def __getattr__(self, name: str) -> Callable[..., None]:
        # Every other listener hook is a no-op.
        return lambda *args, **kwargs: None


@pytest.fixture
def storage_factory(monkeypatch: pytest.MonkeyPatch) -> Iterator[StorageFactory]:
    """Real CPU allocators and L1 lifecycle; host pinning and the event bus
    are replaced. Events published by the StorageManager are captured."""
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)
    stores: list[StorageManager] = []

    def create(sizes: tuple[int, ...] = (OBJECT_BYTES,)) -> Storage:
        events: list[Event] = []
        bus = Mock()
        bus.publish.side_effect = events.append
        monkeypatch.setattr(storage_manager_module, "get_event_bus", lambda: bus)
        configs = [
            L1ManagerConfig(
                L1MemoryManagerConfig(size_in_bytes=size, use_lazy=False, shm_name=""),
                tag="_default" if index == 0 else f"l1-{index}",
            )
            for index, size in enumerate(sizes)
        ]
        managers = tuple(L1Manager(config) for config in configs)
        store = StorageManager(
            StorageManagerConfig(configs[0], EvictionConfig("noop")),
            _l1_managers=managers,
            _write_policy=OrderedWritePolicy(tuple(m.l1_manager_id for m in managers)),
        )
        stores.append(store)
        return store, managers, events

    yield create
    for store in reversed(stores):
        store.close()


def _attach_listener(manager: L1Manager) -> _RecordingListener:
    listener = _RecordingListener()
    manager.register_listener(listener)  # type: ignore[arg-type]
    return listener


def _aborted(events: list[Event]) -> list[dict[str, Any]]:
    return [e.metadata for e in events if e.event_type == EventType.SM_WRITE_ABORTED]


def test_abort_write_discards_reservations_without_publishing(
    storage_factory: StorageFactory,
) -> None:
    """Aborted reservations free their memory, are never reported as written,
    and leave the keys reservable again (regression: leaked write locks filled
    L1 until every later reserve failed)."""
    store, (l1,), events = storage_factory((2 * OBJECT_BYTES,))
    listener = _attach_listener(l1)
    keys = [key(1), key(2)]

    assert set(store.reserve_write(keys, LAYOUT)) == set(keys)
    assert l1.get_staging_memory_usage() == 2 * OBJECT_BYTES

    store.abort_write(keys)

    assert l1.get_staging_memory_usage() == 0
    assert all(l1.get_object_state(k) is None for k in keys)
    assert listener.written == []
    assert set(listener.deleted) == set(keys)
    assert _aborted(events) == [{"succeeded_keys": keys, "failed_keys": []}]
    # The writer can reserve the same keys again.
    assert set(store.reserve_write(keys, LAYOUT)) == set(keys)
    store.abort_write(keys)
    assert store.memcheck()


def test_abort_write_never_touches_a_resident_object(
    storage_factory: StorageFactory,
) -> None:
    """Aborting a key with no live reservation reports it as failed and keeps
    a resident copy, unlike delete_l1_keys."""
    store, (l1,), events = storage_factory((OBJECT_BYTES,))
    store.reserve_write([key(1)], LAYOUT)
    store.finish_write([key(1)])

    store.abort_write([key(1), key(2)])

    assert l1.get_object_state(key(1)) is not None
    assert _aborted(events) == [{"succeeded_keys": [], "failed_keys": [key(1), key(2)]}]


def test_abort_write_by_owner_routes_each_key_to_its_l1(
    storage_factory: StorageFactory,
) -> None:
    """With several L1s the reservations overflow into different managers;
    the owner-routed abort releases each one where it lives."""
    store, managers, events = storage_factory((OBJECT_BYTES, OBJECT_BYTES))
    listeners = [_attach_listener(m) for m in managers]
    keys = [key(1), key(2)]
    # Each L1 holds one object and overflow does not split a batch, so
    # reserve one key at a time: the second spills into the second L1.
    objects = store.reserve_write(keys[:1], LAYOUT)
    objects.update(store.reserve_write(keys[1:], LAYOUT))
    owners = {k: obj.get_l1_manager() for k, obj in objects.items()}
    assert set(owners.values()) == {m.l1_manager_id for m in managers}

    with pytest.raises(ValueError):
        store.abort_write(keys)

    store.abort_write_by_owner(store.prepare_write_completion(objects))

    assert all(m.get_staging_memory_usage() == 0 for m in managers)
    assert all(lst.written == [] for lst in listeners)
    assert [k for lst in listeners for k in lst.deleted] == keys
    assert len(_aborted(events)) == 1
    assert set(_aborted(events)[0]["succeeded_keys"]) == set(keys)
    # Both L1s have room again.
    assert set(store.reserve_write(keys[:1], LAYOUT)) == set(keys[:1])
    assert set(store.reserve_write(keys[1:], LAYOUT)) == set(keys[1:])


class _DriftedReservations:
    """Real StorageManager whose reservations after the first key use
    DRIFTED_LAYOUT, so their chunks fail the inline store's shape check."""

    def __init__(self, store: StorageManager) -> None:
        self._store = store

    def reserve_write(self, keys: list[ObjectKey], layout_desc: MemoryLayoutDesc):
        reserved = self._store.reserve_write(keys[:1], layout_desc)
        reserved.update(self._store.reserve_write(keys[1:], DRIFTED_LAYOUT))
        return reserved

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)


def test_inline_commit_store_aborts_unwritten_reservations(
    storage_factory: StorageFactory,
) -> None:
    """Reservations the inline store cannot fill are aborted: not left
    write-locked, never reported to store listeners as written, and not
    resident, while the written key is admitted."""
    store, (l1,), _ = storage_factory((3 * OBJECT_BYTES,))
    listener = _attach_listener(l1)
    strategy = InlineTransferStrategy(_DriftedReservations(store))  # type: ignore[arg-type]
    keys = [key(30 + i) for i in range(3)]
    request_key = IPCCacheServerKey.from_token_ids(
        "abort-test", 1, 0, [1] * 48, start=0, end=48, request_id="req-abort"
    )
    payload = encode_tensors([torch.full((1024,), float(i)) for i in range(3)])

    ok = strategy.commit_store(
        key=request_key,
        instance_id=1,
        cpu_data=payload,
        context=EngineDrivenContextMetadata(
            layout_desc=LAYOUT, block_size=16, use_mla=False
        ),
        resolve_obj_keys=lambda _k: keys,
    )

    assert ok is False
    assert listener.written == [keys[0]]
    assert l1.get_object_state(keys[0]) is not None
    assert all(l1.get_object_state(k) is None for k in keys[1:])
    assert l1.get_staging_memory_usage() == 0
    assert set(listener.deleted) == set(keys[1:])


@pytest.mark.parametrize("owner_completion", [False, True])
def test_single_l1_abort_trace_replays(
    storage_factory: StorageFactory,
    monkeypatch: pytest.MonkeyPatch,
    owner_completion: bool,
) -> None:
    """Both abort entry points record the key-only ``abort_write`` call, and
    replaying the trace releases the reservation on a fresh manager."""
    source, _, _ = storage_factory((OBJECT_BYTES,))
    trace_events: list[Event] = []
    bus = Mock()
    bus.publish.side_effect = trace_events.append
    monkeypatch.setattr(trace_decorator, "get_event_bus", lambda: bus)
    trace_decorator.set_tracing_enabled(True)
    try:
        objects = source.reserve_write([key(1)], LAYOUT)
        if owner_completion:
            source.abort_write_by_owner(source.prepare_write_completion(objects))
        else:
            source.abort_write([key(1)])
    finally:
        trace_decorator.set_tracing_enabled(False)

    prefix = "lmcache.v1.distributed.storage_manager.StorageManager."
    assert [e.metadata["qualname"] for e in trace_events] == [
        prefix + "reserve_write",
        prefix + "abort_write",
    ]
    assert trace_events[-1].metadata["args"] == {"keys": [key(1)]}

    target, (target_l1,), _ = storage_factory((OBJECT_BYTES,))
    dispatcher = build_default_dispatcher()
    for event in trace_events:
        qualname = event.metadata["qualname"]
        assert dispatcher.has(qualname)
        payload = msgspec.msgpack.encode(codecs.encode_args(event.metadata["args"]))
        dispatcher.dispatch(
            qualname,
            ReplayContext(target),
            codecs.decode_args(msgspec.msgpack.decode(payload)),
        )
    assert target_l1.get_staging_memory_usage() == 0
    assert target_l1.get_object_state(key(1)) is None
