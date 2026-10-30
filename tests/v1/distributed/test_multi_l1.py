# SPDX-License-Identifier: Apache-2.0
"""Per-key L1 placement, owner completion, and overlapping read contracts."""

# Standard
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast
from unittest.mock import Mock
import logging

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.cli.commands.trace._dispatch import ReplayContext, build_default_dispatcher
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.error import L1Error, L1ReconfigureError
from lmcache.v1.distributed.internal_api import L1ManagerInterface
from lmcache.v1.distributed.l1_manager.base import BaseL1Manager
from lmcache.v1.distributed.storage_manager import StorageManager, l1_owners_of
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.trace import codecs
from lmcache.v1.mp_observability.trace import decorator as trace_decorator
from lmcache.v1.mp_observability.trace.reader import TraceReader
from lmcache.v1.mp_observability.trace.recorder import StorageTraceRecorder
from tests.v1.distributed.utils import (
    ROUND_ROBIN_WRITE_POLICY,
    SCRIPTED_WRITE_POLICY,
    ScriptedWritePolicy,
    single_row_spec,
)
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator

LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([1024])], dtypes=[torch.float32])
DRAM_LOCAL = MemoryTags(MemorySegment.DRAM, MemoryResidency.LOCAL)
StorageFactory = Callable[..., tuple[StorageManager, tuple[BaseL1Manager, ...]]]


def key(index: int) -> ObjectKey:
    """Return a deterministic test key."""
    return ObjectKey(ObjectKey.IntHash2Bytes(index), "overflow-test", 0)


@pytest.fixture
def storage_factory(monkeypatch: pytest.MonkeyPatch) -> Iterator[StorageFactory]:
    """Use real CPU allocators and L1 lifecycle; replace only host pinning."""
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)
    stores: list[StorageManager] = []

    def create(
        sizes: tuple[int, ...] = (4096, 4096),
        write_policy: str = "default",
    ) -> tuple[StorageManager, tuple[BaseL1Manager, ...]]:
        configs = [
            L1ManagerConfig(
                L1MemoryManagerConfig(size_in_bytes=size, use_lazy=False, shm_name=""),
                tag="_default" if index == 0 else f"l1-{index}",
            )
            for index, size in enumerate(sizes)
        ]
        store = StorageManager(
            StorageManagerConfig(
                l1_manager_config=configs[0],
                eviction_config=EvictionConfig("noop"),
                l1_manager_configs=configs,
                write_policy=write_policy,
            )
        )
        stores.append(store)
        managers = tuple(cast(BaseL1Manager, m) for m in store.l1_managers())
        return store, managers

    yield create
    for store in reversed(stores):
        store.close()


def _wrap_reserve_write(
    monkeypatch: pytest.MonkeyPatch, managers: tuple[BaseL1Manager, ...]
) -> list[Mock]:
    """Wrap every manager's ``reserve_write`` so calls can be counted."""
    attempts = []
    for manager in managers:
        attempt = Mock(wraps=manager.reserve_write)
        monkeypatch.setattr(manager, "reserve_write", attempt)
        attempts.append(attempt)
    return attempts


def test_default_policy_places_every_key_in_the_first_l1(
    storage_factory: StorageFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, (primary, fallback) = storage_factory((8192, 8192))
    _, fallback_attempt = _wrap_reserve_write(monkeypatch, (primary, fallback))
    objects = store.reserve_write([key(1), key(2)], LAYOUT)
    assert set(objects) == {key(1), key(2)}
    assert all(o.get_l1_manager() == primary.l1_manager_id for o in objects.values())
    store.finish_write(list(objects), l1_owners_of(objects))
    assert primary.get_object_state(key(1)) is not None
    assert fallback.get_object_state(key(1)) is None
    fallback_attempt.assert_not_called()


def test_policy_sees_the_request_keys_and_manager_ids_on_every_call(
    storage_factory: StorageFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, managers = storage_factory((8192, 8192), SCRIPTED_WRITE_POLICY)
    calls: list[tuple[list[ObjectKey], list[int]]] = []

    def answer(
        keys: list[ObjectKey], l1_managers: list[L1ManagerInterface]
    ) -> dict[int, list[ObjectKey]]:
        calls.append((list(keys), [m.l1_manager_id for m in l1_managers]))
        return {l1_managers[0].l1_manager_id: keys}

    monkeypatch.setattr(ScriptedWritePolicy, "answer", answer)
    store.reserve_write([key(1)], LAYOUT)
    store.reserve_write([key(2), key(3)], LAYOUT)
    ids = [m.l1_manager_id for m in managers]
    assert calls == [([key(1)], ids), ([key(2), key(3)], ids)]


def test_split_policy_routes_each_key_to_its_selected_l1(
    storage_factory: StorageFactory,
) -> None:
    store, (primary, fallback) = storage_factory((8192, 8192), ROUND_ROBIN_WRITE_POLICY)
    keys = [key(1), key(2), key(3), key(4)]
    objects = store.reserve_write(keys, LAYOUT)
    assert list(objects) == keys
    owners = [objects[k].get_l1_manager() for k in keys]
    assert owners == [
        primary.l1_manager_id,
        fallback.l1_manager_id,
        primary.l1_manager_id,
        fallback.l1_manager_id,
    ]
    store.finish_write(list(objects), l1_owners_of(objects))
    assert [primary.get_object_state(k) is not None for k in keys] == [
        True,
        False,
        True,
        False,
    ]
    assert [fallback.get_object_state(k) is not None for k in keys] == [
        False,
        True,
        False,
        True,
    ]
    assert primary.get_memory_usage()[0] == fallback.get_memory_usage()[0] == 8192


def test_oom_on_the_selected_l1_does_not_fall_back(
    storage_factory: StorageFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, (primary, fallback) = storage_factory((4096, 8192))
    primary_attempt, fallback_attempt = _wrap_reserve_write(
        monkeypatch, (primary, fallback)
    )
    keys = [key(1), key(2)]
    # The batch does not fit the selected L1 even though the other one is idle.
    assert store.reserve_write(keys, LAYOUT) == {}
    primary_attempt.assert_called_once()
    assert primary_attempt.call_args.kwargs["keys"] == keys
    fallback_attempt.assert_not_called()
    assert primary.get_memory_usage()[0] == 0
    assert fallback.get_memory_usage()[0] == 0


@pytest.mark.parametrize("bad", ["unregistered", "duplicate", "unplaced"])
def test_invalid_policy_answers_raise_before_any_reservation(
    storage_factory: StorageFactory, monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    store, managers = storage_factory((4096, 4096), SCRIPTED_WRITE_POLICY)
    attempts = _wrap_reserve_write(monkeypatch, managers)
    primary_id, fallback_id = (m.l1_manager_id for m in managers)

    def answer(
        keys: list[ObjectKey], _l1_managers: list[L1ManagerInterface]
    ) -> dict[int, list[ObjectKey]]:
        if bad == "unregistered":
            return {primary_id: [keys[0]], 2**62: [keys[1]]}
        if bad == "duplicate":
            return {primary_id: keys, fallback_id: [keys[1]]}
        return {primary_id: keys[:1]}

    monkeypatch.setattr(ScriptedWritePolicy, "answer", answer)
    with pytest.raises(ValueError, match="write policy"):
        store.reserve_write([key(1), key(2)], LAYOUT)
    for attempt in attempts:
        attempt.assert_not_called()
    assert all(m.get_memory_usage()[0] == 0 for m in managers)


def test_exception_is_not_capacity_failure(
    storage_factory: StorageFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, (primary, _) = storage_factory((4096, 4096))
    monkeypatch.setattr(
        primary, "reserve_write", Mock(side_effect=RuntimeError("failure"))
    )
    with pytest.raises(RuntimeError, match="failure"):
        store.reserve_write([key(1)], LAYOUT)


def test_same_key_copies_finish_only_the_selected_owner(
    storage_factory: StorageFactory,
) -> None:
    store, (primary, fallback) = storage_factory((4096, 4096))
    objects = []
    for manager in (primary, fallback):
        error, obj = manager.reserve_write(
            [key(1)], [False], LAYOUT, tag="storage_manager"
        )[key(1)]
        assert error == L1Error.SUCCESS and obj is not None
        objects.append(obj)
    assert objects[0].get_l1_manager() != objects[1].get_l1_manager()
    store.finish_write([key(1)], {key(1): objects[1].get_l1_manager()})
    assert primary.reserve_read([key(1)])[key(1)][0] == L1Error.KEY_NOT_EXIST
    assert primary.report_status()["write_locked_count"] == 1
    assert fallback.reserve_read([key(1)])[key(1)][0] == L1Error.SUCCESS
    fallback.finish_read([key(1)])
    assert primary.report_status()["write_locked_count"] == 1


def test_unknown_or_unset_owner_is_not_guessed(storage_factory: StorageFactory) -> None:
    store, _ = storage_factory((4096, 4096))
    obj = memory_management.BytesBufferMemoryObj(b"test")
    with pytest.raises(ValueError, match="no L1 owner"):
        l1_owners_of({key(1): obj})
    with pytest.raises(ValueError, match="registered L1 owner"):
        store.finish_write([key(1)], {key(1): 2**62})
    with pytest.raises(ValueError, match="registered L1 owner"):
        store.finish_write([key(1)], {})


def test_multi_manager_owner_resolution_and_tag_lookup(
    storage_factory: StorageFactory,
) -> None:
    store, managers = storage_factory((4096, 4096))
    # A key no L1 holds is simply not found, with or without owners.
    assert store.unsafe_read([key(1)]) == ([], [])
    with store.read_prefetched_results([key(1)]) as read:
        assert read is None
    # An owner id that is not a registered L1, or a missing owner, is an error.
    for operation in (store.finish_read_prefetched, store.unsafe_read):
        with pytest.raises(ValueError, match="registered L1 owner"):
            operation([key(1)], l1_owners={key(1): 2**62})
        with pytest.raises(ValueError, match="registered L1 owner"):
            operation([key(1)], l1_owners={})
    assert store.l1_managers() == list(managers)
    # Two DRAM L1s are ambiguous for a tag lookup; no L1 is CXL.
    with pytest.raises(ValueError, match="found 2"):
        store.get_l1_manager_by_tags(DRAM_LOCAL)
    with pytest.raises(ValueError, match="found 0"):
        store.get_l1_manager_by_tags(
            MemoryTags(MemorySegment.CXL, MemoryResidency.LOCAL)
        )
    for legacy_operation in (
        store.get_l1_devdax_arena_statuses,
        lambda: store.add_l1_devdax_device("/unused", 4096),
        lambda: store.remove_l1_devdax_device("/unused"),
    ):
        with pytest.raises(L1ReconfigureError, match="not Device-DAX backed"):
            legacy_operation()
    assert store.memcheck()
    assert store.get_l1_usage() == (0, 8192)
    assert set(store.report_status()["l1_managers"]) == {"_default", "l1-1"}
    store.publish_capacity()


def test_reads_without_owners_fan_out_with_a_warning(
    storage_factory: StorageFactory,
) -> None:
    """A read that names no owners finds the key on whichever L1 holds it."""
    store, (primary, fallback) = storage_factory((4096, 4096), ROUND_ROBIN_WRITE_POLICY)
    objects = store.reserve_write([key(1), key(2)], LAYOUT)
    assert objects[key(2)].get_l1_manager() == fallback.l1_manager_id
    store.finish_write(list(objects), l1_owners_of(objects))
    handle = store.submit_prefetch_task(
        single_row_spec([key(1), key(2)], LAYOUT, allowed_l2_memory_tags=frozenset())
    )
    result = store.query_prefetch_status(handle)
    assert result is not None and result.hit_cells[0].popcount() == 2

    class _Collect(logging.Handler):
        records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    handler = _Collect()
    sm_logger = logging.getLogger("lmcache.v1.distributed.storage_manager")
    sm_logger.addHandler(handler)
    try:
        found, read = store.unsafe_read([key(2)])
        assert found == [key(2)] and read[0].get_l1_manager() == fallback.l1_manager_id
        store.finish_read_prefetched([key(1), key(2)])
    finally:
        sm_logger.removeHandler(handler)
    # Every owner-less call on several L1s is announced.
    assert [r.levelno for r in handler.records] == [logging.WARNING] * 2
    assert all("probing every L1" in r.getMessage() for r in handler.records)
    assert primary.report_status()["read_locked_count"] == 0
    assert fallback.report_status()["read_locked_count"] == 0


def test_single_manager_write_prefetch_and_read(
    storage_factory: StorageFactory,
) -> None:
    store, (manager,) = storage_factory((4096,))
    objects = store.reserve_write([key(1)], LAYOUT)
    tensor = objects[key(1)].tensor
    assert tensor is not None
    tensor.fill_(7)
    store.finish_write(list(objects), l1_owners_of(objects))
    handle = store.submit_prefetch_task(
        single_row_spec([key(1)], LAYOUT, allowed_l2_memory_tags=frozenset())
    )
    result = store.query_prefetch_status(handle)
    assert result is not None and result.hit_cells[0].popcount() == 1
    assert result.l1_owners == {key(1): manager.l1_manager_id}
    with store.read_prefetched_results([key(1)], result.l1_owners) as read:
        assert read is not None
        tensor = read[0].tensor
        assert tensor is not None and torch.all(tensor == 7)
    store.finish_read_prefetched([key(1)], l1_owners=result.l1_owners)
    assert manager.report_status()["read_locked_count"] == 0
    assert store.memcheck()
    assert store.get_l1_usage() == (4096, 4096)
    assert store.report_status()["l1_manager"] == manager.report_status()
    store.publish_capacity()


def test_single_l1_completion_trace_replays_without_process_local_owners(
    storage_factory: StorageFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completion keeps the key-only trace schema and replays on a fresh L1."""
    source, (source_l1,) = storage_factory((4096,))
    events: list[Event] = []
    bus = Mock()
    bus.publish.side_effect = events.append
    monkeypatch.setattr(trace_decorator, "get_event_bus", lambda: bus)
    trace_decorator.set_tracing_enabled(True)
    try:
        objects = source.reserve_write([key(1)], LAYOUT)
        source.finish_write(list(objects), l1_owners_of(objects))
    finally:
        trace_decorator.set_tracing_enabled(False)

    prefix = "lmcache.v1.distributed.storage_manager.StorageManager."
    assert [event.metadata["qualname"] for event in events] == [
        prefix + "reserve_write",
        prefix + "finish_write",
    ]
    assert events[-1].metadata["args"] == {"keys": [key(1)]}
    target, (target_l1,) = storage_factory((4096,))
    assert target_l1.l1_manager_id != source_l1.l1_manager_id
    dispatcher = build_default_dispatcher()
    replay = ReplayContext(target)
    for event in events:
        qualname = event.metadata["qualname"]
        assert dispatcher.has(qualname)
        payload = msgspec.msgpack.encode(codecs.encode_args(event.metadata["args"]))
        dispatcher.dispatch(
            qualname,
            replay,
            codecs.decode_args(msgspec.msgpack.decode(payload)),
        )
    assert target_l1.get_object_state(key(1)) is not None


def test_read_release_trace_redacts_owners_and_replays_on_single_l1(
    storage_factory: StorageFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Owner redaction preserves live release and a replayable on-disk record."""
    source, (source_l1,) = storage_factory((4096,))
    target, (target_l1,) = storage_factory((4096,))
    keys = [key(1)]
    for store, manager in ((source, source_l1), (target, target_l1)):
        objects = store.reserve_write(keys, LAYOUT)
        store.finish_write(list(objects), l1_owners_of(objects))
        assert manager.reserve_read(keys, read_locks=2)[keys[0]][0] == L1Error.SUCCESS
        assert manager.report_status()["read_locked_count"] == 1
    assert target_l1.l1_manager_id != source_l1.l1_manager_id
    owners = {keys[0]: source_l1.l1_manager_id}
    release = Mock(wraps=source._finish_read_objects)
    monkeypatch.setattr(source, "_finish_read_objects", release)

    path = str(tmp_path / "read-release.lct")
    saved_gate = trace_decorator.is_tracing_enabled()
    recorder = StorageTraceRecorder(path)
    bus = Mock()
    # Deliver synchronously through the production subscriber, without a
    # drain-thread sleep or an alternate serialization path.
    bus.publish.side_effect = recorder.get_subscriptions()[EventType.TRACE_CALL]
    try:
        with monkeypatch.context() as capture:
            capture.setattr(trace_decorator, "get_event_bus", lambda: bus)
            source.finish_read_prefetched(keys, read_locks=2, l1_owners=owners)
    finally:
        recorder.close()
        trace_decorator.set_tracing_enabled(saved_gate)

    release.assert_called_once_with(keys, 2, owners)
    assert release.call_args.args[0] is keys
    assert release.call_args.args[2] is owners
    assert source_l1.report_status()["read_locked_count"] == 0
    with TraceReader(path) as reader:
        records = list(reader.records())
    qualname = (
        "lmcache.v1.distributed.storage_manager.StorageManager.finish_read_prefetched"
    )
    assert [record.qualname for record in records] == [qualname]
    bus.publish.assert_called_once()
    event = bus.publish.call_args.args[0]
    assert event.metadata["qualname"] == qualname
    assert event.metadata["args"] == {"keys": keys, "read_locks": 2}
    decoded = codecs.decode_args(records[0].args)
    assert decoded == {"keys": keys, "read_locks": 2}
    assert recorder.dropped_count == 0

    dispatcher = build_default_dispatcher()
    assert dispatcher.has(qualname)
    dispatcher.dispatch(qualname, ReplayContext(target), decoded)
    assert target_l1.report_status()["read_locked_count"] == 0


def test_overlapping_reads_preserve_each_selected_owner(
    storage_factory: StorageFactory,
) -> None:
    store, (primary, fallback) = storage_factory((4096, 4096))
    # The first lookup retains the fallback copy. A later store creates the
    # same key in the primary while that earlier read remains in flight.
    fallback.reserve_write([key(1)], [False], LAYOUT)
    fallback.finish_write([key(1)])
    first = store.query_prefetch_status(
        store.submit_prefetch_task(
            single_row_spec([key(1)], LAYOUT, allowed_l2_memory_tags=frozenset())
        )
    )
    primary.reserve_write([key(1)], [False], LAYOUT)
    primary.finish_write([key(1)])
    second = store.query_prefetch_status(
        store.submit_prefetch_task(
            single_row_spec([key(1)], LAYOUT, allowed_l2_memory_tags=frozenset())
        )
    )
    assert first is not None and second is not None
    assert first.l1_owners == {key(1): fallback.l1_manager_id}
    assert second.l1_owners == {key(1): primary.l1_manager_id}
    store.finish_read_by_owner(
        store.prepare_read_completion([key(1)], second.l1_owners)
    )
    assert primary.report_status()["read_locked_count"] == 0
    assert fallback.report_status()["read_locked_count"] == 1
    store.finish_read_by_owner(store.prepare_read_completion([key(1)], first.l1_owners))
    assert fallback.report_status()["read_locked_count"] == 0


def test_failed_peer_lookup_releases_earlier_locks(
    storage_factory: StorageFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, (primary, fallback) = storage_factory((4096, 4096))
    objects = store.reserve_write([key(1)], LAYOUT)
    store.finish_write(list(objects), l1_owners_of(objects))
    monkeypatch.setattr(
        fallback, "reserve_read", Mock(side_effect=RuntimeError("lookup failed"))
    )
    with pytest.raises(RuntimeError, match="lookup failed"):
        store.submit_prefetch_task(single_row_spec([key(1)], LAYOUT, num_kv_readers=4))
    assert primary.report_status()["read_locked_count"] == 0
    assert store.memcheck()


def test_combined_read_completion_keeps_duplicate_key_owners(
    storage_factory: StorageFactory,
) -> None:
    store, managers = storage_factory((4096, 4096))
    completion = []
    for manager in managers:
        manager.reserve_write([key(1)], [False], LAYOUT)
        manager.finish_write([key(1)])
        manager.reserve_read([key(1)])
        completion.append((manager.l1_manager_id, [key(1)]))
    store.finish_read_by_owner(completion)
    assert all(
        manager.report_status()["read_locked_count"] == 0 for manager in managers
    )


@pytest.mark.parametrize("locked_index", [0, 1])
def test_delete_reports_a_key_as_skipped_while_any_copy_is_locked(
    storage_factory: StorageFactory, locked_index: int
) -> None:
    store, managers = storage_factory((4096, 4096))
    keys = [key(1)]
    for manager in managers:
        manager.reserve_write(keys, [False], LAYOUT)
        manager.finish_write(keys)
    managers[locked_index].reserve_read(keys)
    assert store.delete_l1_keys(keys) == (0, 1)
    managers[locked_index].finish_read(keys)
    assert store.delete_l1_keys(keys) == (1, 0)
