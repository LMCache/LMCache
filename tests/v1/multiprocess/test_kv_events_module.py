# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the server-side KV event channel (``POLL_KV_EVENTS``).

Cover the public contracts of the event log, the bus subscriber, and the
module handler. No GPU or live server is required.
"""

# Standard
from unittest.mock import MagicMock
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import L1BackendType, ObjectKey
from lmcache.v1.distributed.internal_api import L1ObjectMeta
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.multiprocess.custom_types import (
    KV_EVENT_CAPABILITY,
    KV_EVENT_KIND_REMOVED,
    KV_EVENT_KIND_STORED,
    KV_EVENT_MEDIUM_CPU,
    KV_EVENT_MEDIUM_STORAGE,
    KVEventPollResult,
)
from lmcache.v1.multiprocess.modules.kv_events import (
    KVEventLog,
    KVEventModule,
    KVEventSubscriber,
)
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.protocols.base import HandlerType, RequestType
from lmcache.v1.multiprocess.request_handler import iter_request_handlers

CHUNK_SIZE = 4
MODEL = "model-a"
OTHER_MODEL = "model-b"


def _hash(index: int) -> bytes:
    return bytes([index]) * 32


def _key(index: int, model: str = MODEL, kv_rank: int = 0, group: int = 0) -> ObjectKey:
    return ObjectKey(_hash(index), model, kv_rank, object_group_id=group)


def _tokens(index: int) -> list[int]:
    return [index * 10 + offset for offset in range(CHUNK_SIZE)]


def _bus(enabled: bool = True, max_queue_size: int = 10_000) -> EventBus:
    return EventBus(EventBusConfig(enabled=enabled, max_queue_size=max_queue_size))


def _subscriber(bus: EventBus | None = None) -> tuple[KVEventLog, KVEventSubscriber]:
    log = KVEventLog(capacity=64)
    return log, KVEventSubscriber(log, bus or _bus(), CHUNK_SIZE)


def _emit(subscriber: KVEventSubscriber, event_type: EventType, **metadata) -> None:
    subscriber.get_subscriptions()[event_type](Event(event_type, metadata=metadata))


def _bind(subscriber: KVEventSubscriber, *indices: int, start: int = 0) -> None:
    """Publish ``MP_TOKENS`` for chunks ``indices`` stored from token ``start``."""
    hashes = [_hash(i) for i in indices]
    parents: list[bytes | None] = [None] + hashes[:-1]
    if start > 0:
        parents[0] = _hash(99)
    _emit(
        subscriber,
        EventType.MP_TOKENS,
        chunk_hashes=hashes,
        token_chunks=[_tokens(i) for i in indices],
        token_offsets=[start + n * CHUNK_SIZE for n in range(len(indices))],
        parent_hashes=parents,
    )


def _l1_keys(*keys: ObjectKey) -> dict:
    return {
        "keys": list(keys),
        "meta": [L1ObjectMeta(size_bytes=8, backend=L1BackendType.DRAM) for _ in keys],
    }


# -- KVEventLog ---------------------------------------------------------------


def test_log_numbers_records_consecutively_and_reads_after_cursor() -> None:
    log = KVEventLog(capacity=8)
    first = log.append(KV_EVENT_KIND_REMOVED, KV_EVENT_MEDIUM_CPU, MODEL, [_hash(1)])
    second = log.append(KV_EVENT_KIND_REMOVED, KV_EVENT_MEDIUM_CPU, MODEL, [_hash(2)])
    assert (first.seq, second.seq, log.next_seq, len(log)) == (1, 2, 3, 2)

    records, next_cursor, lost = log.read_after(0, MODEL, max_events=10)
    assert [r.block_hashes for r in records] == [[_hash(1)], [_hash(2)]]
    assert (next_cursor, lost) == (2, False)

    assert log.read_after(2, MODEL, max_events=10) == ([], 2, False)
    records, next_cursor, lost = log.read_after(1, MODEL, max_events=10)
    assert ([r.seq for r in records], next_cursor, lost) == ([2], 2, False)


def test_log_filters_by_model_but_advances_past_other_models() -> None:
    log = KVEventLog(capacity=8)
    log.append(KV_EVENT_KIND_REMOVED, KV_EVENT_MEDIUM_CPU, OTHER_MODEL, [_hash(1)])
    log.append(KV_EVENT_KIND_REMOVED, KV_EVENT_MEDIUM_CPU, MODEL, [_hash(2)])
    log.append(KV_EVENT_KIND_REMOVED, KV_EVENT_MEDIUM_CPU, OTHER_MODEL, [_hash(3)])

    records, next_cursor, lost = log.read_after(0, MODEL, max_events=10)
    assert [r.seq for r in records] == [2]
    assert (next_cursor, lost) == (3, False)


def test_log_pages_with_max_events() -> None:
    log = KVEventLog(capacity=8)
    for index in range(1, 6):
        log.append(KV_EVENT_KIND_REMOVED, KV_EVENT_MEDIUM_CPU, MODEL, [_hash(index)])

    records, next_cursor, lost = log.read_after(0, MODEL, max_events=2)
    assert ([r.seq for r in records], next_cursor, lost) == ([1, 2], 2, False)
    records, next_cursor, lost = log.read_after(next_cursor, MODEL, max_events=2)
    assert ([r.seq for r in records], next_cursor, lost) == ([3, 4], 4, False)
    records, next_cursor, lost = log.read_after(next_cursor, MODEL, max_events=2)
    assert ([r.seq for r in records], next_cursor, lost) == ([5], 5, False)


def test_log_reports_loss_when_the_cursor_predates_retained_records() -> None:
    log = KVEventLog(capacity=3)
    for index in range(1, 6):
        log.append(KV_EVENT_KIND_REMOVED, KV_EVENT_MEDIUM_CPU, MODEL, [_hash(index)])

    # Records 1 and 2 were discarded.
    records, next_cursor, lost = log.read_after(0, MODEL, max_events=10)
    assert ([r.seq for r in records], next_cursor, lost) == ([3, 4, 5], 5, True)
    _, _, lost = log.read_after(1, MODEL, max_events=10)
    assert lost
    # A reader that consumed record 2 missed nothing.
    records, _, lost = log.read_after(2, MODEL, max_events=10)
    assert ([r.seq for r in records], lost) == ([3, 4, 5], False)


def test_log_reports_loss_for_a_cursor_it_never_issued() -> None:
    log = KVEventLog(capacity=3)
    assert log.read_after(0, MODEL, max_events=1) == ([], 0, False)
    assert log.read_after(5, MODEL, max_events=1) == ([], 0, True)
    log.append(KV_EVENT_KIND_REMOVED, KV_EVENT_MEDIUM_CPU, MODEL, [_hash(1)])
    assert log.read_after(5, MODEL, max_events=1) == ([], 1, True)


def test_log_loss_marker_flags_loss_and_drops_earlier_records() -> None:
    log = KVEventLog(capacity=8)
    log.append(KV_EVENT_KIND_REMOVED, KV_EVENT_MEDIUM_CPU, MODEL, [_hash(1)])
    log.note_dropped_events(total_dropped=1)
    log.note_dropped_events(total_dropped=1)  # unchanged count: no new marker
    log.append(KV_EVENT_KIND_REMOVED, KV_EVENT_MEDIUM_CPU, MODEL, [_hash(2)])
    assert log.lost_markers == 1

    records, next_cursor, lost = log.read_after(0, MODEL, max_events=10)
    assert ([r.block_hashes for r in records], next_cursor, lost) == (
        [[_hash(2)]],
        3,
        True,
    )
    # Past the marker, reads are exact again.
    assert log.read_after(3, MODEL, max_events=10) == ([], 3, False)


def test_log_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="capacity"):
        KVEventLog(capacity=0)
    log = KVEventLog(capacity=1)
    with pytest.raises(ValueError, match="cursor"):
        log.read_after(-1, MODEL, max_events=1)
    with pytest.raises(ValueError, match="max_events"):
        log.read_after(0, MODEL, max_events=0)


# -- KVEventSubscriber --------------------------------------------------------


def test_subscriber_records_l1_stores_with_their_token_bindings() -> None:
    log, subscriber = _subscriber()
    _bind(subscriber, 1, 2)
    # Both KV ranks and a second object group store the same two chunks.
    _emit(
        subscriber,
        EventType.L1_WRITE_FINISHED,
        **_l1_keys(_key(1), _key(1, kv_rank=1), _key(2), _key(2, group=1)),
    )

    records, _, _ = log.read_after(0, MODEL, max_events=10)
    assert [(r.kind, r.medium, r.block_hashes) for r in records] == [
        (KV_EVENT_KIND_STORED, KV_EVENT_MEDIUM_CPU, [_hash(1)]),
        (KV_EVENT_KIND_STORED, KV_EVENT_MEDIUM_CPU, [_hash(2)]),
    ]
    assert (records[0].parent_block_hash, records[0].token_ids) == (None, _tokens(1))
    assert (records[1].parent_block_hash, records[1].token_ids) == (
        _hash(1),
        _tokens(2),
    )
    assert all(r.block_size == CHUNK_SIZE for r in records)
    assert subscriber.unbound_stores == 0


def test_subscriber_skips_and_counts_stores_without_a_binding() -> None:
    log, subscriber = _subscriber()
    _emit(
        subscriber,
        EventType.L1_WRITE_FINISHED_AND_READ_RESERVED,
        **_l1_keys(_key(5)),
    )
    assert log.read_after(0, MODEL, max_events=10) == ([], 0, False)
    assert subscriber.unbound_stores == 1


def test_subscriber_binds_a_mid_sequence_store_to_its_predecessor() -> None:
    log, subscriber = _subscriber()
    _bind(subscriber, 3, 4, start=2 * CHUNK_SIZE)
    _emit(subscriber, EventType.L1_WRITE_FINISHED, **_l1_keys(_key(3), _key(4)))

    records, _, _ = log.read_after(0, MODEL, max_events=10)
    assert [r.parent_block_hash for r in records] == [_hash(99), _hash(3)]


def test_subscriber_without_parent_hashes_leaves_unknown_predecessors_unbound() -> None:
    """Older producers omit ``parent_hashes``: a first chunk that does not
    start the sequence must not be reported as a sequence start."""
    log, subscriber = _subscriber()
    _emit(
        subscriber,
        EventType.MP_TOKENS,
        chunk_hashes=[_hash(3), _hash(4)],
        token_chunks=[_tokens(3), _tokens(4)],
        token_offsets=[2 * CHUNK_SIZE, 3 * CHUNK_SIZE],
    )
    _emit(subscriber, EventType.L1_WRITE_FINISHED, **_l1_keys(_key(3), _key(4)))

    records, _, _ = log.read_after(0, MODEL, max_events=10)
    assert [(r.block_hashes, r.parent_block_hash) for r in records] == [
        ([_hash(4)], _hash(3))
    ]
    assert subscriber.unbound_stores == 1


def test_subscriber_records_removals_per_model_without_duplicates() -> None:
    log, subscriber = _subscriber()
    _emit(
        subscriber,
        EventType.L1_KEYS_EVICTED,
        **_l1_keys(
            _key(1),
            _key(1, kv_rank=1),
            _key(2, group=1),
            _key(3, model=OTHER_MODEL),
        ),
    )

    records, _, _ = log.read_after(0, MODEL, max_events=10)
    assert [(r.kind, r.medium, r.block_hashes, r.token_ids) for r in records] == [
        (KV_EVENT_KIND_REMOVED, KV_EVENT_MEDIUM_CPU, [_hash(1), _hash(2)], [])
    ]
    records, _, _ = log.read_after(0, OTHER_MODEL, max_events=10)
    assert [r.block_hashes for r in records] == [[_hash(3)]]


def test_subscriber_records_l2_stores_and_deletes_as_storage() -> None:
    log, subscriber = _subscriber()
    _bind(subscriber, 1)
    _emit(
        subscriber,
        EventType.L2_KEYS_STORED,
        keys=[_key(1)],
        sizes=[8],
        backend="fs",
        shared=False,
    )
    _emit(
        subscriber,
        EventType.L2_KEYS_DELETED,
        keys=[_key(1)],
        backend="fs",
        shared=False,
    )

    records, _, _ = log.read_after(0, MODEL, max_events=10)
    assert [(r.kind, r.medium) for r in records] == [
        (KV_EVENT_KIND_STORED, KV_EVENT_MEDIUM_STORAGE),
        (KV_EVENT_KIND_REMOVED, KV_EVENT_MEDIUM_STORAGE),
    ]
    assert records[0].token_ids == _tokens(1)


def test_subscriber_flags_events_the_bus_dropped() -> None:
    bus = _bus(max_queue_size=1)  # not started: the second publish is dropped
    log, subscriber = _subscriber(bus)
    _bind(subscriber, 1)
    _emit(subscriber, EventType.L1_WRITE_FINISHED, **_l1_keys(_key(1)))
    bus.publish(Event(EventType.L1_KEYS_EVICTED, metadata=_l1_keys(_key(1))))
    bus.publish(Event(EventType.L1_KEYS_EVICTED, metadata=_l1_keys(_key(1))))
    assert bus.dropped_events_count() == 1

    _emit(subscriber, EventType.L1_WRITE_FINISHED, **_l1_keys(_key(2)))
    records, _, lost = log.read_after(0, MODEL, max_events=10)
    assert lost
    assert [r.block_hashes for r in records] == []
    assert log.lost_markers == 1


# -- KVEventModule ------------------------------------------------------------


def _context(bus: EventBus) -> MagicMock:
    ctx = MagicMock(name="ctx")
    ctx.event_bus = bus
    ctx.chunk_size = CHUNK_SIZE
    return ctx


def test_module_exposes_a_sync_poll_handler() -> None:
    handlers = iter_request_handlers(KVEventModule)
    assert [(h.options.request_type, h.options.handler_type) for h in handlers] == [
        (RequestType.POLL_KV_EVENTS, HandlerType.SYNC)
    ]


def test_module_is_disabled_without_a_log_or_an_enabled_bus() -> None:
    off = KVEventModule(_context(_bus(enabled=False)), log_size=8)
    zero = KVEventModule(_context(_bus()), log_size=0)
    for module in (off, zero):
        assert not module.enabled
        result = module.poll_kv_events(MODEL, 0, 8)
        assert isinstance(result, KVEventPollResult)
        assert (result.enabled, result.events) == (False, [])
        assert module.report_status()["kv_events"]["enabled"] is False
    with pytest.raises(ValueError, match="log_size"):
        KVEventModule(_context(_bus()), log_size=-1)


def test_module_validates_poll_arguments() -> None:
    module = KVEventModule(_context(_bus()), log_size=8)
    with pytest.raises(ValueError, match="cursor"):
        module.poll_kv_events(MODEL, -1, 8)
    with pytest.raises(ValueError, match="max_events"):
        module.poll_kv_events(MODEL, 0, 0)


def test_module_serves_bus_events_end_to_end() -> None:
    bus = _bus()
    bus.start()
    try:
        module = KVEventModule(_context(bus), log_size=16)
        assert module.enabled
        bus.publish(
            Event(
                EventType.MP_TOKENS,
                metadata={
                    "chunk_hashes": [_hash(1)],
                    "token_chunks": [_tokens(1)],
                    "token_offsets": [0],
                    "parent_hashes": [None],
                },
            )
        )
        bus.publish(Event(EventType.L1_WRITE_FINISHED, metadata=_l1_keys(_key(1))))
        bus.publish(Event(EventType.L1_KEYS_EVICTED, metadata=_l1_keys(_key(1))))

        deadline = time.monotonic() + 5.0
        while True:
            result = module.poll_kv_events(MODEL, 0, 8)
            if len(result.events) == 2 or time.monotonic() > deadline:
                break
            time.sleep(0.01)
        assert result.enabled and not result.lost
        assert result.incarnation == module.incarnation
        assert [(r.kind, r.block_hashes) for r in result.events] == [
            (KV_EVENT_KIND_STORED, [_hash(1)]),
            (KV_EVENT_KIND_REMOVED, [_hash(1)]),
        ]
        assert result.next_cursor == 2

        again = module.poll_kv_events(MODEL, result.next_cursor, 8)
        assert (again.events, again.next_cursor, again.incarnation) == (
            [],
            2,
            module.incarnation,
        )
        status = module.report_status()["kv_events"]
        assert status["enabled"] and status["log_depth"] == 2
        assert status["log_capacity"] == 16 and status["next_seq"] == 3
    finally:
        bus.stop()


def test_module_reports_bus_drops_as_lost_on_the_next_poll() -> None:
    bus = _bus(max_queue_size=1)  # not started: the second publish is dropped
    module = KVEventModule(_context(bus), log_size=16)
    bus.publish(Event(EventType.L1_KEYS_EVICTED, metadata=_l1_keys(_key(1))))
    bus.publish(Event(EventType.L1_KEYS_EVICTED, metadata=_l1_keys(_key(1))))

    result = module.poll_kv_events(MODEL, 0, 8)
    assert (result.lost, result.events, result.next_cursor) == (True, [], 1)
    assert module.poll_kv_events(MODEL, result.next_cursor, 8).lost is False


def test_the_kv_event_capability_is_advertised_to_engines() -> None:
    """An engine polls only a server that advertises the channel, so the
    management module must carry the flag alongside the transfer types."""
    advertising = ManagementModule(_context(_bus()), capabilities=[KV_EVENT_CAPABILITY])
    silent = ManagementModule(_context(_bus()))

    assert KV_EVENT_CAPABILITY in advertising.get_experimental()
    assert KV_EVENT_CAPABILITY not in silent.get_experimental()
