# SPDX-License-Identifier: Apache-2.0
"""Native identity joins committed rank/group/copy residency, never STORE intent."""

# Standard
from collections.abc import Iterator
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from typing import cast

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc, MemoryLayoutDesc, ObjectKey
from lmcache.v1.mp_observability.event import Event, EventType, next_transfer_key
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.multiprocess.custom_types import (
    EventBinding,
    IPCCacheServerKey,
    KVEventBatch,
    KVEventProfile,
    KVEventRecord,
    NativeBlockHash,
    get_customized_decoder,
    get_customized_encoder,
)
from lmcache.v1.multiprocess.engine_context import (
    LayoutDescRegistry,
    MPCacheServerContext,
)
from lmcache.v1.multiprocess.modules.kv_events import KVEventModule
from lmcache.v1.multiprocess.token_codec import pack_token_ids
from lmcache.v1.multiprocess.transport.grpc_impl.method_registry import (
    get_method_codec_registry,
)

PROFILE = KVEventProfile("native-test", 2)
BINDING = EventBinding(PROFILE.profile_id, (10, 11), None, (0, 1, 2, 3), 2)
KEY = IPCCacheServerKey(
    "model",
    2,
    0,
    pack_token_ids(tuple(range(8))),
    0,
    4,
    "request",
    event_profile=PROFILE,
    event_bindings=(BINDING,),
)


@pytest.fixture
def cache() -> Iterator[tuple[KVEventModule, EventBus]]:
    """Use the real event projection with a small deterministic bus and log."""
    bus = EventBus(EventBusConfig(max_queue_size=4))
    registry = LayoutDescRegistry()
    for world in (1, 2):
        registry.register(
            "model", world, MemoryLayoutDesc([], []), AttnWindowDesc([-1, -1])
        )
    context = SimpleNamespace(
        event_bus=bus,
        chunk_size=4,
        layout_desc_registry=registry,
        validate_native_event_storage=lambda: None,
    )
    module = KVEventModule(cast(MPCacheServerContext, context), kv_event_log_size=16)
    yield module, bus
    module.close()
    bus.stop()


def keys(chunk: bytes = b"chunk", world: int = 2) -> list[ObjectKey]:
    """Return every required rank/group key for one fixture chunk."""
    return [
        ObjectKey(
            chunk, "model", ObjectKey.ComputeKVRank(world, rank, world, rank), group
        )
        for rank in range(world)
        for group in range(2)
    ]


def publish(
    bus: EventBus, kind: EventType, objects: list[ObjectKey], owner: str = "cpu0"
) -> None:
    """Drain a real commit/eviction notification before inspecting the projection."""
    bus.publish(Event(kind, metadata={"keys": objects, "l1_tag": owner}))
    bus.stop()


def finish(bus: EventBus, transfer: str = "store") -> None:
    """Retire pending ownership through the host-completion marker."""
    bus.publish(
        Event(
            EventType.MP_TOKENS,
            session_id="request",
            metadata={
                "native_store_finished": transfer,
                "chunk_hashes": [],
                "token_chunks": [],
                "token_offsets": [],
                "parent_hashes": [],
            },
        )
    )
    bus.stop()


def events(module: KVEventModule) -> list[KVEventRecord]:
    """Inspect retained records without consuming their replay history."""
    return module.read_kv_events("model", 0, 100, 2).events


@pytest.mark.parametrize("binding_first", [False, True])
def test_complete_ranks_groups_copies_and_both_join_orders(
    cache: tuple[KVEventModule, EventBus], binding_first: bool
) -> None:
    module, bus = cache
    objects = keys()
    if binding_first:
        module.bind_store(KEY, objects[:1], "store")
    publish(bus, EventType.L1_WRITE_FINISHED, objects[:-1])
    assert events(module) == []
    publish(bus, EventType.L1_WRITE_FINISHED_AND_READ_RESERVED, objects[-1:])
    if not binding_first:
        assert events(module) == []
        module.bind_store(KEY, objects[:1], "store")
        bus.stop()
    finish(bus)
    (stored,) = events(module)
    assert (stored.block_hashes, stored.token_ids, stored.block_size) == (
        (10, 11),
        (0, 1, 2, 3),
        2,
    )
    # Duplicate metadata/commits and another readable L1 copy are idempotent.
    module.bind_store(KEY, objects[:1], "duplicate")
    publish(bus, EventType.L1_WRITE_FINISHED, objects)
    publish(bus, EventType.L1_WRITE_FINISHED, objects, "cpu1")
    finish(bus, "duplicate")
    publish(bus, EventType.L1_KEYS_EVICTED, objects)
    assert events(module) == [stored]
    publish(bus, EventType.L1_KEYS_EVICTED, objects[-1:], "cpu1")
    publish(bus, EventType.L1_WRITE_FINISHED, objects[-1:], "cpu1")
    assert [e.kind for e in events(module)] == ["stored", "removed", "stored"]
    assert all(e.block_hashes == stored.block_hashes for e in events(module))
    publish(bus, EventType.L1_KEYS_EVICTED, objects, "cpu1")
    assert module.report_status()["kv_events"]["bindings"] == 0


def test_failed_pending_cleanup_and_immutable_child_replay(
    cache: tuple[KVEventModule, EventBus],
) -> None:
    module, bus = cache
    parent, child = keys(), keys(b"child")
    descriptor = replace(
        BINDING, block_hashes=(14, 15), parent_block_hash=11, token_ids=(4, 5, 6, 7)
    )
    module.bind_store(KEY, parent[:1], "parent")
    module.bind_store(
        replace(KEY, start=4, end=8, event_bindings=(descriptor,)), child[:1], "child"
    )
    publish(bus, EventType.L1_WRITE_FINISHED, parent + child)
    finish(bus, "parent")
    finish(bus, "child")
    original = events(module)[1]
    for index in range(100):
        transfers = [next_transfer_key(KEY.request_id) for _ in range(2)]
        for transfer in transfers:
            module.bind_store(KEY, keys(str(index).encode())[:1], transfer)
        if index % 2:
            transfers.reverse()
        first, last = transfers
        # Duplicate cleanup cannot release another transfer's uncommitted binding.
        for _ in range(2):
            finish(bus, first)
            status = module.report_status()["kv_events"]
            assert status["pending_stores"] == 1 and status["bindings"] == 3
        finish(bus, last)
    assert module.report_status()["kv_events"]["pending_stores"] == 0
    assert len(events(module)) == 2  # Failed stores never become placements.
    publish(bus, EventType.L1_KEYS_EVICTED, parent)
    publish(bus, EventType.L1_KEYS_EVICTED, child)
    assert original.parent_block_hash == 11 and original.token_ids == (4, 5, 6, 7)
    assert events(module)[-1].block_hashes == original.block_hashes
    with pytest.raises(FrozenInstanceError):
        original.block_size = 4  # type: ignore[misc]
    assert module.report_status()["kv_events"]["bindings"] == 0


def test_binding_conflicts_and_context_isolation(
    cache: tuple[KVEventModule, EventBus],
) -> None:
    module, bus = cache
    module.bind_store(KEY, keys()[:1], "store")
    with pytest.raises(ValueError, match="conflicting"):
        module.bind_store(
            replace(KEY, event_bindings=(replace(BINDING, block_hashes=(99, 100)),)),
            keys()[:1],
            "bad",
        )
    with pytest.raises(ValueError, match="profile.*mismatch"):
        module.register_profile("model", 2, KVEventProfile("other", 2))
    with pytest.raises(ValueError, match="cache_salt"):
        module.bind_store(replace(KEY, cache_salt="tenant"), keys()[:1], "salt")
    publish(bus, EventType.L1_WRITE_FINISHED, keys(world=1))
    assert (
        events(module) == []
    )  # Same chunk in another layout cannot complete this one.


@pytest.mark.parametrize(
    "world,block,attention",
    [
        (2, 3, AttnWindowDesc([-1])),
        (0, 2, AttnWindowDesc([-1])),
        (2, 2, AttnWindowDesc([2])),
        (2, 2, AttnWindowDesc([])),
        (2, 2, AttnWindowDesc([-1], group_kinds=("recurrent",))),
    ],
)
def test_registration_rejects_unsupported_geometry(
    cache: tuple[KVEventModule, EventBus],
    world: int,
    block: int,
    attention: AttnWindowDesc,
) -> None:
    module, _ = cache
    with pytest.raises(ValueError):
        module.register_profile(
            "model", world, KVEventProfile("bad", block), attn_desc=attention
        )
    assert events(module) == []


def test_event_bus_loss_discards_projection_and_pre_loss_replay(
    cache: tuple[KVEventModule, EventBus],
) -> None:
    module, bus = cache
    for index in range(3):
        objects = keys(str(index).encode())
        module.bind_store(KEY, objects[:1], str(index))
        publish(bus, EventType.L1_WRITE_FINISHED, objects)
        finish(bus, str(index))
    assert len(events(module)) == 3
    # A queued completion must not advertise using a shard whose removal
    # was dropped before the loss barrier runs.
    module.bind_store(KEY, keys()[:1], "partial")
    publish(bus, EventType.L1_WRITE_FINISHED, keys()[:-1])
    observed = []
    bus.subscribe(
        EventType.L1_WRITE_FINISHED, lambda event: observed.extend(events(module)[3:])
    )
    queued = [(EventType.L1_WRITE_FINISHED, keys()[-1:])]
    queued += [(EventType.L1_KEYS_EVICTED, [])] * 3
    queued += [(EventType.L1_KEYS_EVICTED, keys()[:-1])]
    for kind, objects in queued:
        bus.publish(Event(kind, metadata={"keys": objects, "l1_tag": "cpu0"}))
    bus.stop()
    assert observed == []
    lost = module.read_kv_events("model", 0, 1, 2)
    assert lost.lost and not lost.events
    assert module.report_status()["kv_events"]["live_objects"] == 0
    settled = module.read_kv_events("model", lost.next_cursor, 1, 2)
    assert not settled.lost and not settled.events
    assert settled.next_cursor == lost.next_cursor
    # A later drop with an empty log still wakes a tail subscriber once.
    for _ in range(5):
        bus.publish(Event(EventType.MP_TOKENS, metadata={}))
    bus.stop()
    empty_loss = module.read_kv_events("model", lost.next_cursor, 1, 2)
    assert empty_loss.lost and not empty_loss.events
    assert empty_loss.next_cursor > lost.next_cursor
    lost = empty_loss
    module.bind_store(KEY, keys()[:1], "fresh")
    finish(bus, "partial")
    assert module.report_status()["kv_events"]["pending_stores"] == 1
    publish(bus, EventType.L1_WRITE_FINISHED, keys()[:-1])
    assert events(module) == []
    publish(bus, EventType.L1_WRITE_FINISHED, keys()[-1:])
    recovered = module.read_kv_events("model", lost.next_cursor, 1, 2)
    assert [e.kind for e in recovered.events] == ["stored"]
    assert recovered.events[0].seq > lost.next_cursor


@pytest.mark.parametrize(
    "hashes,parent",
    [
        ((b"a", b"b"), None),
        ((-(2**63), 2**63 - 1), -1),
        ((2**64 - 1, 2**63), 2**64 - 2),
    ],
)
def test_native_metadata_round_trips_all_wire_representations(
    hashes: tuple[NativeBlockHash, ...], parent: NativeBlockHash | None
) -> None:
    descriptor = replace(BINDING, block_hashes=hashes, parent_block_hash=parent)
    key = replace(KEY, event_bindings=(descriptor,))
    record = KVEventRecord(
        1,
        "stored",
        "CPU",
        "model",
        hashes,
        parent,
        BINDING.token_ids,
        2,
        PROFILE.profile_id,
        2,
    )
    batch = KVEventBatch(True, 1, 1, False, [record], schema_version=1)
    for value in (key, batch):
        decoded = get_customized_decoder(type(value)).decode(
            get_customized_encoder(type(value)).encode(value)
        )
        assert decoded == value
        if isinstance(value, IPCCacheServerKey):
            assert decoded.event_bindings == (descriptor,)
            assert decoded.event_profile == PROFILE
    codecs = get_method_codec_registry().by_full_name
    store = codecs["lmcache.mp.LMCacheDrivenService.Store"]
    decoded = store.request_decoder(
        store.request_encoder((key, 42, [[1]], b"event"), {})
    )[0]
    # IPC key equality deliberately excludes metadata; assert it separately.
    assert decoded.event_bindings == (descriptor,) and decoded.event_profile == PROFILE
    subscribe = codecs["lmcache.mp.KVEventService.SubscribeKvEvents"]
    assert subscribe.response_decoder(subscribe.response_encoder(batch)) == batch
