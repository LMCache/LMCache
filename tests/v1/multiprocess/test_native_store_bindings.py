# SPDX-License-Identifier: Apache-2.0
"""Native STORE metadata is retained independently of the legacy event records."""

# Standard
from dataclasses import replace
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.v1.common.custom_types import (
    EventBinding,
    IPCCacheServerKey,
    KVEventProfile,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.transport.errors import InvalidRequestError
from lmcache.v1.transport.grpc.method_registry import get_method_codec_registry
from tests.v1.multiprocess.test_worker_liveness import (
    _bind_kv,
    _event_key,
    _event_module,
    _publish_kv,
)

PROFILE = KVEventProfile("native-v1", 2)


@pytest.mark.parametrize("hashes", [(b"a", b"b"), (-4, -1), (2**64 - 2, 2**64 - 1)])
def test_store_binding_wire_preserves_native_hashes(hashes: tuple[Any, Any]) -> None:
    key = IPCCacheServerKey.from_token_ids(
        "model", 1, 0, (1, 2, 3, 4), start=0, end=4, request_id="id"
    )
    bound = replace(key, event_bindings=(EventBinding(PROFILE, hashes, hashes[0]),))
    assert bound == key
    assert not bound.no_worker_id_version().event_bindings
    codec = get_method_codec_registry().by_full_name[
        "lmcache.mp.LMCacheDrivenService.Store"
    ]
    for original in (key, bound):
        wire = codec.request_encoder((original, 3, [[7]], b"event"), {})
        restored, *_ = codec.request_decoder(wire)
        assert restored.event_bindings == original.event_bindings
        assert restored.token_bytes == original.token_bytes


@pytest.mark.parametrize(("first", "last"), [("a", "b"), ("b", "a")])
def test_binding_lifetime_follows_transfers_and_l1_owners(
    first: str, last: str
) -> None:
    module, bus = _event_module()
    obj = _event_key(1)
    binding = EventBinding(PROFILE, (3, 5), None)
    key = IPCCacheServerKey.from_token_ids(
        "model", 1, 0, (10, 11, 12, 13), start=0, end=4, request_id="id"
    )
    key = replace(key, event_bindings=(binding,))
    _bind_kv(bus, [obj])
    module.bind_store(key, [obj], PROFILE, 4, "a")
    module.bind_store(key, [obj], PROFILE, 4, "b")
    _publish_kv(bus, EventType.MP_STORE_BINDINGS_RELEASED, transfer_key=first)
    _publish_kv(bus, EventType.MP_STORE_BINDINGS_RELEASED, transfer_key=first)
    assert module.get_store_binding(obj) == (binding, key.token_bytes)
    _publish_kv(bus, EventType.L1_WRITE_FINISHED, keys=[obj], l1_tag="one")
    _publish_kv(bus, EventType.L1_WRITE_FINISHED, keys=[obj], l1_tag="two")
    _publish_kv(bus, EventType.MP_STORE_BINDINGS_RELEASED, transfer_key=last)
    _publish_kv(bus, EventType.L1_KEYS_EVICTED, keys=[obj], l1_tag="one")
    assert module.get_store_binding(obj) == (binding, key.token_bytes)
    _publish_kv(bus, EventType.L1_KEYS_EVICTED, keys=[obj], l1_tag="two")
    assert module.get_store_binding(obj) is None
    events = module.read_kv_events("model", 0, 8).events
    assert [(e.kind, e.block_hashes) for e in events] == [
        ("stored", [obj.chunk_hash]),
        ("removed", [obj.chunk_hash]),
    ]


def test_conflict_rejects_an_entire_store_before_binding() -> None:
    module, bus = _event_module()
    first, second = _event_key(1), _event_key(2)
    key = IPCCacheServerKey.from_token_ids(
        "model", 1, 0, (1, 2, 3, 4), start=0, end=4, request_id="first"
    )
    a = EventBinding(PROFILE, (10, 11), None)
    b = EventBinding(PROFILE, (22, 23), 11)
    module.bind_store(replace(key, event_bindings=(a,)), [second], PROFILE, 4, "first")
    both = IPCCacheServerKey.from_token_ids(
        "model", 1, 0, tuple(range(8)), start=0, end=8, request_id="next"
    )
    with pytest.raises(InvalidRequestError, match="conflict"):
        module.bind_store(
            replace(both, event_bindings=(a, b)), [first, second], PROFILE, 4, "next"
        )
    assert module.get_store_binding(first) is None
    assert module.get_store_binding(second) == (a, key.token_bytes)
    _publish_kv(bus, EventType.MP_STORE_BINDINGS_RELEASED, transfer_key="first")
    assert module.get_store_binding(second) is None


def test_binding_slices_only_complete_chunks_and_checks_registered_profile() -> None:
    module, _ = _event_module()
    obj = _event_key(1)
    b = EventBinding(PROFILE, (12, 13), 11)
    key = IPCCacheServerKey.from_token_ids(
        "model", 1, 0, tuple(range(10)), start=4, end=10, request_id="slice"
    )
    bound = replace(key, event_bindings=(b,))
    for profile in (None, KVEventProfile("other", 2), KVEventProfile("native-v1", 4)):
        with pytest.raises(InvalidRequestError, match="profile"):
            module.bind_store(bound, [obj], profile, 4, "bad")
    with pytest.raises(InvalidRequestError, match="parent"):
        module.bind_store(
            replace(bound, event_bindings=(replace(b, parent_block_hash=None),)),
            [obj],
            PROFILE,
            4,
            "bad",
        )
    with pytest.raises(InvalidRequestError, match="unsalted"):
        module.bind_store(replace(bound, cache_salt="other"), [obj], PROFILE, 4, "bad")
    assert module.get_store_binding(obj) is None
    module.bind_store(bound, [obj], PROFILE, 4, "good")
    assert module.get_store_binding(obj) == (b, key.token_bytes[16:32])


def test_event_loss_discards_native_binding_even_with_pending_store() -> None:
    module, bus = _event_module(bus=EventBus(EventBusConfig(max_queue_size=0)))
    obj = _event_key(1)
    b = EventBinding(PROFILE, (1, 2), None)
    key = IPCCacheServerKey.from_token_ids(
        "model", 1, 0, (1, 2, 3, 4), start=0, end=4, request_id="lost"
    )
    module.bind_store(replace(key, event_bindings=(b,)), [obj], PROFILE, 4, "old")
    bus.publish(
        Event(EventType.MP_STORE_BINDINGS_RELEASED, metadata={"transfer_key": "old"})
    )
    with pytest.raises(InvalidRequestError, match="loss recovery"):
        module.bind_store(replace(key, event_bindings=(b,)), [obj], PROFILE, 4, "new")
    bus.stop()
    assert module.read_kv_events("model", 0, 8).lost
    assert module.get_store_binding(obj) is None
