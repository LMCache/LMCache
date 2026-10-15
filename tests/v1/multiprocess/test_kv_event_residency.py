# SPDX-License-Identifier: Apache-2.0
"""Pure committed-residency contracts without a server, EventBus or engine."""

# Standard
from dataclasses import FrozenInstanceError, replace

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.kv_event_residency import (
    ChunkIdentity,
    CommittedKVResidency,
    KVResidencyTransition,
)
from lmcache.v1.multiprocess.kv_event_types import EventBinding, KVEventProfile

PROFILE = KVEventProfile("native-test", 2)
BINDING = EventBinding(PROFILE.profile_id, (10, 11), None, (0, 1, 2, 3), 2)
IDENTITY: ChunkIdentity = ("model", 2, "", b"chunk")
PENDING = ("request", "transfer")
REQUIRED = frozenset(
    (ObjectKey.ComputeKVRank(2, rank, 2, rank), group)
    for rank in range(2)
    for group in range(2)
)


def keys(identity: ChunkIdentity = IDENTITY) -> list[ObjectKey]:
    """Return every registered shard of one chunk."""
    model, world, salt, chunk = identity
    return [
        ObjectKey(
            chunk, model, ObjectKey.ComputeKVRank(world, rank, world, rank), group, salt
        )
        for rank in range(world)
        for group in range(2)
    ]


@pytest.fixture
def residency() -> CommittedKVResidency:
    state = CommittedKVResidency()
    state.register_profile("model", 2, PROFILE, REQUIRED)
    return state


@pytest.mark.parametrize("binding_first", [False, True])
def test_complete_geometry_and_both_join_orders(
    residency: CommittedKVResidency, binding_first: bool
) -> None:
    objects = keys()
    if binding_first:
        residency.bind([(IDENTITY, BINDING)], PENDING)
        assert residency.refresh([IDENTITY]) == []
    assert residency.commit(objects[:-1], "cpu0") == []
    transitions = residency.commit(objects[-1:], "cpu0")
    if not binding_first:
        assert transitions == []
        residency.bind([(IDENTITY, BINDING)], PENDING)
        transitions = residency.refresh([IDENTITY])
    assert transitions == [KVResidencyTransition("stored", IDENTITY, BINDING)]
    assert residency.commit(objects, "cpu0") == residency.refresh([IDENTITY]) == []
    assert residency.finish(PENDING) == []
    assert residency.evict(objects, "cpu0") == [
        KVResidencyTransition("removed", IDENTITY, BINDING)
    ]
    assert residency.report_status() == {
        "bindings": 0,
        "live_objects": 0,
        "pending_stores": 0,
    }


def test_copy_ownership_and_store_remove_store_order(
    residency: CommittedKVResidency,
) -> None:
    objects = keys()
    residency.bind([(IDENTITY, BINDING)], PENDING)
    transitions = residency.commit(objects, "cpu0")
    assert residency.commit(objects, "cpu1") == []
    assert residency.commit(objects, "cpu1") == []
    assert residency.evict(objects, "cpu0") == []
    assert residency.evict(objects, "cpu0") == []
    transitions += residency.evict(objects[:1], "cpu1")
    assert residency.evict(objects[:1], "cpu1") == []
    transitions += residency.commit(objects[:1], "cpu1")
    assert [t.kind for t in transitions] == ["stored", "removed", "stored"]
    assert all(t.binding is BINDING for t in transitions)


def test_pending_and_partial_residency_bound_metadata(
    residency: CommittedKVResidency,
) -> None:
    second = ("request", "second-transfer")
    residency.bind([(IDENTITY, BINDING)], PENDING)
    residency.bind([(IDENTITY, BINDING)], PENDING)
    residency.bind([(IDENTITY, BINDING)], second)
    assert residency.report_status()["pending_stores"] == 2
    assert residency.finish(PENDING) == residency.finish(PENDING) == []
    assert residency.commit(keys()[:1], "cpu0") == []
    assert residency.finish(second) == []
    assert residency.report_status() == {
        "bindings": 1,
        "live_objects": 1,
        "pending_stores": 0,
    }
    assert residency.evict(keys()[:1], "cpu0") == []
    assert residency.report_status()["bindings"] == 0
    residency.bind([(IDENTITY, BINDING)], PENDING)
    assert residency.finish(PENDING) == []  # Failed write, with no committed object.
    assert residency.report_status()["bindings"] == 0


def test_conflict_rejects_whole_bind_batch(residency: CommittedKVResidency) -> None:
    residency.bind([(IDENTITY, BINDING)], PENDING)
    fresh: ChunkIdentity = ("model", 2, "", b"fresh")
    with pytest.raises(ValueError, match="conflicting"):
        residency.bind(
            [(fresh, BINDING), (IDENTITY, replace(BINDING, block_hashes=(20, 21)))],
            ("other-request", "other-transfer"),
        )
    assert residency.report_status() == {
        "bindings": 1,
        "live_objects": 0,
        "pending_stores": 1,
    }
    assert residency.commit(keys(fresh), "cpu0") == []


@pytest.mark.parametrize(
    "other",
    [
        ("other-model", 2, "", b"chunk"),
        ("model", 1, "", b"chunk"),
        ("model", 2, "other-salt", b"chunk"),
    ],
)
def test_profiles_and_namespaces_remain_separate(
    residency: CommittedKVResidency,
    other: ChunkIdentity,
) -> None:
    residency.register_profile("model", 2, PROFILE, REQUIRED)
    for profile, required in (
        (KVEventProfile("different", 2), REQUIRED),
        (PROFILE, frozenset(list(REQUIRED)[:1])),
    ):
        with pytest.raises(ValueError, match="mismatch"):
            residency.register_profile("model", 2, profile, required)
    residency.register_profile(
        other[0],
        other[1],
        PROFILE,
        frozenset((key.kv_rank, key.object_group_id) for key in keys(other)),
    )
    residency.bind([(IDENTITY, BINDING), (other, BINDING)], PENDING)
    assert residency.commit(keys(), "cpu0") == [
        KVResidencyTransition("stored", IDENTITY, BINDING)
    ]
    assert residency.refresh([other]) == []
    assert residency.commit(keys(other), "cpu0") == [
        KVResidencyTransition("stored", other, BINDING)
    ]


def test_reset_requires_new_complete_evidence(
    residency: CommittedKVResidency,
) -> None:
    residency.bind([(IDENTITY, BINDING)], PENDING)
    assert residency.commit(keys(), "cpu0")
    residency.reset()
    assert residency.report_status() == {
        "bindings": 0,
        "live_objects": 0,
        "pending_stores": 0,
    }
    with pytest.raises(ValueError, match="mismatch"):
        residency.register_profile("model", 2, KVEventProfile("other", 2), REQUIRED)
    residency.bind([(IDENTITY, BINDING)], ("request", "new-transfer"))
    assert residency.finish(PENDING) == []
    assert residency.refresh([IDENTITY]) == []
    assert residency.commit(keys()[:-1], "cpu0") == []
    assert residency.commit(keys()[-1:], "cpu0") == [
        KVResidencyTransition("stored", IDENTITY, BINDING)
    ]


def test_live_child_and_returned_descriptors_survive_cleanup(
    residency: CommittedKVResidency,
) -> None:
    child: ChunkIdentity = ("model", 2, "", b"child")
    descriptor = replace(
        BINDING, block_hashes=(12, 13), parent_block_hash=11, token_ids=(4, 5, 6, 7)
    )
    residency.bind([(IDENTITY, BINDING), (child, descriptor)], PENDING)
    transitions = residency.commit(keys() + keys(child), "cpu0")
    residency.finish(PENDING)
    residency.evict(keys(), "cpu0")
    for index in range(100):
        identity: ChunkIdentity = ("model", 2, "", str(index).encode())
        residency.bind([(identity, BINDING)], PENDING)
        residency.finish(PENDING)
    assert residency.report_status()["bindings"] == 1
    removed = residency.evict(keys(child), "cpu0")
    assert removed == [KVResidencyTransition("removed", child, descriptor)]
    assert transitions[1].binding is descriptor and descriptor.parent_block_hash == 11
    with pytest.raises(FrozenInstanceError):
        transitions[1].kind = "removed"  # type: ignore[misc]
