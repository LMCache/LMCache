# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator key directory (I1): the state it builds
from gate-admitted cache events, lookup, listing, and fencing cleanup.
Stream admission itself (seq dedup, gap detection, incarnation
comparison) is the gate's job -- see ``test_event_gate.py``."""

# Third Party
import numpy as np
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    UNKNOWN_TOKEN_OFFSET,
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.key_directory import KeyDirectory


def _key(hash_byte: int) -> ObjectKey:
    return ObjectKey(chunk_hash=bytes([hash_byte]) * 4, model_name="m", kv_rank=0)


def _batch(
    instance_id: str = "node-a",
    incarnation: int = 1,
    seq: int = 1,
    event_type: CacheEventType = CacheEventType.STORE,
    tier: Tier = Tier.L1,
    backend: str = "dram",
    keys: list[ObjectKey] | None = None,
    size_bytes: int = 1024,
    shared: bool = False,
    ts: float = 0.0,
    token_ids: list[int] | None = None,
    token_offset: int = 0,
) -> CacheEventBatch:
    entries = [
        CacheEventEntry(
            key=k.to_encoded_object_key(),
            size_bytes=size_bytes,
            token_ids=token_ids or [],
            token_offset=token_offset,
        )
        for k in (keys or [_key(0xAA)])
    ]
    return CacheEventBatch(
        instance_id=instance_id,
        incarnation=incarnation,
        seq=seq,
        event_type=event_type,
        tier=tier,
        backend=backend,
        entries=entries,
        shared=shared,
        ts=ts,
    )


# -- Store / lookup ----------------------------------------------------------


def test_store_then_lookup():
    directory = KeyDirectory()
    directory.consume(_batch(keys=[_key(1)]))

    [placements] = directory.lookup([_key(1)])
    assert len(placements) == 1
    p = placements[0]
    assert p.instance_id == "node-a"
    assert p.incarnation == 1
    assert p.tier == Tier.L1
    assert p.backend == "dram"
    assert p.size_bytes == 1024


def test_lookup_unknown_key_is_empty():
    directory = KeyDirectory()
    assert directory.lookup([_key(9)]) == [[]]


def test_lookup_preserves_request_order():
    directory = KeyDirectory()
    directory.consume(_batch(keys=[_key(2)]))
    results = directory.lookup([_key(1), _key(2)])
    assert results[0] == []
    assert len(results[1]) == 1


def test_restore_updates_size_without_duplicating_placement():
    directory = KeyDirectory()
    directory.consume(_batch(seq=1, keys=[_key(1)], size_bytes=100))
    directory.consume(_batch(seq=2, keys=[_key(1)], size_bytes=200))

    [placements] = directory.lookup([_key(1)])
    assert len(placements) == 1
    assert placements[0].size_bytes == 200


def test_same_key_on_two_instances():
    directory = KeyDirectory()
    directory.consume(_batch(instance_id="node-b", keys=[_key(1)]))
    directory.consume(_batch(instance_id="node-a", keys=[_key(1)]))

    [placements] = directory.lookup([_key(1)])
    assert [p.instance_id for p in placements] == ["node-a", "node-b"]


def test_same_key_on_two_tiers_of_one_instance():
    directory = KeyDirectory()
    directory.consume(_batch(seq=1, tier=Tier.L1, backend="dram", keys=[_key(1)]))
    directory.consume(_batch(seq=2, tier=Tier.L2, backend="fs", keys=[_key(1)]))

    [placements] = directory.lookup([_key(1)])
    assert [(p.tier, p.backend) for p in placements] == [
        (Tier.L1, "dram"),
        (Tier.L2, "fs"),
    ]


# -- Delete ------------------------------------------------------------------


def test_delete_drops_placement_and_empty_record():
    directory = KeyDirectory()
    directory.consume(_batch(seq=1, keys=[_key(1)]))
    directory.consume(_batch(seq=2, event_type=CacheEventType.DELETE, keys=[_key(1)]))

    assert directory.lookup([_key(1)]) == [[]]
    stats = directory.stats()
    assert stats.num_keys == 0
    assert stats.num_placements == 0
    assert stats.l1_keys_by_instance["node-a"] == 0


def test_removal_of_one_tier_keeps_the_other():
    directory = KeyDirectory()
    directory.consume(_batch(seq=1, tier=Tier.L1, keys=[_key(1)]))
    directory.consume(_batch(seq=2, tier=Tier.L2, backend="fs", keys=[_key(1)]))
    directory.consume(
        _batch(seq=3, event_type=CacheEventType.DELETE, tier=Tier.L1, keys=[_key(1)])
    )

    [placements] = directory.lookup([_key(1)])
    assert [(p.tier, p.backend) for p in placements] == [(Tier.L2, "fs")]
    # ``l1_keys_by_instance`` counts the fencing index; the L1 delete
    # removed the key from it even though L2 still holds it (the
    # placement itself stays visible via lookup / the keys listing).
    assert directory.stats().l1_keys_by_instance["node-a"] == 0


def test_removal_of_unknown_key_is_noop():
    directory = KeyDirectory()
    directory.consume(_batch(event_type=CacheEventType.DELETE, keys=[_key(7)]))
    assert directory.stats().num_keys == 0


# -- Access ------------------------------------------------------------------


def test_access_does_not_create_records():
    directory = KeyDirectory()
    directory.consume(_batch(event_type=CacheEventType.ACCESS, keys=[_key(1)]))
    assert directory.stats().num_keys == 0


def test_access_batch_allows_empty_backend():
    """ACCESS carries no placement identity, so ``backend`` may be empty;
    applying it refreshes recency without touching placements."""
    directory = KeyDirectory()
    directory.consume(_batch(seq=1, keys=[_key(1)], size_bytes=100))
    directory.consume(
        _batch(seq=2, event_type=CacheEventType.ACCESS, keys=[_key(1)], backend="")
    )
    [placements] = directory.lookup([_key(1)])
    assert len(placements) == 1  # placement identity untouched


def test_placement_bearing_batches_require_backend():
    with pytest.raises(ValueError):
        _batch(event_type=CacheEventType.STORE, keys=[_key(1)], backend="")
    with pytest.raises(ValueError):
        _batch(event_type=CacheEventType.DELETE, keys=[_key(1)], backend="")


# -- Fencing -----------------------------------------------------------------


def test_fence_instance_removes_all_its_placements():
    directory = KeyDirectory()
    directory.consume(_batch(instance_id="node-a", keys=[_key(1), _key(2)]))
    directory.consume(_batch(instance_id="node-b", keys=[_key(1)]))

    directory.fence_instance("node-a")

    [placements] = directory.lookup([_key(1)])
    assert [p.instance_id for p in placements] == ["node-b"]
    assert directory.lookup([_key(2)]) == [[]]
    assert "node-a" not in directory.stats().l1_keys_by_instance


def test_fence_unknown_instance_is_noop():
    directory = KeyDirectory()
    directory.fence_instance("ghost")
    assert directory.stats().num_keys == 0


# -- Token bindings ----------------------------------------------------------


def _chash(hash_byte: int) -> bytes:
    return bytes([hash_byte]) * 4


def _rank_key(hash_byte: int, kv_rank: int) -> ObjectKey:
    return ObjectKey(chunk_hash=_chash(hash_byte), model_name="m", kv_rank=kv_rank)


def _tokens(directory: KeyDirectory, *hash_bytes: int) -> list[list[int]]:
    """The known token ids of each chunk, as plain lists."""
    return [list(t) for t in directory.get_token_ids([_chash(b) for b in hash_bytes])]


def test_binding_links_on_store_and_drops_on_delete():
    directory = KeyDirectory()
    directory.consume(_batch(seq=1, keys=[_key(1)], token_ids=[1, 2]))
    assert _tokens(directory, 1) == [[1, 2]]

    directory.consume(_batch(seq=2, event_type=CacheEventType.DELETE, keys=[_key(1)]))
    assert _tokens(directory, 1) == [[]]


def test_binding_survives_until_last_record_of_the_chunk():
    """Records sharing a chunk hash (ranks/groups) hold one binding."""
    directory = KeyDirectory()
    directory.consume(_batch(seq=1, keys=[_key(1), _rank_key(1, 1)], token_ids=[1, 2]))

    directory.consume(_batch(seq=2, event_type=CacheEventType.DELETE, keys=[_key(1)]))
    assert _tokens(directory, 1) == [[1, 2]]

    directory.consume(
        _batch(seq=3, event_type=CacheEventType.DELETE, keys=[_rank_key(1, 1)])
    )
    assert _tokens(directory, 1) == [[]]


def test_chunks_bind_independently():
    directory = KeyDirectory()
    directory.consume(_batch(seq=1, keys=[_key(1)], token_ids=[1, 2]))
    directory.consume(_batch(seq=2, keys=[_key(2)], token_ids=[3, 4]))

    directory.consume(_batch(seq=3, event_type=CacheEventType.DELETE, keys=[_key(1)]))
    assert _tokens(directory, 1, 2) == [[], [3, 4]]


def test_token_less_entry_links_empty_binding():
    """An entry the emitter could not stamp still links its key; the
    chunk's next stamped entry fills the tokens in."""
    directory = KeyDirectory()
    directory.consume(_batch(seq=1, keys=[_rank_key(1, 1)]))
    assert _tokens(directory, 1) == [[]]

    directory.consume(_batch(seq=2, keys=[_key(1)], token_ids=[1, 2]))
    assert _tokens(directory, 1) == [[1, 2]]

    # The unstamped record still holds its reference.
    directory.consume(_batch(seq=3, event_type=CacheEventType.DELETE, keys=[_key(1)]))
    assert _tokens(directory, 1) == [[1, 2]]


def test_untagged_record_heals_on_restore():
    directory = KeyDirectory()
    directory.consume(_batch(seq=1, keys=[_key(1)]))
    assert _tokens(directory, 1) == [[]]

    directory.consume(_batch(seq=2, keys=[_key(1)], token_ids=[1, 2]))
    assert _tokens(directory, 1) == [[1, 2]]

    directory.consume(_batch(seq=3, event_type=CacheEventType.DELETE, keys=[_key(1)]))
    assert _tokens(directory, 1) == [[]]


# -- Token offsets and storage representation --------------------------------


def test_binding_records_the_chunks_token_offset():
    """Chunk hashes are prefix-chained, so the offset cannot be derived —
    it rides the entry and reaches the match as ``old_st``."""
    directory = KeyDirectory()
    directory.enable_blend_lookup(chunk_size=2, probe_stride=1)
    directory.consume(_batch(seq=1, keys=[_key(1)], token_ids=[7, 8], token_offset=512))

    assert directory.get_token_ids([_chash(1)]) == [(7, 8)]
    (match,) = directory.blend_match(np.asarray([7, 8], dtype=np.uint64))
    assert match.old_st == 512


def test_unknown_chunk_yields_empty_tokens():
    directory = KeyDirectory()

    assert directory.get_token_ids([_chash(9)]) == [()]


def test_bindings_are_returned_in_request_order():
    directory = KeyDirectory()
    directory.consume(_batch(seq=1, keys=[_key(1)], token_ids=[1], token_offset=0))
    directory.consume(_batch(seq=2, keys=[_key(2)], token_ids=[2], token_offset=256))

    assert directory.get_token_ids([_chash(2), _chash(9), _chash(1)]) == [
        (2,),
        (),
        (1,),
    ]


def test_restore_replaces_tokens_and_offset():
    directory = KeyDirectory()
    directory.enable_blend_lookup(chunk_size=2, probe_stride=1)
    directory.consume(_batch(seq=1, keys=[_key(1)], token_ids=[1, 2], token_offset=0))
    directory.consume(_batch(seq=2, keys=[_key(1)], token_ids=[3, 4], token_offset=256))

    assert directory.get_token_ids([_chash(1)]) == [(3, 4)]
    (match,) = directory.blend_match(np.asarray([3, 4], dtype=np.uint64))
    assert match.old_st == 256
    # The superseded content is no longer discoverable.
    assert directory.blend_match(np.asarray([1, 2], dtype=np.uint64)) == []


def test_token_ids_outside_uint32_leave_the_binding_unfilled():
    """A malformed entry must not fail the batch: the placement still
    applies and the binding stays a lookup miss."""
    directory = KeyDirectory()

    directory.consume(_batch(seq=1, keys=[_key(1)], token_ids=[1, 2**32]))

    assert directory.lookup([_key(1)])[0]  # placement applied
    assert _tokens(directory, 1) == [[]]

    # Repaired by the chunk's next well-formed entry.
    directory.consume(_batch(seq=2, keys=[_key(1)], token_ids=[1, 2]))
    assert _tokens(directory, 1) == [[1, 2]]


def test_negative_token_offset_is_rejected():
    with pytest.raises(ValueError, match="token_offset must be >= 0"):
        CacheEventEntry(
            key=_key(1).to_encoded_object_key(), token_ids=[1], token_offset=-2
        )


def test_unreported_offset_defaults_to_unknown_not_zero():
    """An emitter predating token offsets must not be read as claiming
    position 0 — that would re-RoPE reused KV from the wrong source."""
    entry = CacheEventEntry(key=_key(1).to_encoded_object_key(), token_ids=[1, 2])

    assert entry.token_offset == UNKNOWN_TOKEN_OFFSET


def test_restore_with_new_tokens_rebinds():
    directory = KeyDirectory()
    directory.consume(_batch(seq=1, keys=[_key(1)], token_ids=[1, 2]))
    directory.consume(_batch(seq=2, keys=[_key(1)], token_ids=[3, 4]))

    assert _tokens(directory, 1) == [[3, 4]]


def test_fencing_releases_bindings():
    directory = KeyDirectory()
    directory.consume(_batch(seq=1, keys=[_key(1)], token_ids=[1, 2]))

    directory.fence_instance("node-a")
    assert _tokens(directory, 1) == [[]]


# -- Listing -------------------------------------------------------------------


def test_list_keys_returns_pages_of_matching_keys():
    directory = KeyDirectory()
    directory.consume(_batch(seq=1, keys=[_key(1)], token_ids=[1, 2]))
    directory.consume(_batch(seq=2, keys=[_key(2)]))

    total, page = directory.list_keys()
    assert total == 2
    assert set(page) == {_key(1), _key(2)}
    assert len(page[_key(1)]) == 1


def test_list_keys_filters_keep_matching_placements_only():
    directory = KeyDirectory()
    directory.consume(_batch(seq=1, tier=Tier.L1, backend="dram", keys=[_key(1)]))
    directory.consume(_batch(seq=2, tier=Tier.L2, backend="fs", keys=[_key(1)]))
    directory.consume(
        _batch(instance_id="node-b", tier=Tier.L1, backend="dram", keys=[_key(2)])
    )

    total, page = directory.list_keys(tier=Tier.L2)
    assert total == 1
    assert [(p.tier, p.backend) for p in page[_key(1)]] == [(Tier.L2, "fs")]

    _, node_b = directory.list_keys(instance_id="node-b")
    assert set(node_b) == {_key(2)}

    assert directory.list_keys(backend="fs")[0] == 1
    assert directory.list_keys(backend="valkey")[0] == 0


def test_list_keys_paginates_with_stable_total():
    directory = KeyDirectory()
    directory.consume(_batch(seq=1, keys=[_key(1), _key(2), _key(3)]))

    first_total, first = directory.list_keys(offset=0, limit=2)
    rest_total, rest = directory.list_keys(offset=2, limit=2)
    assert first_total == rest_total == 3
    assert len(first) == 2
    assert len(rest) == 1
    assert set(first) | set(rest) == {_key(1), _key(2), _key(3)}


def test_list_keys_rejects_negative_paging():
    directory = KeyDirectory()
    with pytest.raises(ValueError, match="offset"):
        directory.list_keys(offset=-1)
    with pytest.raises(ValueError, match="limit"):
        directory.list_keys(limit=-1)


# -- Intrinsic invariants ------------------------------------------------------


def test_tier_all_is_unconstructible():
    with pytest.raises(ValueError, match="concrete tier"):
        _batch(tier=Tier.ALL)


def test_seq_below_one_is_unconstructible():
    with pytest.raises(ValueError, match="seq"):
        _batch(seq=0)


def test_negative_size_is_unconstructible():
    with pytest.raises(ValueError, match="size_bytes"):
        CacheEventEntry(key=_key(1).to_encoded_object_key(), size_bytes=-1)


# -- Stats -------------------------------------------------------------------


def test_stats_counts_keys_and_placements():
    directory = KeyDirectory()
    directory.consume(_batch(instance_id="node-a", seq=1, keys=[_key(1), _key(2)]))
    directory.consume(
        _batch(instance_id="node-b", seq=1, tier=Tier.L2, backend="fs", keys=[_key(1)])
    )

    stats = directory.stats()
    assert stats.num_keys == 2
    assert stats.num_placements == 3
    assert stats.l1_keys_by_instance["node-a"] == 2
    # node-b reported only an L2 placement: absent from the L1 fencing
    # index (its placement stays visible via lookup / the keys listing).
    assert stats.l1_keys_by_instance["node-b"] == 0


# -- Shared locations ----------------------------------------------------------


def test_shared_pool_dedups_across_reporters():
    """Stores of one key into the same shared pool by different
    instances upsert a single placement."""
    directory = KeyDirectory()
    directory.consume(
        _batch(
            instance_id="node-a",
            seq=1,
            keys=[_key(1)],
            tier=Tier.L2,
            backend="s3",
            shared=True,
            size_bytes=100,
        )
    )
    directory.consume(
        _batch(
            instance_id="node-b",
            seq=1,
            keys=[_key(1)],
            tier=Tier.L2,
            backend="s3",
            shared=True,
            size_bytes=100,
        )
    )
    [placements] = directory.lookup([_key(1)])
    assert len(placements) == 1
    assert placements[0].shared is True
    assert placements[0].instance_id == "node-b"  # last reporter


def test_shared_pool_delete_from_any_reporter():
    directory = KeyDirectory()
    directory.consume(
        _batch(
            instance_id="node-a",
            seq=1,
            keys=[_key(1)],
            tier=Tier.L2,
            backend="s3",
            shared=True,
        )
    )
    directory.consume(
        _batch(
            instance_id="node-b",
            seq=1,
            keys=[_key(1)],
            event_type=CacheEventType.DELETE,
            tier=Tier.L2,
            backend="s3",
            shared=True,
        )
    )
    [placements] = directory.lookup([_key(1)])
    assert placements == []


def test_fencing_drops_l1_and_keeps_l2():
    """Fencing is scoped to L1: memory dies with the reporting stream (a
    shared-L1 pool controller clears its pool by restarting), while L2
    bytes persist across restarts and their placements survive — private
    and shared alike."""
    directory = KeyDirectory()
    # Shared L1 pool (e.g. CXL) reported by its controller stream.
    directory.consume(
        _batch(
            instance_id="pool-controller",
            incarnation=1,
            seq=1,
            keys=[_key(1)],
            tier=Tier.L1,
            backend="cxl",
            shared=True,
        )
    )
    # Private L2 and shared L2 from an ordinary instance.
    directory.consume(
        _batch(
            instance_id="node-b",
            incarnation=1,
            seq=1,
            keys=[_key(2)],
            tier=Tier.L2,
            backend="fs",
        )
    )
    directory.consume(
        _batch(
            instance_id="node-b",
            incarnation=1,
            seq=2,
            keys=[_key(2)],
            tier=Tier.L2,
            backend="s3",
            shared=True,
        )
    )
    # Controller restart: its L1 pool placements are fenced.
    directory.fence_instance("pool-controller")
    # Instance restart: its L2 placements (private and shared) survive.
    directory.fence_instance("node-b")
    p1, p2 = directory.lookup([_key(1), _key(2)])
    assert p1 == []  # L1 pool contents fenced with their reporting stream
    assert sorted((p.tier, p.backend) for p in p2) == [
        (Tier.L2, "fs"),
        (Tier.L2, "s3"),
    ]


def test_fencing_keeps_shared_placement_re_reported_elsewhere():
    """A shared placement upserted by a later reporter survives the
    original reporter's restart: the fence matches each placement's
    recorded reporter, not the stale reverse-index entry."""
    directory = KeyDirectory()
    directory.consume(
        _batch(
            instance_id="node-a",
            incarnation=1,
            seq=1,
            keys=[_key(1)],
            tier=Tier.L2,
            backend="s3",
            shared=True,
        )
    )
    directory.consume(
        _batch(
            instance_id="node-b",
            seq=1,
            keys=[_key(1)],
            tier=Tier.L2,
            backend="s3",
            shared=True,
        )
    )
    # node-a restarts; the placement's reporter of record is now node-b.
    directory.fence_instance("node-a")
    [p1] = directory.lookup([_key(1)])
    assert [p.instance_id for p in p1] == ["node-b"]


def test_fence_instance_drops_l1_and_keeps_l2():
    directory = KeyDirectory()
    directory.consume(_batch(instance_id="node-a", seq=1, keys=[_key(1)]))
    directory.consume(
        _batch(
            instance_id="node-a",
            seq=2,
            keys=[_key(1)],
            tier=Tier.L2,
            backend="s3",
            shared=True,
        )
    )
    directory.fence_instance("node-a")  # L2 bytes persist in the pool
    [placements] = directory.lookup([_key(1)])
    assert [(p.tier, p.shared) for p in placements] == [(Tier.L2, True)]


def test_controller_stream_reports_shared_events():
    """A shared medium's controller is just another emitter: it reports
    under its own stream id, and its deletes match instance-reported
    shared placements."""
    directory = KeyDirectory()
    directory.consume(
        _batch(
            instance_id="node-a",
            seq=1,
            keys=[_key(1)],
            tier=Tier.L2,
            backend="s3",
            shared=True,
        )
    )
    # Controller-driven eviction removes the instance-reported placement.
    directory.consume(
        _batch(
            instance_id="pool-controller",
            seq=1,
            keys=[_key(1)],
            event_type=CacheEventType.DELETE,
            tier=Tier.L2,
            backend="s3",
            shared=True,
        )
    )
    assert directory.lookup([_key(1)]) == [[]]


def test_empty_instance_id_is_unconstructible():
    with pytest.raises(ValueError, match="instance_id"):
        _batch(instance_id="")


def test_same_backend_private_and_shared_are_distinct_placements():
    """The same (tier, backend) can hold both a private and a shared
    placement of one key — sharedness is part of the identity."""
    directory = KeyDirectory()
    directory.consume(
        _batch(instance_id="node-a", seq=1, keys=[_key(1)], tier=Tier.L2, backend="fs")
    )
    directory.consume(
        _batch(
            instance_id="node-a",
            seq=2,
            keys=[_key(1)],
            tier=Tier.L2,
            backend="fs",
            shared=True,
        )
    )
    [placements] = directory.lookup([_key(1)])
    assert sorted(p.shared for p in placements) == [False, True]
