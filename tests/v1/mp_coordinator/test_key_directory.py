# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator key directory (I1): event application
semantics (seq dedup, gap detection, incarnation fencing), lookup, and
instance cleanup."""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.key_directory import ApplyResult, KeyDirectory


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
) -> CacheEventBatch:
    entries = [
        CacheEventEntry(
            key=k.to_encoded_object_key(),
            size_bytes=size_bytes,
            token_ids=token_ids or [],
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
    assert directory.apply_batch(_batch(keys=[_key(1)])) == ApplyResult.APPLIED

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
    directory.apply_batch(_batch(keys=[_key(2)]))
    results = directory.lookup([_key(1), _key(2)])
    assert results[0] == []
    assert len(results[1]) == 1


def test_restore_updates_size_without_duplicating_placement():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, keys=[_key(1)], size_bytes=100))
    directory.apply_batch(_batch(seq=2, keys=[_key(1)], size_bytes=200))

    [placements] = directory.lookup([_key(1)])
    assert len(placements) == 1
    assert placements[0].size_bytes == 200


def test_same_key_on_two_instances():
    directory = KeyDirectory()
    directory.apply_batch(_batch(instance_id="node-b", keys=[_key(1)]))
    directory.apply_batch(_batch(instance_id="node-a", keys=[_key(1)]))

    [placements] = directory.lookup([_key(1)])
    assert [p.instance_id for p in placements] == ["node-a", "node-b"]


def test_same_key_on_two_tiers_of_one_instance():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, tier=Tier.L1, backend="dram", keys=[_key(1)]))
    directory.apply_batch(_batch(seq=2, tier=Tier.L2, backend="fs", keys=[_key(1)]))

    [placements] = directory.lookup([_key(1)])
    assert [(p.tier, p.backend) for p in placements] == [
        (Tier.L1, "dram"),
        (Tier.L2, "fs"),
    ]


# -- Delete ------------------------------------------------------------------


def test_delete_drops_placement_and_empty_record():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, keys=[_key(1)]))
    outcome = directory.apply_batch(
        _batch(seq=2, event_type=CacheEventType.DELETE, keys=[_key(1)])
    )

    assert outcome == ApplyResult.APPLIED
    assert directory.lookup([_key(1)]) == [[]]
    stats = directory.stats()
    assert stats.num_keys == 0
    assert stats.num_placements == 0
    assert stats.instances["node-a"].num_l1_keys == 0


def test_removal_of_one_tier_keeps_the_other():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, tier=Tier.L1, keys=[_key(1)]))
    directory.apply_batch(_batch(seq=2, tier=Tier.L2, backend="fs", keys=[_key(1)]))
    directory.apply_batch(
        _batch(seq=3, event_type=CacheEventType.DELETE, tier=Tier.L1, keys=[_key(1)])
    )

    [placements] = directory.lookup([_key(1)])
    assert [(p.tier, p.backend) for p in placements] == [(Tier.L2, "fs")]
    # ``num_l1_keys`` counts the fencing index; the L1 delete removed
    # the key from it even though L2 still holds it (the placement
    # itself stays visible via lookup / the keys listing).
    assert directory.stats().instances["node-a"].num_l1_keys == 0


def test_removal_of_unknown_key_is_noop():
    directory = KeyDirectory()
    outcome = directory.apply_batch(
        _batch(event_type=CacheEventType.DELETE, keys=[_key(7)])
    )
    assert outcome == ApplyResult.APPLIED
    assert directory.stats().num_keys == 0


# -- Access ------------------------------------------------------------------


def test_access_does_not_create_records():
    directory = KeyDirectory()
    outcome = directory.apply_batch(
        _batch(event_type=CacheEventType.ACCESS, keys=[_key(1)])
    )
    assert outcome == ApplyResult.APPLIED
    assert directory.stats().num_keys == 0


def test_access_batch_allows_empty_backend():
    """ACCESS carries no placement identity, so ``backend`` may be empty;
    applying it refreshes recency without touching placements."""
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, keys=[_key(1)], size_bytes=100))
    outcome = directory.apply_batch(
        _batch(seq=2, event_type=CacheEventType.ACCESS, keys=[_key(1)], backend="")
    )
    assert outcome == ApplyResult.APPLIED
    [placements] = directory.lookup([_key(1)])
    assert len(placements) == 1  # placement identity untouched


def test_placement_bearing_batches_require_backend():
    with pytest.raises(ValueError):
        _batch(event_type=CacheEventType.STORE, keys=[_key(1)], backend="")
    with pytest.raises(ValueError):
        _batch(event_type=CacheEventType.DELETE, keys=[_key(1)], backend="")


# -- Seq handling ------------------------------------------------------------


def test_duplicate_seq_is_dropped():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, keys=[_key(1)], size_bytes=100))
    outcome = directory.apply_batch(_batch(seq=1, keys=[_key(1)], size_bytes=999))

    assert outcome == ApplyResult.DUPLICATE
    [placements] = directory.lookup([_key(1)])
    assert placements[0].size_bytes == 100


def test_replayed_older_seq_is_dropped():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, keys=[_key(1)]))
    directory.apply_batch(
        _batch(seq=2, event_type=CacheEventType.DELETE, keys=[_key(1)])
    )
    outcome = directory.apply_batch(_batch(seq=1, keys=[_key(1)]))

    assert outcome == ApplyResult.DUPLICATE
    assert directory.lookup([_key(1)]) == [[]]


def test_seq_gap_sets_resync_flag_but_applies():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, keys=[_key(1)]))
    outcome = directory.apply_batch(_batch(seq=5, keys=[_key(2)]))

    assert outcome == ApplyResult.APPLIED
    info = directory.stats().instances["node-a"]
    assert info.gap_detected is True
    assert info.last_seq == 5
    assert len(directory.lookup([_key(2)])[0]) == 1


def test_contiguous_seqs_do_not_flag_gap():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, keys=[_key(1)]))
    directory.apply_batch(_batch(seq=2, keys=[_key(2)]))
    assert directory.stats().instances["node-a"].gap_detected is False


# -- Incarnation fencing -----------------------------------------------------


def test_new_incarnation_fences_old_placements():
    directory = KeyDirectory()
    directory.apply_batch(_batch(incarnation=1, seq=1, keys=[_key(1), _key(2)]))
    outcome = directory.apply_batch(_batch(incarnation=2, seq=1, keys=[_key(3)]))

    assert outcome == ApplyResult.APPLIED
    assert directory.lookup([_key(1)]) == [[]]
    assert directory.lookup([_key(2)]) == [[]]
    [placements] = directory.lookup([_key(3)])
    assert placements[0].incarnation == 2
    info = directory.stats().instances["node-a"]
    assert info.incarnation == 2
    assert info.last_seq == 1


def test_fence_spares_other_instances_placements():
    directory = KeyDirectory()
    directory.apply_batch(_batch(instance_id="node-a", keys=[_key(1)]))
    directory.apply_batch(_batch(instance_id="node-b", keys=[_key(1)]))
    directory.apply_batch(_batch(instance_id="node-a", incarnation=2, keys=[_key(9)]))

    [placements] = directory.lookup([_key(1)])
    assert [p.instance_id for p in placements] == ["node-b"]


def test_stale_incarnation_batch_is_dropped():
    directory = KeyDirectory()
    directory.apply_batch(_batch(incarnation=2, seq=1, keys=[_key(1)]))
    outcome = directory.apply_batch(_batch(incarnation=1, seq=99, keys=[_key(2)]))

    assert outcome == ApplyResult.STALE_INCARNATION
    assert directory.lookup([_key(2)]) == [[]]


# -- drop_instance -----------------------------------------------------------


def test_drop_instance_removes_all_its_placements():
    directory = KeyDirectory()
    directory.apply_batch(_batch(instance_id="node-a", keys=[_key(1), _key(2)]))
    directory.apply_batch(_batch(instance_id="node-b", keys=[_key(1)]))

    removed = directory.drop_instance("node-a")

    assert removed == 2
    [placements] = directory.lookup([_key(1)])
    assert [p.instance_id for p in placements] == ["node-b"]
    assert directory.lookup([_key(2)]) == [[]]
    assert "node-a" not in directory.stats().instances


def test_drop_unknown_instance_returns_zero():
    directory = KeyDirectory()
    assert directory.drop_instance("ghost") == 0


# -- Token bindings ----------------------------------------------------------


def _chash(hash_byte: int) -> bytes:
    return bytes([hash_byte]) * 4


def _rank_key(hash_byte: int, kv_rank: int) -> ObjectKey:
    return ObjectKey(chunk_hash=_chash(hash_byte), model_name="m", kv_rank=kv_rank)


def test_binding_links_on_store_and_drops_on_delete():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, keys=[_key(1)], token_ids=[1, 2]))
    assert directory.get_token_ids([_chash(1)]) == [(1, 2)]

    directory.apply_batch(
        _batch(seq=2, event_type=CacheEventType.DELETE, keys=[_key(1)])
    )
    assert directory.get_token_ids([_chash(1)]) == [()]


def test_binding_survives_until_last_record_of_the_chunk():
    """Records sharing a chunk hash (ranks/groups) hold one binding."""
    directory = KeyDirectory()
    directory.apply_batch(
        _batch(seq=1, keys=[_key(1), _rank_key(1, 1)], token_ids=[1, 2])
    )

    directory.apply_batch(
        _batch(seq=2, event_type=CacheEventType.DELETE, keys=[_key(1)])
    )
    assert directory.get_token_ids([_chash(1)]) == [(1, 2)]

    directory.apply_batch(
        _batch(seq=3, event_type=CacheEventType.DELETE, keys=[_rank_key(1, 1)])
    )
    assert directory.get_token_ids([_chash(1)]) == [()]


def test_chunks_bind_independently():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, keys=[_key(1)], token_ids=[1, 2]))
    directory.apply_batch(_batch(seq=2, keys=[_key(2)], token_ids=[3, 4]))

    directory.apply_batch(
        _batch(seq=3, event_type=CacheEventType.DELETE, keys=[_key(1)])
    )
    assert directory.get_token_ids([_chash(1), _chash(2)]) == [(), (3, 4)]


def test_token_less_entry_links_empty_binding():
    """An entry the emitter could not stamp still links its key; the
    chunk's next stamped entry fills the tokens in."""
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, keys=[_rank_key(1, 1)]))
    assert directory.get_token_ids([_chash(1)]) == [()]

    directory.apply_batch(_batch(seq=2, keys=[_key(1)], token_ids=[1, 2]))
    assert directory.get_token_ids([_chash(1)]) == [(1, 2)]

    # The unstamped record still holds its reference.
    directory.apply_batch(
        _batch(seq=3, event_type=CacheEventType.DELETE, keys=[_key(1)])
    )
    assert directory.get_token_ids([_chash(1)]) == [(1, 2)]


def test_untagged_record_heals_on_restore():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, keys=[_key(1)]))
    assert directory.get_token_ids([_chash(1)]) == [()]

    directory.apply_batch(_batch(seq=2, keys=[_key(1)], token_ids=[1, 2]))
    assert directory.get_token_ids([_chash(1)]) == [(1, 2)]

    directory.apply_batch(
        _batch(seq=3, event_type=CacheEventType.DELETE, keys=[_key(1)])
    )
    assert directory.get_token_ids([_chash(1)]) == [()]


def test_restore_with_new_tokens_rebinds():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, keys=[_key(1)], token_ids=[1, 2]))
    directory.apply_batch(_batch(seq=2, keys=[_key(1)], token_ids=[3, 4]))

    assert directory.get_token_ids([_chash(1)]) == [(3, 4)]


def test_fencing_releases_bindings():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, keys=[_key(1)], token_ids=[1, 2]))

    directory.apply_batch(_batch(seq=1, incarnation=2, keys=[_key(9)]))
    assert directory.get_token_ids([_chash(1)]) == [()]


def test_drop_instance_releases_bindings():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, keys=[_key(1)], token_ids=[1, 2]))

    directory.drop_instance("node-a")
    assert directory.get_token_ids([_chash(1)]) == [()]


# -- Listing -------------------------------------------------------------------


def test_list_keys_returns_pages_of_matching_keys():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, keys=[_key(1)], token_ids=[1, 2]))
    directory.apply_batch(_batch(seq=2, keys=[_key(2)]))

    total, page = directory.list_keys()
    assert total == 2
    assert set(page) == {_key(1), _key(2)}
    assert len(page[_key(1)]) == 1


def test_list_keys_filters_keep_matching_placements_only():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, tier=Tier.L1, backend="dram", keys=[_key(1)]))
    directory.apply_batch(_batch(seq=2, tier=Tier.L2, backend="fs", keys=[_key(1)]))
    directory.apply_batch(
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
    directory.apply_batch(_batch(seq=1, keys=[_key(1), _key(2), _key(3)]))

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
    directory.apply_batch(_batch(instance_id="node-a", seq=1, keys=[_key(1), _key(2)]))
    directory.apply_batch(
        _batch(instance_id="node-b", seq=1, tier=Tier.L2, backend="fs", keys=[_key(1)])
    )

    stats = directory.stats()
    assert stats.num_keys == 2
    assert stats.num_placements == 3
    assert stats.instances["node-a"].num_l1_keys == 2
    # node-b reported only an L2 placement: absent from the L1 fencing
    # index (its placement stays visible via lookup / the keys listing).
    assert stats.instances["node-b"].num_l1_keys == 0


# -- Shared locations ----------------------------------------------------------


def test_shared_pool_dedups_across_reporters():
    """Stores of one key into the same shared pool by different
    instances upsert a single placement."""
    directory = KeyDirectory()
    directory.apply_batch(
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
    directory.apply_batch(
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
    directory.apply_batch(
        _batch(
            instance_id="node-a",
            seq=1,
            keys=[_key(1)],
            tier=Tier.L2,
            backend="s3",
            shared=True,
        )
    )
    directory.apply_batch(
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


def test_incarnation_fencing_drops_l1_and_keeps_l2():
    """Fencing is scoped to L1: memory dies with the reporting stream (a
    shared-L1 pool controller clears its pool by restarting), while L2
    bytes persist across restarts and their placements survive — private
    and shared alike."""
    directory = KeyDirectory()
    # Shared L1 pool (e.g. CXL) reported by its controller stream.
    directory.apply_batch(
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
    directory.apply_batch(
        _batch(
            instance_id="node-b",
            incarnation=1,
            seq=1,
            keys=[_key(2)],
            tier=Tier.L2,
            backend="fs",
        )
    )
    directory.apply_batch(
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
    directory.apply_batch(
        _batch(
            instance_id="pool-controller",
            incarnation=2,
            seq=1,
            keys=[_key(3)],
            tier=Tier.L1,
            backend="cxl",
            shared=True,
        )
    )
    # Instance restart: its L2 placements (private and shared) survive.
    directory.apply_batch(
        _batch(instance_id="node-b", incarnation=2, seq=1, keys=[_key(3)])
    )
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
    directory.apply_batch(
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
    directory.apply_batch(
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
    directory.apply_batch(
        _batch(instance_id="node-a", incarnation=2, seq=1, keys=[_key(2)])
    )
    [p1] = directory.lookup([_key(1)])
    assert [p.instance_id for p in p1] == ["node-b"]


def test_drop_instance_drops_l1_and_keeps_l2():
    directory = KeyDirectory()
    directory.apply_batch(_batch(instance_id="node-a", seq=1, keys=[_key(1)]))
    directory.apply_batch(
        _batch(
            instance_id="node-a",
            seq=2,
            keys=[_key(1)],
            tier=Tier.L2,
            backend="s3",
            shared=True,
        )
    )
    removed = directory.drop_instance("node-a")
    assert removed == 1  # the L1 placement; L2 bytes persist in the pool
    [placements] = directory.lookup([_key(1)])
    assert [(p.tier, p.shared) for p in placements] == [(Tier.L2, True)]


def test_controller_stream_reports_shared_events():
    """A shared medium's controller is just another emitter: it reports
    under its own stream id, and its deletes match instance-reported
    shared placements."""
    directory = KeyDirectory()
    directory.apply_batch(
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
    outcome = directory.apply_batch(
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
    assert outcome == ApplyResult.APPLIED
    assert directory.lookup([_key(1)]) == [[]]
    # The controller's stream dedups by seq like any other.
    duplicate = directory.apply_batch(
        _batch(
            instance_id="pool-controller",
            seq=1,
            keys=[_key(2)],
            tier=Tier.L2,
            backend="s3",
            shared=True,
        )
    )
    assert duplicate == ApplyResult.DUPLICATE


def test_empty_instance_id_is_unconstructible():
    with pytest.raises(ValueError, match="instance_id"):
        _batch(instance_id="")


def test_same_backend_private_and_shared_are_distinct_placements():
    """The same (tier, backend) can hold both a private and a shared
    placement of one key — sharedness is part of the identity."""
    directory = KeyDirectory()
    directory.apply_batch(
        _batch(instance_id="node-a", seq=1, keys=[_key(1)], tier=Tier.L2, backend="fs")
    )
    directory.apply_batch(
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


# -- Reconcile (gate-bypassing backfill) ---------------------------------------


def test_reconcile_seeds_placements_that_live_deletes_remove():
    """A backfilled placement carries the live stream's identity
    (backend + shared), so a later DELETE event removes it."""
    directory = KeyDirectory()
    directory.reconcile(
        _batch(incarnation=0, seq=1, tier=Tier.L2, backend="fs", keys=[_key(1)])
    )
    [placements] = directory.lookup([_key(1)])
    assert [(p.tier, p.backend) for p in placements] == [(Tier.L2, "fs")]
    # Live stream (higher incarnation) deletes the same placement.
    outcome = directory.apply_batch(
        _batch(
            incarnation=5,
            seq=1,
            event_type=CacheEventType.DELETE,
            tier=Tier.L2,
            backend="fs",
            keys=[_key(1)],
        )
    )
    assert outcome == ApplyResult.APPLIED
    assert directory.lookup([_key(1)]) == [[]]


def test_reconcile_does_not_disturb_live_stream_cursor():
    """Backfill bypasses incarnation/seq bookkeeping: it is neither
    rejected as stale nor does it advance the live cursor."""
    directory = KeyDirectory()
    directory.apply_batch(
        _batch(incarnation=7, seq=3, tier=Tier.L2, backend="fs", keys=[_key(1)])
    )
    directory.reconcile(
        _batch(incarnation=0, seq=1, tier=Tier.L2, backend="fs", keys=[_key(2)])
    )
    info = directory.stats().instances["node-a"]
    assert info.incarnation == 7
    assert info.last_seq == 3
    assert len(directory.lookup([_key(2)])[0]) == 1
    # The live stream continues undisturbed.
    assert (
        directory.apply_batch(_batch(incarnation=7, seq=4, keys=[_key(3)]))
        == ApplyResult.APPLIED
    )


def test_reconcile_is_idempotent():
    directory = KeyDirectory()
    batch = _batch(incarnation=0, seq=1, tier=Tier.L2, backend="fs", keys=[_key(1)])
    directory.reconcile(batch)
    directory.reconcile(batch)
    [placements] = directory.lookup([_key(1)])
    assert len(placements) == 1
    assert placements[0].size_bytes == 1024
