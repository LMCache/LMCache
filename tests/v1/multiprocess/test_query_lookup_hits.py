# SPDX-License-Identifier: Apache-2.0
"""Tests for the key rows the LOOKUP handler submits to the storage manager."""

# Standard
from unittest.mock import MagicMock

# First Party
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    GroupedObjectKeys,
    ObjectKey,
    RankGroupLayouts,
    ipc_key_to_object_keys,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.token_codec import pack_token_ids

# ============================================================================
# Grouped key layout
# ============================================================================


def _lookup_key(world_size: int) -> IPCCacheServerKey:
    """A lookup-side IPC key (worker_id None -> expand over all workers)."""
    return IPCCacheServerKey(
        model_name="m",
        world_size=world_size,
        num_kv_readers=1,
        worker_id=None,
        token_bytes=pack_token_ids((0,)),
        start=0,
        end=0,
        request_id="r",
    )


def _captured_lookup_key_groups(
    world_size: int,
    num_groups: int,
    chunk_hashes: list[bytes],
    rank_groups: dict[int, RankGroupLayouts] | None = None,
) -> list[GroupedObjectKeys]:
    """Drive the public ``lookup()`` and return the key rows it submits.

    The engine context is mocked so ``lookup()`` runs end-to-end; the rows
    are recovered from the ``submit_prefetch_task`` call rather than by
    reaching into a private helper.
    """
    ctx = MagicMock()
    # No previous lookup to settle: ``lookup`` claims any unreleased
    # result before ``begin_lookup`` resets it.
    _sess = ctx.session_manager.get_or_create.return_value
    _sess.take_prefetch_release.return_value = None
    ctx.chunk_size = 16
    ctx.event_bus.has_subscribers.return_value = False
    ctx.layout_desc_registry.find.return_value = MagicMock()  # non-None layout
    # lookup() requires a registered per-group layout map (it error-returns
    # without one); one entry per object group, keyed by object_group_id.
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {
        gid: MagicMock() for gid in range(num_groups)
    }
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1] * num_groups
    )
    ctx.layout_desc_registry.find_rank_group_layouts.return_value = rank_groups or {}
    ctx.token_hasher.compute_packed_chunk_hashes.return_value = chunk_hashes

    module = LookupModule(ctx)
    module.lookup(_lookup_key(world_size=world_size), tp_size=1)

    ctx.storage_manager.submit_prefetch_task.assert_called_once()
    spec = ctx.storage_manager.submit_prefetch_task.call_args.args[0]
    assert spec.fetching_policy == "prefix"
    return spec.key_groups


def test_lookup_submits_one_row_per_group_and_rank_group_major():
    """lookup() submits one chunk-ordered key row per (object group, kv_rank),
    group-major / rank-minor, each row carrying its own group id."""
    rows = _captured_lookup_key_groups(
        world_size=2, num_groups=2, chunk_hashes=[b"c0", b"c1"]
    )

    # 2 groups * 2 ranks rows, each of 2 chunks.
    assert len(rows) == 4
    assert [row.object_group_id for row in rows] == [0, 0, 1, 1]
    for row in rows:
        assert [k.chunk_hash for k in row.keys] == [b"c0", b"c1"]
        assert {k.object_group_id for k in row.keys} == {row.object_group_id}
        assert len({k.kv_rank for k in row.keys}) == 1
    # The two rank rows of one group address distinct kv_ranks.
    assert rows[0].keys[0].kv_rank != rows[1].keys[0].kv_rank
    # Every group's rows use the same rank order.
    assert [r.keys[0].kv_rank for r in rows[:2]] == [
        r.keys[0].kv_rank for r in rows[2:]
    ]


def test_lookup_asks_each_rank_for_only_the_groups_it_registered():
    """Pipeline stages can store different object groups: a stage of dense
    layers has no indexer cache. Each rank is asked for exactly its own groups
    with its own windows, so a group one rank never stores cannot zero the
    hit, and a group only one rank stores is still looked up."""
    rank_groups = {
        0: RankGroupLayouts(
            attn_desc=AttnWindowDesc(num_chunks_in_sw=[-1]),
            group_layout_descs={0: MagicMock()},
        ),
        1: RankGroupLayouts(
            attn_desc=AttnWindowDesc(num_chunks_in_sw=[-1, 2]),
            group_layout_descs={0: MagicMock(), 1: MagicMock()},
        ),
    }

    # The registration-wide desc is the last registrant's: here rank 0's
    # single group, which alone would never look up rank 1's second group.
    rows = _captured_lookup_key_groups(
        world_size=2, num_groups=1, chunk_hashes=[b"c0"], rank_groups=rank_groups
    )

    def kv_rank(worker_id: int) -> int:
        return ObjectKey.ComputeKVRank(
            world_size=2,
            global_rank=worker_id,
            local_world_size=2,
            local_rank=worker_id,
        )

    assert [
        (row.object_group_id, row.keys[0].kv_rank, row.sliding_window_size)
        for row in rows
    ] == [(0, kv_rank(0), -1), (0, kv_rank(1), -1), (1, kv_rank(1), 2)]


def test_lookup_single_group_matches_single_group_expansion():
    """With one object group the submitted keys are exactly the single-group
    expansion, one row per kv_rank (the object-group-separation-disabled /
    non-hybrid case)."""
    chunk_hashes = [b"c0", b"c1"]
    rows = _captured_lookup_key_groups(
        world_size=2, num_groups=1, chunk_hashes=chunk_hashes
    )

    expected = ipc_key_to_object_keys(_lookup_key(world_size=2), chunk_hashes, [0])[0]
    assert len(rows) == 2
    submitted = [k for row in rows for k in row.keys]
    assert sorted(submitted, key=repr) == sorted(expected, key=repr)


def test_lookup_hashing_stops_at_key_end() -> None:
    """LOOKUP must not hash chunks beyond the IPC key's requested range."""
    ctx = MagicMock()
    # No previous lookup to settle: ``lookup`` claims any unreleased
    # result before ``begin_lookup`` resets it.
    _sess = ctx.session_manager.get_or_create.return_value
    _sess.take_prefetch_release.return_value = None
    ctx.chunk_size = 16
    ctx.event_bus.has_subscribers.return_value = False
    ctx.layout_desc_registry.find.return_value = MagicMock()
    ctx.layout_desc_registry.find_rank_group_layouts.return_value = {}
    ctx.token_hasher.compute_packed_chunk_hashes.return_value = []
    key = IPCCacheServerKey(
        model_name="m",
        world_size=1,
        num_kv_readers=1,
        worker_id=None,
        token_bytes=pack_token_ids(range(32)),
        start=0,
        end=16,
        request_id="r",
    )

    LookupModule(ctx).lookup(key, tp_size=1)

    ctx.token_hasher.compute_packed_chunk_hashes.assert_called_once_with(
        pack_token_ids(range(32)), end=16
    )
