# SPDX-License-Identifier: Apache-2.0
"""Async loading helpers of the sharded PCP store (no GPU, no Redis)."""

# Standard
import random

# First Party
from lmcache.v1 import pcp_shard as ps


def _chunks(n_tokens, chunk_size=256):
    out, start = [], 0
    while start < n_tokens:
        end = min(start + chunk_size, n_tokens)
        out.append((start, end, f"key{start}"))
        start = end
    return out


def _reported(rank, chunks, present, world, chunk_size=256):
    """What the storage manager reports for this rank: cum[number of leading owned hits]
    (cum[0] when nothing hits, as patched)."""
    keys, cum = ps.async_lookup_keys(chunks, chunk_size, world, rank)
    by_key = {c[2]: ok for c, ok in zip(chunks, present, strict=False)}
    hits = 0
    for k in keys:
        if not by_key[k]:
            break
        hits += 1
    return cum[hits]


def test_min_over_ranks_is_sharded_prefix():
    rng = random.Random(0)
    for _ in range(3000):
        world = rng.randint(1, 8)
        n_tokens = rng.randint(0, 40 * 256 + 100)
        chunks = _chunks(n_tokens)
        present = [rng.random() < 0.85 for _ in chunks]
        got = min(_reported(r, chunks, present, world) for r in range(world))
        want = ps.global_prefix_tokens([(s, e) for s, e, _ in chunks], present)
        assert got == want, (world, n_tokens, present, got, want)


def test_matches_sync_rank_answer():
    rng = random.Random(1)
    for _ in range(2000):
        world = rng.randint(2, 8)
        chunks = _chunks(rng.randint(1, 30 * 256))
        present = [rng.random() < 0.8 for _ in chunks]
        bounds = [(s, e) for s, e, _ in chunks]
        for r in range(world):
            owned = ps.owned_positions([s for s, _ in bounds], 256, world, r)
            hits = 0
            for pos in owned:
                if not present[pos]:
                    break
                hits += 1
            assert _reported(r, chunks, present, world) == ps.rank_lookup_tokens(
                bounds, owned, hits
            )


def test_rank_without_chunks_and_empty_request():
    chunks = _chunks(300)  # 2 chunks, world 8: ranks 2..7 own nothing
    keys, cum = ps.async_lookup_keys(chunks, 256, 8, 5)
    assert keys == [] and cum == [300]
    assert ps.async_lookup_keys([], 256, 8, 0) == ([], [0])


def test_select_prefetched():
    owned = [(1, "a", 256, 512), (3, "b", 768, 1024), (5, "c", 1280, 1536)]
    objs = {"a": object(), "b": object(), "c": object()}
    kept, first_fail, unused = ps.select_prefetched(owned, 8, dict(objs))
    assert (
        kept == {1: objs["a"], 3: objs["b"], 5: objs["c"]}
        and first_fail == 8
        and unused == []
    )
    partial = {"a": objs["a"], "c": objs["c"]}  # "b" missing: stop there, "c" unused
    kept, first_fail, unused = ps.select_prefetched(owned, 8, partial)
    assert kept == {1: objs["a"]} and first_fail == 3 and unused == [objs["c"]]
    kept, first_fail, unused = ps.select_prefetched(owned, 8, {})
    assert kept == {} and first_fail == 1 and unused == []


def test_async_no_longer_rejected():
    class Cfg:
        enable_async_loading = True
        use_layerwise = enable_blending = enable_scheduler_bypass_lookup = False
        enable_pd = enable_p2p = False
        external_lookup_client = None

        def get_extra_config_value(self, key, default):
            return {"pcp_shard_store": True}.get(key, default)

    assert ps.shard_store_enabled(Cfg(), True, 8) is True
