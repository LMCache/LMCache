# SPDX-License-Identifier: Apache-2.0
"""Pure logic of lmcache.v1.pcp_shard (no torch.distributed, no engine)."""

# Standard
from types import SimpleNamespace
import itertools
import random

# Third Party
import pytest

# First Party
from lmcache.v1 import pcp_shard as ps

CHUNK = 256


def bounds_for(n_tokens, chunk=CHUNK):
    return [(s, min(s + chunk, n_tokens)) for s in range(0, n_tokens, chunk)]


def per_rank_answers(bounds, present, world, chunk=CHUNK):
    out = []
    for r in range(world):
        owned = ps.owned_positions([s for s, _ in bounds], chunk, world, r)
        hits = 0
        for pos in owned:
            if not present[pos]:
                break
            hits += 1
        out.append(ps.rank_lookup_tokens(bounds, owned, hits))
    return out


def test_owner_round_robin():
    assert [ps.chunk_owner(i * CHUNK, CHUNK, 8) for i in range(10)] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        0,
        1,
    ]
    # the partial last chunk belongs to its index's owner
    assert ps.chunk_owner(9 * CHUNK, CHUNK, 8) == 1
    assert ps.owned_positions([0, 256, 512, 768, 1024], CHUNK, 4, 0) == [0, 4]
    assert ps.owned_positions([512, 768], CHUNK, 4, 2) == [0]


@pytest.mark.parametrize("world", [2, 3, 4, 8])
def test_lookup_min_is_sharded_prefix_exhaustive(world):
    """For every presence pattern of up to 10 chunks (and a partial last
    chunk), min over ranks == longest prefix of present chunks."""
    for n_chunks in range(0, 11):
        n_tokens = n_chunks * CHUNK - (37 if n_chunks else 0)
        bounds = bounds_for(n_tokens)
        assert len(bounds) == n_chunks
        for present in itertools.product([True, False], repeat=n_chunks):
            got = ps.combine_lookup(per_rank_answers(bounds, present, world))
            assert got == ps.global_prefix_tokens(bounds, present), (present, world)


def test_lookup_random_long():
    rng = random.Random(0)
    for _ in range(2000):
        world = rng.choice([2, 4, 8])
        n_tokens = rng.randint(0, 800 * CHUNK)
        bounds = bounds_for(n_tokens)
        present = [rng.random() > 0.01 for _ in bounds]
        got = ps.combine_lookup(per_rank_answers(bounds, present, world))
        assert got == ps.global_prefix_tokens(bounds, present)


def test_rank_without_chunks_vouches_for_all():
    bounds = bounds_for(2 * CHUNK)
    # world 8, rank 5 owns nothing
    assert ps.rank_lookup_tokens(bounds, [], 0) == 2 * CHUNK
    assert ps.rank_lookup_tokens([], [], 0) == 0


def msg(fp, first_fail, metas):
    return (fp, first_fail, {j: {"m": j} for j in metas})


def test_agree_prefix():
    owners = [j % 4 for j in range(10)]
    owned = {r: [j for j in range(10) if j % 4 == r] for r in range(4)}
    full = [msg("a", 10, owned[r]) for r in range(4)]
    assert ps.agree_prefix(full, 10, owners) == 10
    # rank 2 fails at chunk 6 -> prefix 6
    m = list(full)
    m[2] = msg("a", 6, [2])
    assert ps.agree_prefix(m, 10, owners) == 6
    # the reported first failure counts even if metadata went further
    m2 = list(full)
    m2[2] = msg("a", 6, owned[2])
    assert ps.agree_prefix(m2, 10, owners) == 6
    # two failures: the smaller wins
    m[1] = msg("a", 1, [])
    assert ps.agree_prefix(m, 10, owners) == 1
    # fingerprint mismatch -> 0
    m = list(full)
    m[3] = msg("b", 10, owned[3])
    assert ps.agree_prefix(m, 10, owners) == 0
    # a missing message -> 0
    m = list(full)
    m[0] = None
    assert ps.agree_prefix(m, 10, owners) == 0
    # inconsistent message (claims no failure but lacks chunk 5's metadata)
    m = list(full)
    m[1] = msg("a", 10, [1, 9])
    assert ps.agree_prefix(m, 10, owners) == 5
    # nothing to load
    assert ps.agree_prefix([msg("a", 0, [])] * 4, 0, []) == 0


def test_fingerprint_stable_and_sensitive():
    b = bounds_for(1000)
    h = [11, 22, 33, 44]
    assert ps.fingerprint(b, h) == ps.fingerprint(list(b), list(h))
    assert ps.fingerprint(b, h) != ps.fingerprint(b, [11, 22, 33, 45])
    assert ps.fingerprint(b, h) != ps.fingerprint(b[:3], h[:3])
    assert ps.fingerprint(b, h) != ps.fingerprint(bounds_for(999), h)


def test_exchange_and_broadcast_single_process_fakes():
    """Records the collective sequence each rank would issue: identical
    (src order, count) on every rank whatever it fetched."""
    world = 4
    owners = [j % world for j in range(7)]
    fails = {0: 7, 1: 7, 2: 6, 3: 7}  # rank 2 fails at chunk 6 (its 2nd)
    msgs_by_src = {
        r: (
            "fp",
            fails[r],
            {j: {"n": j} for j in range(7) if owners[j] == r and j < fails[r]},
        )
        for r in range(world)
    }
    seqs = []
    for rank in range(world):
        calls = []

        def bobj(obj, src, calls=calls, rank=rank):
            calls.append(("obj", src))
            if src == rank:
                assert obj == msgs_by_src[rank]
            return msgs_by_src[src]

        def btensor(t, src, calls=calls):
            calls.append(("tensor", src))

        msgs = ps.exchange(rank, world, msgs_by_src[rank], bobj)
        prefix = ps.agree_prefix(msgs, 7, owners)
        assert prefix == 6
        local = {j: f"t{j}" for j in range(prefix) if owners[j] == rank}
        out = ps.broadcast_chunks(
            rank, prefix, owners, msgs, local, lambda meta: f"r{meta['n']}", btensor
        )
        assert sorted(out) == list(range(6))
        seqs.append(calls)
    assert all(s == seqs[0] for s in seqs)
    assert seqs[0] == [("obj", r) for r in range(4)] + [
        ("tensor", j % 4) for j in range(6)
    ]


def cfg(extra, **flags):
    c = SimpleNamespace(
        extra_config=extra,
        use_layerwise=False,
        enable_async_loading=False,
        enable_blending=False,
        enable_scheduler_bypass_lookup=False,
        enable_pd=False,
        enable_p2p=False,
        external_lookup_client=None,
    )
    for k, v in flags.items():
        setattr(c, k, v)
    c.get_extra_config_value = lambda key, default=None: (
        c.extra_config.get(key, default) if c.extra_config is not None else default
    )
    return c


def test_shard_store_enabled_gates():
    on = {"pcp_shard_store": True}
    assert ps.shard_store_enabled(cfg(on), True, 8)
    assert ps.shard_store_enabled(cfg({"pcp_shard_store": "true"}), True, 8)
    assert not ps.shard_store_enabled(cfg({}), True, 8)
    assert not ps.shard_store_enabled(cfg({"pcp_shard_store": False}), True, 8)
    assert not ps.shard_store_enabled(cfg({"pcp_shard_store": "0"}), True, 8)
    assert not ps.shard_store_enabled(cfg(None), True, 8)
    assert not ps.shard_store_enabled(cfg(on), False, 8)  # not MLA
    assert not ps.shard_store_enabled(cfg(on), True, 1)  # single rank
    off_sofr = {"pcp_shard_store": True, "save_only_first_rank": False}
    assert not ps.shard_store_enabled(cfg(off_sofr), True, 8)
    for flag in [
        "use_layerwise",
        "enable_async_loading",
        "enable_blending",
        "enable_scheduler_bypass_lookup",
        "enable_pd",
        "enable_p2p",
    ]:
        with pytest.raises(ValueError, match=flag):
            ps.shard_store_enabled(cfg(on, **{flag: True}), True, 8)
        # not requested: the flag does not matter
        assert not ps.shard_store_enabled(cfg({}, **{flag: True}), True, 8)
    with pytest.raises(ValueError, match="external_lookup_client"):
        ps.shard_store_enabled(cfg(on, external_lookup_client="x://y"), True, 8)


def test_check_broadcast_group():
    class Group:
        def __init__(self, world, rank):
            self.world_size, self.rank_in_group = world, rank

        def broadcast(self, t, src):
            return t

    ps.check_broadcast_group(Group(8, 3).broadcast, 3, 8)
    ps.check_broadcast_group(lambda t, s: t, 3, 8)  # plain function: trusted
    with pytest.raises(ValueError):
        ps.check_broadcast_group(Group(1, 0).broadcast, 3, 8)  # TP group of 1
    with pytest.raises(ValueError):
        ps.check_broadcast_group(Group(8, 2).broadcast, 3, 8)
