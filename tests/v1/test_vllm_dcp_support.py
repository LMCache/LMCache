# SPDX-License-Identifier: Apache-2.0
"""Unit tests for decode-context-parallel (DCP) support in the MP connector.

Under DCP vLLM shards the KV cache across ranks along the token axis, so a
chunk has ``dcp_size`` distinct shards instead of one replicated MLA copy, and
one attention block id spans ``block_size * dcp_size`` global tokens. These
tests pin the three public contracts that encode that:

* ``ParallelStrategy`` -- shard count, this rank's shard index, writer election.
* ``compute_extra_count`` -- how many readers share one stored object.
* ``get_tokens_per_block`` -- the block-size scaling, attention only.

Cross-rank hit folding is covered by
``tests/v1/distributed/test_bitmap_ops.py::test_ranked_chunk_present_only_if_all_ranks_present``,
which already asserts a chunk counts as a hit only when every rank's shard is
present. No GPU or live server needed.
"""

# Standard
from dataclasses import dataclass, field
from types import SimpleNamespace
import importlib.util

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.kv_cache_groups import get_tokens_per_block
from lmcache.integration.vllm.vllm_multi_process_adapter import ParallelStrategy
from lmcache.v1.multiprocess.modules.lookup import compute_extra_count

# ``lmcache_mp_connector`` imports vLLM at module scope, but the unit-test
# environment deliberately runs without vLLM (see
# .buildkite/k3_harness/setup-lmcache-only-env.sh), so those tests import it
# lazily and skip when it is absent. Everything else here uses test doubles
# that mirror the vLLM class names, as test_vllm_kv_cache_groups.py does.
requires_vllm = pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None, reason="requires vLLM"
)


def _strategy(
    *,
    tp_size: int,
    dcp_size: int,
    worker_id: int,
    mla_only: bool = True,
    n_servers: int = 1,
    pp_size: int = 1,
) -> ParallelStrategy:
    """Build an MLA ParallelStrategy for a TP x PP x DCP topology."""
    return ParallelStrategy(
        mla_only=mla_only,
        vllm_world_size=tp_size * pp_size,
        vllm_worker_id=worker_id,
        tp_size=tp_size,
        pp_size=pp_size,
        n_servers=n_servers,
        dcp_size=dcp_size,
    )


# --------------------------------------------------------------------------- #
# ParallelStrategy: shard count, shard index, writer election                  #
# --------------------------------------------------------------------------- #


def test_dcp_size_defaults_to_one_and_preserves_legacy_behaviour():
    """Callers that never set dcp_size must be bit-for-bit unchanged."""
    legacy = ParallelStrategy(
        mla_only=True,
        vllm_world_size=8,
        vllm_worker_id=3,
        tp_size=8,
        pp_size=1,
        n_servers=1,
    )
    assert legacy.dcp_size == 1
    assert legacy.kv_world_size == 1  # MLA replicated: one object per chunk
    assert legacy.kv_worker_id == 0
    assert legacy.is_kv_writer is False  # only rank 0 writes


def test_positional_construction_still_works():
    """Existing positional callers (6 args) must not break on the new field."""
    strategy = ParallelStrategy(False, 1, 0, 1, 1, 1)
    assert strategy.dcp_size == 1


@pytest.mark.parametrize("dcp_size", [2, 4, 8])
def test_kv_world_size_is_dcp_size_under_dcp(dcp_size: int):
    """A chunk has one shard per DCP rank, not one replica for all TP ranks."""
    strategy = _strategy(tp_size=8, dcp_size=dcp_size, worker_id=0)
    assert strategy.kv_world_size == dcp_size


def test_kv_worker_id_is_tp_rank_mod_dcp_size():
    """DCP is a TP subdivision: dcp_rank == tp_rank % dcp_size."""
    dcp_size, tp_size = 2, 8
    observed = [
        _strategy(tp_size=tp_size, dcp_size=dcp_size, worker_id=r).kv_worker_id
        for r in range(tp_size)
    ]
    assert observed == [0, 1, 0, 1, 0, 1, 0, 1]


def test_kv_worker_id_covers_every_shard_exactly_once_per_replica_group():
    """Each replica group of tp/dcp ranks must span all shard indices."""
    dcp_size, tp_size = 4, 8
    ids = [
        _strategy(tp_size=tp_size, dcp_size=dcp_size, worker_id=r).kv_worker_id
        for r in range(tp_size)
    ]
    assert set(ids) == set(range(dcp_size))


def test_exactly_one_writer_per_shard():
    """Two writers on one object would race the write lock; elect one each."""
    dcp_size, tp_size = 2, 8
    writers = [
        r
        for r in range(tp_size)
        if _strategy(tp_size=tp_size, dcp_size=dcp_size, worker_id=r).is_kv_writer
    ]
    assert writers == [0, 1]

    written_shards = [
        _strategy(tp_size=tp_size, dcp_size=dcp_size, worker_id=r).kv_worker_id
        for r in writers
    ]
    assert sorted(written_shards) == list(range(dcp_size))


def test_non_mla_unaffected_by_dcp_field():
    """Head-sharded models already write per rank; DCP must not alter them."""
    strategy = _strategy(tp_size=4, dcp_size=2, worker_id=2, mla_only=False)
    assert strategy.kv_world_size == 4
    assert strategy.kv_worker_id == 2
    assert strategy.is_kv_writer is True


# --------------------------------------------------------------------------- #
# compute_extra_count: readers sharing one stored object                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "label,readers",
    [("non-mla", 1), ("mla-tp8", 8), ("mla-tp4-pp2", 4), ("mla-tp8-dcp2", 4)],
)
def test_extra_count_is_exact_when_reader_count_is_sent(label: str, readers: int):
    """With num_kv_readers the answer is exact regardless of topology.

    tp_size/world_size are deliberately passed as values that would mislead
    the legacy heuristic, to prove the exact path wins.
    """
    assert compute_extra_count(8, 2, readers) == readers - 1, label


@pytest.mark.parametrize(
    "label,tp_size,world_size,expected",
    [
        ("mla-tp8-pp1", 8, 1, 7),
        ("mla-tp4-pp2", 4, 2, 3),
        ("non-mla-tp8", 8, 8, 0),
        ("no-tp-size", 1, 4, 0),
    ],
)
def test_extra_count_falls_back_to_legacy_heuristic(
    label: str, tp_size: int, world_size: int, expected: int
):
    """num_kv_readers == 0 means an older client; keep the previous behaviour."""
    assert compute_extra_count(tp_size, world_size) == expected, label
    assert compute_extra_count(tp_size, world_size, 0) == expected, label


@pytest.mark.parametrize(
    "tp_size,dcp_size,n_servers,mla_only,expected",
    [
        (8, 1, 1, True, 8),  # MLA replicated across all TP ranks
        (8, 2, 1, True, 4),  # DCP splits them into 2 shards
        (8, 8, 1, True, 1),  # fully sharded: one reader each
        (8, 2, 1, False, 1),  # head-sharded: always one reader
        # Multi-server: each server only sees its own tp_size // n_servers
        # ranks, so counting all of them would over-reserve by n_servers.
        (8, 1, 2, True, 4),
        (8, 1, 4, True, 2),
    ],
)
def test_num_kv_readers(
    tp_size: int, dcp_size: int, n_servers: int, mla_only: bool, expected: int
):
    """The client-side reader count the server now relies on."""
    strategy = _strategy(
        tp_size=tp_size,
        dcp_size=dcp_size,
        worker_id=0,
        mla_only=mla_only,
        n_servers=n_servers,
    )
    assert strategy.num_kv_readers == expected


def test_reader_count_round_trips_and_balances_locks():
    """End-to-end of the accounting: producer -> key -> server == readers - 1."""
    for tp_size, dcp_size, pp_size in [(8, 1, 1), (8, 2, 1), (4, 1, 2), (8, 2, 2)]:
        strategy = _strategy(
            tp_size=tp_size, dcp_size=dcp_size, worker_id=0, pp_size=pp_size
        )
        readers = strategy.num_kv_readers
        assert readers == tp_size // dcp_size
        assert (
            compute_extra_count(tp_size, strategy.kv_world_size, readers) == readers - 1
        )


# --------------------------------------------------------------------------- #
# PP x DCP shard space                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "tp_size,pp_size,dcp_size",
    [(8, 1, 1), (8, 1, 2), (4, 2, 1), (4, 2, 2), (2, 4, 2)],
)
def test_shard_space_is_pp_times_dcp(tp_size: int, pp_size: int, dcp_size: int):
    """Shards are (pipeline stage, token shard); ids cover it exactly once."""
    world = tp_size * pp_size
    ids = [
        _strategy(
            tp_size=tp_size, dcp_size=dcp_size, worker_id=r, pp_size=pp_size
        ).kv_worker_id
        for r in range(world)
    ]
    expected_shards = pp_size * dcp_size
    strategy0 = _strategy(
        tp_size=tp_size, dcp_size=dcp_size, worker_id=0, pp_size=pp_size
    )
    assert strategy0.kv_world_size == expected_shards
    assert set(ids) == set(range(expected_shards)), "every shard must be covered"
    assert all(0 <= i < expected_shards for i in ids)


def test_writers_cover_every_shard_exactly_once_with_pp():
    """One writer per (stage, shard) -- no shard unwritten, none written twice."""
    tp_size, pp_size, dcp_size = 4, 2, 2
    world = tp_size * pp_size
    writers = [
        r
        for r in range(world)
        if _strategy(
            tp_size=tp_size, dcp_size=dcp_size, worker_id=r, pp_size=pp_size
        ).is_kv_writer
    ]
    shards = sorted(
        _strategy(
            tp_size=tp_size, dcp_size=dcp_size, worker_id=r, pp_size=pp_size
        ).kv_worker_id
        for r in writers
    )
    assert shards == list(range(pp_size * dcp_size))


# --------------------------------------------------------------------------- #
# get_tokens_per_block: attention scaled, Mamba not                      #
# --------------------------------------------------------------------------- #


@dataclass
class _FakeMambaSpec:
    """Stands in for MambaSpec: anything that is not an AttentionSpec."""

    block_size: int


@dataclass
class AttentionSpec:
    """Double for vLLM's AttentionSpec base; detection is by class name."""

    block_size: int


@dataclass
class FullAttentionSpec(AttentionSpec):
    """Double for a concrete paged-attention spec."""


@dataclass
class UniformTypeKVCacheSpecs:
    """Double for vLLM's container: derives from KVCacheSpec, not
    AttentionSpec, so it must be unwrapped before the MRO check."""

    block_size: int
    kv_cache_specs: dict = field(default_factory=dict)


def _attention_spec(block_size: int) -> AttentionSpec:
    """Build an attention spec double so the MRO-name check is exercised."""
    return FullAttentionSpec(block_size=block_size)


@pytest.mark.parametrize("dcp_size", [2, 4])
def test_attention_group_scaled_by_dcp(dcp_size: int):
    """One attention block id spans block_size * dcp global tokens."""
    spec = _attention_spec(1024)
    assert get_tokens_per_block(spec, dcp_size) == 1024 * dcp_size


def test_mamba_group_never_scaled():
    """Recurrent state is replicated per rank, so its span is unchanged."""
    spec = _FakeMambaSpec(block_size=1024)
    assert get_tokens_per_block(spec, 2) == 1024
    assert get_tokens_per_block(spec, 8) == 1024


def test_dcp_one_is_identity_for_every_spec_type():
    """No DCP means no scaling anywhere -- the legacy path is untouched."""
    assert get_tokens_per_block(_attention_spec(1024), 1) == 1024
    assert get_tokens_per_block(_FakeMambaSpec(block_size=1024), 1) == 1024


def test_hybrid_layout_produces_mixed_spans():
    """MLA + KDA at block 1024, dcp 2 -> [2048, 1024]; LCM alignment 2048."""
    # Standard
    import math

    specs = [_attention_spec(1024), _FakeMambaSpec(block_size=1024)]
    spans = [get_tokens_per_block(s, 2) for s in specs]
    assert spans == [2048, 1024]
    assert math.lcm(*spans) == 2048  # scheduler commit granularity


# --------------------------------------------------------------------------- #
# validate_dcp_support: fail closed on unproven topologies                     #
# --------------------------------------------------------------------------- #


def _import_validate_dcp_support():
    """Import lazily: the module pulls in vLLM at import time."""
    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import validate_dcp_support

    return validate_dcp_support


def _config(
    *,
    dcp_size: int = 1,
    pcp_size: int = 1,
    pp_size: int = 1,
    interleave: int = 1,
    tp_size: int = 8,
) -> SimpleNamespace:
    """Minimal stand-in exposing only the parallel_config fields read."""
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_size,
            prefill_context_parallel_size=pcp_size,
            pipeline_parallel_size=pp_size,
            cp_kv_cache_interleave_size=interleave,
            tensor_parallel_size=tp_size,
            world_size=tp_size * pp_size,
        )
    )


@requires_vllm
def test_validate_accepts_supported_dcp_topology():
    _import_validate_dcp_support()(_config(dcp_size=2, tp_size=8), 1)


@requires_vllm
def test_validate_is_noop_without_dcp():
    """Without DCP, even otherwise-rejected settings must not raise."""
    _import_validate_dcp_support()(
        _config(dcp_size=1, pcp_size=4, pp_size=4, interleave=64), 1
    )


@requires_vllm
def test_validate_accepts_pp_with_dcp():
    """PP + DCP is supported: the shard space is the (stage, shard) product,
    and num_kv_readers gives the server the exact reader count it needs."""
    _import_validate_dcp_support()(_config(dcp_size=2, pp_size=2), 1)


@requires_vllm
def test_validate_rejects_pcp_with_dcp():
    with pytest.raises(ValueError, match="prefill-context parallelism"):
        _import_validate_dcp_support()(_config(dcp_size=2, pcp_size=2), 1)


@requires_vllm
def test_validate_rejects_nontrivial_interleave():
    """The silent-corruption guard: wrong KV stored, no crash, without it."""
    with pytest.raises(ValueError, match="cp_kv_cache_interleave_size"):
        _import_validate_dcp_support()(_config(dcp_size=2, interleave=64), 1)


@requires_vllm
def test_validate_accepts_multi_server_when_each_holds_a_full_shard_set():
    """Ranks split into contiguous per-server blocks; a block of >= dcp_size
    consecutive ranks covers every shard exactly once, so each server is
    independently servable."""
    _import_validate_dcp_support()(_config(dcp_size=2, tp_size=8), 2)
    _import_validate_dcp_support()(_config(dcp_size=4, tp_size=8), 2)


@requires_vllm
def test_validate_rejects_servers_too_small_to_hold_a_shard_set():
    """TP8 across 2 servers leaves 4 ranks each, which cannot hold 8 shards.
    Lookup takes the min across servers, so a partial server reports 0 hits."""
    with pytest.raises(ValueError, match="complete set of shards"):
        _import_validate_dcp_support()(_config(dcp_size=8, tp_size=8), 2)


@requires_vllm
def test_validate_allows_multiple_servers_without_dcp():
    """The multi-server MLA path is unchanged when DCP is off."""
    _import_validate_dcp_support()(_config(dcp_size=1), 4)


@pytest.mark.parametrize("tp_size,dcp_size", [(2, 2), (8, 2), (8, 4), (8, 8)])
def test_dcp_rank_matches_vllm_group_construction(tp_size: int, dcp_size: int):
    """``dcp_rank == tp_rank % dcp_size`` must match how vLLM builds DCP groups.

    vLLM reshapes the rank grid into contiguous runs of ``dcp_size``
    (``all_ranks.transpose(-1, -2).reshape(-1, dcp_size)``), so with pcp == 1 a
    worker's rank-in-group is its offset within a consecutive block of TP
    ranks. This reproduces that construction and asserts our derivation agrees;
    it fails if upstream ever switches to a strided grouping.
    """
    # Third Party
    import torch

    # pcp = pp = dp = 1, mirroring parallel_state.initialize_model_parallel.
    all_ranks = torch.arange(tp_size).reshape(-1, 1, 1, 1, tp_size)
    groups = all_ranks.transpose(-1, -2).reshape(-1, dcp_size).tolist()
    vllm_dcp_rank = {rank: pos for group in groups for pos, rank in enumerate(group)}

    # kv_worker_id is the public surface of the shard index under MLA + DCP.
    for rank in range(tp_size):
        strategy = _strategy(tp_size=tp_size, dcp_size=dcp_size, worker_id=rank)
        assert strategy.kv_worker_id == vllm_dcp_rank[rank], rank

    # The elected writers must cover every shard exactly once.
    writers = [
        r
        for r in range(tp_size)
        if _strategy(tp_size=tp_size, dcp_size=dcp_size, worker_id=r).is_kv_writer
    ]
    assert sorted(vllm_dcp_rank[w] for w in writers) == list(range(dcp_size))


def test_uniform_type_wrapper_still_scales_under_dcp():
    """vLLM wraps same-type layers in UniformTypeKVCacheSpecs, which derives
    from KVCacheSpec rather than AttentionSpec. Without unwrapping, a wrapped
    attention group skips DCP scaling and every object is sized dcp times too
    large -- silently, since the arithmetic stays self-consistent."""
    inner = _attention_spec(1024)
    wrapped = UniformTypeKVCacheSpecs(
        block_size=1024, kv_cache_specs={"layer.0": inner, "layer.1": inner}
    )
    assert get_tokens_per_block(wrapped, 2) == 2048
    assert get_tokens_per_block(wrapped, 1) == 1024


def test_uniform_type_wrapper_of_mamba_is_not_scaled():
    """Unwrapping must not turn a wrapped recurrent group into a scaled one."""

    wrapped = UniformTypeKVCacheSpecs(
        block_size=1024, kv_cache_specs={"l0": _FakeMambaSpec(block_size=1024)}
    )
    assert get_tokens_per_block(wrapped, 4) == 1024


@pytest.mark.parametrize(
    "tp_size,dcp_size,n_servers",
    [(8, 2, 1), (8, 2, 2), (8, 4, 2), (8, 2, 4), (4, 2, 2)],
)
def test_every_server_elects_one_writer_per_shard(
    tp_size: int, dcp_size: int, n_servers: int
):
    """Each server must independently cover every shard exactly once.

    Ranks map to servers in contiguous blocks and lookup takes the minimum hit
    count across servers, so a server missing any shard reports no hits at all.
    """
    ranks_per_server = tp_size // n_servers
    for server in range(n_servers):
        block = range(server * ranks_per_server, (server + 1) * ranks_per_server)
        shards = sorted(
            _strategy(
                tp_size=tp_size, dcp_size=dcp_size, worker_id=r, n_servers=n_servers
            ).kv_worker_id
            for r in block
            if _strategy(
                tp_size=tp_size, dcp_size=dcp_size, worker_id=r, n_servers=n_servers
            ).is_kv_writer
        )
        assert shards == list(range(dcp_size)), f"server {server}"


@pytest.mark.parametrize(
    "tp_size,dcp_size,n_servers",
    [(8, 2, 1), (8, 4, 1), (8, 8, 1), (8, 2, 2), (8, 4, 2), (4, 2, 2), (6, 2, 2)],
)
def test_num_kv_readers_never_under_reserves_any_shard(
    tp_size: int, dcp_size: int, n_servers: int
):
    """The single reader count must cover the *busiest* shard.

    ``kv_tp_size / dcp_size`` need not divide evenly -- vLLM accepts TP=6 with
    DCP=2, and across two servers each server's three ranks leave shard 0 with
    two readers and shard 1 with one. Under-reserving unpins an object while a
    reader is still copying, so the count is derived here by simulating which
    ranks actually read each shard rather than by repeating the formula.
    """
    ranks_per_server = tp_size // n_servers
    declared = _strategy(
        tp_size=tp_size, dcp_size=dcp_size, worker_id=0, n_servers=n_servers
    ).num_kv_readers

    for server in range(n_servers):
        block = range(server * ranks_per_server, (server + 1) * ranks_per_server)
        readers_per_shard: dict[int, int] = {}
        for rank in block:
            shard = _strategy(
                tp_size=tp_size, dcp_size=dcp_size, worker_id=rank, n_servers=n_servers
            ).kv_worker_id
            readers_per_shard[shard] = readers_per_shard.get(shard, 0) + 1
        assert declared >= max(readers_per_shard.values()), (
            f"server {server} shard readers {readers_per_shard} exceed {declared}"
        )
