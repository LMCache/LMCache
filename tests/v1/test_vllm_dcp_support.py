# SPDX-License-Identifier: Apache-2.0
"""DCP support in the MP connector: pins ``ParallelStrategy`` (shard count,
shard index, writer election), ``require_num_kv_readers`` (readers per object),
and ``get_tokens_per_block`` (attention-only block scaling). No GPU needed."""

# Standard
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Literal
import importlib.util

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.kv_cache_groups import get_tokens_per_block
from lmcache.integration.vllm.vllm_multi_process_adapter import ParallelStrategy
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

# ``lmcache_mp_connector`` imports vLLM, but the k3 unit env runs without it
# (.buildkite/k3_harness/setup-lmcache-only-env.sh) -- import lazily and skip.
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
# require_num_kv_readers: readers sharing one stored object                    #
# --------------------------------------------------------------------------- #


def _reader_key(readers: int) -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name="m",
        world_size=8,
        num_kv_readers=readers,
        worker_id=0,
        token_ids=(1, 2, 3),
        start=0,
        end=3,
        request_id="req",
    )


@pytest.mark.parametrize(
    "label,readers",
    [("non-mla", 1), ("mla-tp8", 8), ("mla-tp4-pp2", 4), ("mla-tp8-dcp2", 4)],
)
def test_reader_count_is_exact_when_sent(label: str, readers: int):
    """The declared reader count is used as-is: one read lock per reader."""
    assert _reader_key(readers).require_num_kv_readers() == readers, label


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
        assert _reader_key(readers).require_num_kv_readers() == readers


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
class MambaSpec:
    """Mamba spec double detected by its public class name."""

    block_size: int
    mamba_cache_mode: str = "align"


@dataclass
class AttentionSpec:
    """Double for vLLM's AttentionSpec base; detection is by class name."""

    block_size: int
    dcp_replicated: bool = False


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


def test_dcp_replicated_attention_group_never_scaled():
    """Replicated DFlash draft KV keeps ordinary DCP1 block coordinates."""
    spec = FullAttentionSpec(block_size=16, dcp_replicated=True)
    assert get_tokens_per_block(spec, 4) == 16


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


def _import_connector_geometry_helpers():
    """Import helpers lazily because their module imports vLLM."""
    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import (
        get_group_tokens_per_block,
        get_lmcache_model_name,
        get_lmcache_scheduler_block_size,
        validate_mamba_step_alignment,
    )

    return (
        get_group_tokens_per_block,
        get_lmcache_model_name,
        get_lmcache_scheduler_block_size,
        validate_mamba_step_alignment,
    )


def _hybrid_kv_cache_config(
    attention_block_size: int = 2304,
    mamba_block_size: int = 2304,
) -> SimpleNamespace:
    """Resolved GLM-like groups after platform page-size equalization."""
    return SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=_attention_spec(attention_block_size)),
            SimpleNamespace(kv_cache_spec=MambaSpec(mamba_block_size)),
        ]
    )


def _geometry_config(
    *,
    dcp_size: int = 4,
    interleave: int = 4,
    base_block_size: int = 256,
    max_num_batched_tokens: int = 4096,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(model="org/glm-5.3-flash"),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_size,
            cp_kv_cache_interleave_size=interleave,
        ),
        cache_config=SimpleNamespace(
            block_size=base_block_size,
            mamba_cache_mode="align",
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_num_batched_tokens),
    )


@requires_vllm
def test_glm_dcp4_geometry_resolves_physical_mamba_block():
    (
        get_group_tokens_per_block,
        _,
        get_lmcache_scheduler_block_size,
        _,
    ) = _import_connector_geometry_helpers()
    config = _geometry_config()
    kv_config = _hybrid_kv_cache_config()

    assert get_group_tokens_per_block(config, kv_config) == [9216, 2304]
    assert get_lmcache_scheduler_block_size(config, kv_config) == 9216


@requires_vllm
def test_mamba_alignment_uses_resolved_page_not_cli_block_size():
    *_, validate_mamba_step_alignment = _import_connector_geometry_helpers()
    kv_config = _hybrid_kv_cache_config()

    with pytest.raises(ValueError, match=r"block_size=2304"):
        validate_mamba_step_alignment(
            _geometry_config(max_num_batched_tokens=2048), kv_config
        )

    validate_mamba_step_alignment(
        _geometry_config(max_num_batched_tokens=4096), kv_config
    )


@requires_vllm
@pytest.mark.parametrize(
    "kv_layout",
    ["BLHNC", "BLNHC"],
)
def test_mamba_unified_view_preserves_blocks_first_pool_stride(
    kv_layout: Literal["BLHNC", "BLNHC"],
):
    """Jovian Mamba pages use the padded-stride-capable opaque format."""
    # Third Party
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.kv_cache_interface import MambaSpec as VllmMambaSpec

    # First Party
    from lmcache.integration.vllm.kv_cache_group_edits import (
        apply_kv_cache_group_edits,
    )
    from lmcache.v1.gpu_connector.utils import (
        resolve_block_stride_and_log_layout,
    )
    import lmcache.lmcache_native as lmcache_native

    num_blocks, num_layers, row, page_elems = 3, 2, 13, 16
    pool = torch.arange(num_blocks * num_layers * page_elems, dtype=torch.float32)
    layer = pool.as_strided(
        (num_blocks, 1, 1, row),
        (num_layers * page_elems, row, row, 1),
        storage_offset=page_elems,
    )
    spec = VllmMambaSpec(
        block_size=4,
        shapes=((row,),),
        dtypes=(torch.float32,),
        page_size_padded=page_elems * torch.float32.itemsize,
        mamba_cache_mode="align",
    )
    kv_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["mamba"], spec)],
        kv_cache_layout=kv_layout,
    )

    edited = apply_kv_cache_group_edits(
        kv_config,
        {"mamba": layer},
        {"kv_layout": kv_layout},
    )["mamba"]

    assert isinstance(edited, torch.Tensor)
    assert edited.shape == (3, 4, 4)
    assert edited.stride() == (num_layers * page_elems, 4, 1)
    assert edited.stride(0) == num_layers * page_elems
    assert edited.data_ptr() == layer.data_ptr()
    assert (
        resolve_block_stride_and_log_layout(
            [edited],
            lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
            layer_idx=0,
            group_idx=0,
        )
        == num_layers * page_elems
    )


@requires_vllm
@pytest.mark.parametrize(
    ("kv_layout", "target_shape", "target_strides"),
    [
        ("BLHNC", (3, 1, 4, 8), (80, 32, 8, 1)),
        ("BLNHC", (3, 4, 1, 8), (80, 8, 32, 1)),
    ],
)
def test_padded_attention_page_views_preserve_hma_block_padding(
    kv_layout: Literal["BLHNC", "BLNHC"],
    target_shape: tuple[int, ...],
    target_strides: tuple[int, ...],
):
    """Jovian HMA MLA and DFlash pages become padded rank-3 tensors."""
    # Third Party
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    from vllm.v1.kv_cache_interface import MambaSpec as VllmMambaSpec
    from vllm.v1.kv_cache_interface import (
        MLAAttentionSpec,
    )

    # First Party
    from lmcache.integration.vllm.kv_cache_group_edits import (
        apply_kv_cache_group_edits,
    )
    from lmcache.v1.gpu_connector.utils import (
        resolve_block_stride_and_log_layout,
    )
    import lmcache.lmcache_native as lmcache_native

    target_pool = torch.arange(3 * 80, dtype=torch.uint8)
    target = target_pool.as_strided(target_shape, target_strides)
    draft_shape = (*target_shape[:-1], 16)
    if kv_layout == "BLHNC":
        draft_strides = (96, 64, 16, 1)
    else:
        draft_strides = (96, 16, 64, 1)
    draft_pool = torch.arange(3 * 96, dtype=torch.uint8)
    draft = draft_pool.as_strided(draft_shape, draft_strides)
    if kv_layout == "BLHNC":
        dflash_shape = (3, 2, 4, 16)
        dflash_strides = (160, 64, 16, 1)
    else:
        dflash_shape = (3, 4, 2, 16)
        dflash_strides = (160, 32, 16, 1)
    dflash_pool = torch.arange(3 * 160, dtype=torch.uint8)
    dflash = dflash_pool.as_strided(dflash_shape, dflash_strides)
    plain_strides = (64, 64, 16, 1) if kv_layout == "BLHNC" else (64, 16, 64, 1)
    plain_pool = torch.arange(3 * 64, dtype=torch.uint8)
    plain = plain_pool.as_strided(draft_shape, plain_strides)
    mamba_pool = torch.arange(3 * 32, dtype=torch.float32)
    mamba = mamba_pool.as_strided((3, 1, 1, 13), (32, 13, 13, 1))
    mla_spec = MLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.uint8,
    )
    mamba_spec = VllmMambaSpec(
        block_size=4,
        shapes=((13,),),
        dtypes=(torch.float32,),
        page_size_padded=64,
        mamba_cache_mode="align",
    )
    draft_spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.uint8,
    )
    dflash_spec = FullAttentionSpec(
        block_size=8,
        num_kv_heads=2,
        head_size=4,
        dtype=torch.uint8,
    )
    kv_config = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["target"], mla_spec),
            KVCacheGroupSpec(["draft"], draft_spec),
            KVCacheGroupSpec(["dflash"], dflash_spec),
            KVCacheGroupSpec(["plain"], draft_spec),
            KVCacheGroupSpec(["mamba"], mamba_spec),
        ],
        kv_cache_layout=kv_layout,
    )

    edited_caches = apply_kv_cache_group_edits(
        kv_config,
        {
            "target": target,
            "draft": draft,
            "dflash": dflash,
            "plain": plain,
            "mamba": mamba,
        },
        {"kv_layout": kv_layout},
    )
    edited = edited_caches["target"]
    edited_draft = edited_caches["draft"]
    edited_dflash = edited_caches["dflash"]

    assert isinstance(edited, torch.Tensor)
    assert edited.shape == (3, 4, 8)
    assert edited.stride() == (80, 8, 1)
    assert edited.data_ptr() == target.data_ptr()
    assert torch.equal(edited[1].reshape(-1), target[1].reshape(-1))
    assert isinstance(edited_draft, torch.Tensor)
    assert edited_draft.shape == (3, 4, 16)
    assert edited_draft.stride() == (96, 16, 1)
    assert edited_draft.data_ptr() == draft.data_ptr()
    assert torch.equal(edited_draft[1].reshape(-1), draft[1].reshape(-1))
    assert isinstance(edited_dflash, torch.Tensor)
    assert edited_dflash.shape == (3, 8, 16)
    assert edited_dflash.stride() == (160, 16, 1)
    assert edited_dflash.data_ptr() == dflash.data_ptr()
    assert torch.equal(edited_dflash[1].reshape(-1), dflash[1].reshape(-1))
    assert edited_caches["plain"] is plain
    assert (
        resolve_block_stride_and_log_layout(
            [edited],
            lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
            layer_idx=0,
            group_idx=0,
        )
        == 80
    )
    assert (
        resolve_block_stride_and_log_layout(
            [edited_draft],
            lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
            layer_idx=0,
            group_idx=1,
        )
        == 96
    )
    assert (
        resolve_block_stride_and_log_layout(
            [edited_dflash],
            lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
            layer_idx=0,
            group_idx=2,
        )
        == 160
    )


@requires_vllm
def test_dcp_interleave_participates_in_cache_identity():
    _, get_lmcache_model_name, _, _ = _import_connector_geometry_helpers()
    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import (
        get_lmcache_base_model_name,
    )

    interleave4 = get_lmcache_model_name(_geometry_config(interleave=4))
    interleave8 = get_lmcache_model_name(_geometry_config(interleave=8))
    assert interleave4 != interleave8
    assert interleave4.endswith("##lmcache-dcp-layout-v1-d4-interleave4")
    assert get_lmcache_base_model_name(interleave4) == "org/glm-5.3-flash"


@requires_vllm
def test_base_model_name_preserves_undecorated_names():
    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import (
        get_lmcache_base_model_name,
    )

    assert get_lmcache_base_model_name("org/glm-5.3-flash") == ("org/glm-5.3-flash")


@requires_vllm
def test_trivial_interleave_preserves_legacy_cache_identity():
    _, get_lmcache_model_name, _, _ = _import_connector_geometry_helpers()

    assert (
        get_lmcache_model_name(_geometry_config(dcp_size=4, interleave=1))
        == "org/glm-5.3-flash"
    )
    assert (
        get_lmcache_model_name(_geometry_config(dcp_size=1, interleave=4))
        == "org/glm-5.3-flash"
    )


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
    cache_block_size: int = 256,
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
        ),
        cache_config=SimpleNamespace(block_size=cache_block_size),
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
def test_validate_accepts_interleave_when_layout_is_namespaced():
    """Opaque per-rank pages round-trip when their key namespace is isolated."""
    _import_validate_dcp_support()(
        _config(dcp_size=4, interleave=4), 1, _hybrid_kv_cache_config()
    )


@requires_vllm
def test_validate_uses_resolved_attention_block_for_interleave():
    """The resolved 2304 page, rather than the CLI 256 page, is authoritative."""
    validate = _import_validate_dcp_support()
    kv_config = _hybrid_kv_cache_config(attention_block_size=2304)

    validate(_config(dcp_size=4, interleave=768), 1, kv_config)


@requires_vllm
def test_validate_rejects_interleave_not_dividing_resolved_attention_block():
    with pytest.raises(ValueError, match="evenly divide"):
        _import_validate_dcp_support()(
            _config(dcp_size=4, interleave=1000),
            1,
            _hybrid_kv_cache_config(attention_block_size=2304),
        )


@requires_vllm
def test_validate_rejects_nonpositive_interleave():
    with pytest.raises(ValueError, match=">= 1"):
        _import_validate_dcp_support()(_config(dcp_size=4, interleave=0), 1)


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
    """Reproduces vLLM's contiguous-run group construction and asserts
    ``dcp_rank == tp_rank % dcp_size`` agrees; fails if upstream goes
    strided."""
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
    """UniformTypeKVCacheSpecs derives from KVCacheSpec, not AttentionSpec;
    without unwrapping, wrapped attention groups skip DCP scaling."""
    inner = _attention_spec(1024)
    wrapped = UniformTypeKVCacheSpecs(
        block_size=1024, kv_cache_specs={"layer.0": inner, "layer.1": inner}
    )
    assert get_tokens_per_block(wrapped, 2) == 2048
    assert get_tokens_per_block(wrapped, 1) == 1024


def test_uniform_type_wrapper_of_replicated_attention_is_not_scaled():
    """Wrapped DFlash leaves retain replicated block coordinates."""
    inner = FullAttentionSpec(block_size=16, dcp_replicated=True)
    wrapped = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"draft.0": inner, "draft.1": inner},
    )
    assert get_tokens_per_block(wrapped, 4) == 16


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
    """Lookup takes the min across servers, so each server must cover every
    shard exactly once."""
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
    """The count must cover the busiest shard: kv_tp/dcp can split unevenly
    (e.g. TP=6, DCP=2, 2 servers), and under-reserving unpins mid-copy.
    Derived by simulating readers, not by repeating the formula."""
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
