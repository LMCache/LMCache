# SPDX-License-Identifier: Apache-2.0
"""DCP support in the MP connector: pins ``ParallelStrategy`` (kv rank space),
the server's shard keying against vLLM's DCP group construction, and
``get_tokens_per_block`` (attention-only block scaling). No GPU needed."""

# Standard
from dataclasses import dataclass, field
from types import SimpleNamespace
import importlib.util

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.kv_cache_groups import get_tokens_per_block
from lmcache.integration.vllm.vllm_multi_process_adapter import ParallelStrategy
from lmcache.v1.distributed.api import (
    ObjectKey,
    ShardReplicationDesc,
    ipc_key_to_object_keys,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.token_codec import pack_token_ids

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
    n_servers: int = 1,
    pp_size: int = 1,
) -> ParallelStrategy:
    """Build a ParallelStrategy for a TP x PP x DCP topology."""
    return ParallelStrategy(
        vllm_world_size=tp_size * pp_size,
        vllm_worker_id=worker_id,
        tp_size=tp_size,
        pp_size=pp_size,
        n_servers=n_servers,
        dcp_size=dcp_size,
    )


# --------------------------------------------------------------------------- #
# ParallelStrategy: kv rank space                                              #
# --------------------------------------------------------------------------- #


def test_dcp_size_defaults_to_one_and_preserves_legacy_behaviour():
    """Callers that never set dcp_size must be bit-for-bit unchanged."""
    legacy = ParallelStrategy(
        vllm_world_size=8,
        vllm_worker_id=3,
        tp_size=8,
        pp_size=1,
        n_servers=1,
    )
    assert legacy.dcp_size == 1
    assert legacy.kv_world_size == 8
    assert legacy.kv_worker_id == 3


def test_positional_construction_still_works():
    """Existing positional callers (6 args) must not break on the new field."""
    strategy = ParallelStrategy(1, 0, 1, 1, 1)
    assert strategy.dcp_size == 1


def test_kv_rank_space_unaffected_by_dcp_field():
    """DCP shards a replicated group's objects on the server (its
    ``ShardReplicationDesc``); the kv rank space stays one rank per worker."""
    strategy = _strategy(tp_size=4, dcp_size=2, worker_id=2)
    assert strategy.kv_world_size == 4
    assert strategy.kv_worker_id == 2


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


def _import_connector_geometry_helpers():
    """Import helpers lazily because their module imports vLLM."""
    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import (
        get_dcp_decorated_model_name,
        get_group_tokens_per_block,
        get_vllm_scheduler_block_size,
        validate_mamba_step_alignment,
    )

    return (
        get_group_tokens_per_block,
        get_dcp_decorated_model_name,
        get_vllm_scheduler_block_size,
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
        get_vllm_scheduler_block_size,
        _,
    ) = _import_connector_geometry_helpers()
    config = _geometry_config()
    kv_config = _hybrid_kv_cache_config()

    assert get_group_tokens_per_block(config, kv_config) == [9216, 2304]
    assert get_vllm_scheduler_block_size(config, kv_config) == 9216


@requires_vllm
def test_scheduler_block_size_is_group_span_lcm():
    (
        get_group_tokens_per_block,
        _,
        get_vllm_scheduler_block_size,
        _,
    ) = _import_connector_geometry_helpers()
    config = _geometry_config(dcp_size=2)
    kv_config = _hybrid_kv_cache_config(
        attention_block_size=6,
        mamba_block_size=18,
    )

    assert get_group_tokens_per_block(config, kv_config) == [12, 18]
    assert get_vllm_scheduler_block_size(config, kv_config) == 36


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
def test_dcp_interleave_participates_in_cache_identity():
    _, get_dcp_decorated_model_name, _, _ = _import_connector_geometry_helpers()

    interleave4 = get_dcp_decorated_model_name(_geometry_config(interleave=4))
    interleave8 = get_dcp_decorated_model_name(_geometry_config(interleave=8))
    assert interleave4 != interleave8
    assert interleave4.endswith("##lmcache-dcp-layout-v1-d4-interleave4")


@requires_vllm
def test_trivial_interleave_preserves_legacy_cache_identity():
    _, get_dcp_decorated_model_name, _, _ = _import_connector_geometry_helpers()

    assert (
        get_dcp_decorated_model_name(_geometry_config(dcp_size=4, interleave=1))
        == "org/glm-5.3-flash"
    )
    assert (
        get_dcp_decorated_model_name(_geometry_config(dcp_size=1, interleave=4))
        == "org/glm-5.3-flash"
    )


@requires_vllm
@pytest.mark.parametrize("gated", [False, True])
def test_dcp_cache_identity_ignores_worker_mutated_interleave(gated):
    """Workers adjust the interleave, the scheduler does not; names must match."""
    _, get_dcp_decorated_model_name, _, _ = _import_connector_geometry_helpers()

    def config(interleave):
        cfg = _geometry_config(dcp_size=8, interleave=interleave, base_block_size=1536)

        def adjust(kv_cache_config):
            # ``gated`` mimics vLLM main, which aligns only for NixlConnector.
            if not gated:
                cfg.parallel_config.cp_kv_cache_interleave_size = min(
                    g.kv_cache_spec.block_size for g in kv_cache_config.kv_cache_groups
                )

        cfg.adjust_dcp_kv_cache_interleave_size = adjust
        return cfg

    kv_cache_config = _hybrid_kv_cache_config(1536, 1536)
    scheduler = config(1)
    worker = config(1 if gated else 1536)
    assert get_dcp_decorated_model_name(
        scheduler, kv_cache_config
    ) == get_dcp_decorated_model_name(worker, kv_cache_config)
    assert scheduler.parallel_config.cp_kv_cache_interleave_size == 1


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
    """PP + DCP is supported: each pipeline stage has its own TP ranks,
    sharded by DCP among them."""
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
def test_validate_accepts_multi_server_when_each_holds_a_full_shard_set():
    """Ranks split into contiguous per-server blocks; a block of a multiple of
    dcp_size ranks holds whole shard sets, so each server is independently
    servable."""
    _import_validate_dcp_support()(_config(dcp_size=2, tp_size=8), 2)
    _import_validate_dcp_support()(_config(dcp_size=4, tp_size=8), 2)


@requires_vllm
def test_validate_rejects_servers_too_small_to_hold_a_shard_set():
    """TP8 across 2 servers leaves 4 ranks each, which cannot hold 8 shards.
    Lookup takes the min across servers, so a partial server reports 0 hits."""
    with pytest.raises(ValueError, match="complete set of shards"):
        _import_validate_dcp_support()(_config(dcp_size=8, tp_size=8), 2)


@requires_vllm
def test_validate_rejects_servers_whose_ranks_split_a_shard_set():
    """TP6 across 2 servers leaves 3 ranks each: shard 0 twice and shard 1
    once, so a rank's shard is no longer its position among the server's TP
    ranks."""
    with pytest.raises(ValueError, match="complete set of shards"):
        _import_validate_dcp_support()(_config(dcp_size=2, tp_size=6), 2)


@requires_vllm
def test_validate_allows_multiple_servers_without_dcp():
    """The multi-server MLA path is unchanged when DCP is off."""
    _import_validate_dcp_support()(_config(dcp_size=1), 4)


@pytest.mark.parametrize("tp_size,dcp_size", [(2, 2), (8, 2), (8, 4), (8, 8)])
def test_dcp_rank_matches_vllm_group_construction(tp_size: int, dcp_size: int):
    """Reproduces vLLM's contiguous group construction and asserts the
    server keys a replicated group's object by ``tp_rank % dcp_size``, the
    shard vLLM gives that rank; fails if upstream goes strided."""
    # Third Party
    import torch

    # pcp = pp = dp = 1, mirroring parallel_state.initialize_model_parallel.
    all_ranks = torch.arange(tp_size).reshape(-1, 1, 1, 1, tp_size)
    groups = all_ranks.transpose(-1, -2).reshape(-1, dcp_size).tolist()
    vllm_dcp_rank = {rank: pos for group in groups for pos, rank in enumerate(group)}

    shards = ShardReplicationDesc(
        group_tp_replicas=(tp_size,), group_dcp_shards=(dcp_size,)
    )
    for rank in range(tp_size):
        key = IPCCacheServerKey(
            model_name="m",
            world_size=tp_size,
            worker_id=rank,
            token_bytes=pack_token_ids([1, 2, 3]),
            start=0,
            end=3,
            request_id="req",
        )
        [[obj_key]] = ipc_key_to_object_keys(key, [b"chunk"], [0], shards)
        shard = vllm_dcp_rank[rank]
        assert obj_key.kv_rank == ObjectKey.ComputeKVRank(
            world_size=tp_size,
            global_rank=shard,
            local_world_size=tp_size,
            local_rank=shard,
        ), rank


def test_uniform_type_wrapper_still_scales_under_dcp():
    """UniformTypeKVCacheSpecs derives from KVCacheSpec, not AttentionSpec;
    without unwrapping, wrapped attention groups skip DCP scaling."""
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


@dataclass
class CircularBufferSpec(AttentionSpec):
    """Scratch ring double: vLLM marks it non-prefix-cacheable."""

    @property
    def prefix_cacheable(self) -> bool:
        return False


@requires_vllm
def test_scratch_group_spans_zero_and_skips_alignment():
    (
        get_group_tokens_per_block,
        _,
        get_vllm_scheduler_block_size,
        _,
    ) = _import_connector_geometry_helpers()
    config = _geometry_config(dcp_size=1)
    kv_config = _hybrid_kv_cache_config(1600, 1600)
    kv_config.kv_cache_groups.append(
        SimpleNamespace(kv_cache_spec=CircularBufferSpec(block_size=8))
    )

    assert get_group_tokens_per_block(config, kv_config) == [1600, 1600, 0]
    assert get_vllm_scheduler_block_size(config, kv_config) == 1600
