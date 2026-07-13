# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the per-group block-ID grain derivation.

Regression context: under uneven DCP (vLLM fork, --rank-tp-ratio) the
scheduler hands the connector VIRTUAL block IDs for token-split
(attention) groups -- one ID spans ``spec.block_size *
cp_token_split_factor`` global tokens -- while Mamba groups keep their
raw (align-solver-inflated) ``block_size``. LMCache previously used the
raw spec block_size everywhere, which mis-sliced block IDs on the
scheduler side and mis-computed blocks-per-chunk on the worker side:
stores wrote mis-keyed chunk data and retrieves crashed the engine core
with "block_ids length (2) must be at least len(chunks) (2) *
blocks_per_chunk (4)".
"""

# Standard
from types import SimpleNamespace as NS

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.kv_cache_groups import (
    _cp_token_split_factor,
    _is_token_split_spec,
    resolve_group_token_spans,
)
from lmcache.v1.multiprocess.group_view import (
    EngineGroupInfo,
    slice_block_ids_per_group,
)
from lmcache.v1.multiprocess.transfer_context import base as tc_base


class MambaSpec:  # noqa: D401 - duck-typed stand-in for vLLM's MambaSpec
    def __init__(self, block_size: int) -> None:
        self.block_size = block_size


class FullAttentionSpec:
    def __init__(self, block_size: int) -> None:
        self.block_size = block_size


def _kv_cache_config(specs):
    return NS(kv_cache_groups=[NS(kv_cache_spec=s) for s in specs])


def _vllm_config(dcp=1, pcp=1, ratios=None, block_size=16):
    return NS(
        parallel_config=NS(
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
            rank_tp_ratio=ratios,
        ),
        cache_config=NS(block_size=block_size),
    )


# ─── resolve_group_token_spans ───────────────────────────────────────────────


def test_spans_without_cp_are_raw_block_sizes():
    kc = _kv_cache_config([FullAttentionSpec(16), MambaSpec(2048)])
    assert resolve_group_token_spans(kc, _vllm_config()) == [16, 2048]


def test_spans_under_uneven_dcp_scale_token_split_groups_only(monkeypatch):
    # Force the config-derived fallback so the test does not depend on
    # process-global CP vectors installed by a running vLLM engine.
    monkeypatch.setattr(
        "lmcache.integration.vllm.kv_cache_groups._cp_token_split_factor",
        lambda vc: 32,
    )
    kc = _kv_cache_config([FullAttentionSpec(16), MambaSpec(2048)])
    spans = resolve_group_token_spans(kc, _vllm_config(dcp=3, ratios=[16, 8, 8]))
    # Attention: virtual scheduler blocks (16 * sum(16,8,8) = 512).
    # Mamba: full per-sequence state on every rank -> raw block size.
    assert spans == [512, 2048]


def test_cp_split_factor_fallback_uneven_ratios():
    # No fork helper needed: reads rank_tp_ratio from the parallel config.
    vc = _vllm_config(dcp=3, ratios=[16, 8, 8])
    try:
        # If the local vLLM provides the helper it may consult installed
        # process-global vectors; only assert the fallback math when the
        # helper is unavailable.
        # Third Party
        from vllm.distributed.utils import cp_token_split_factor  # noqa: F401

        pytest.skip("local vLLM provides cp_token_split_factor")
    except ImportError:
        pass
    assert _cp_token_split_factor(vc) == 32


def test_cp_split_factor_off_and_even():
    assert _cp_token_split_factor(_vllm_config()) == 1
    assert _cp_token_split_factor(_vllm_config(dcp=1, pcp=1, ratios=[1, 1])) == 1


def test_single_group_fallback_is_cp_adjusted(monkeypatch):
    monkeypatch.setattr(
        "lmcache.integration.vllm.kv_cache_groups._cp_token_split_factor",
        lambda vc: 4,
    )
    assert resolve_group_token_spans(None, _vllm_config(block_size=16)) == [64]


def test_is_token_split_spec_duck_typing():
    assert _is_token_split_spec(FullAttentionSpec(16))
    assert not _is_token_split_spec(MambaSpec(2048))


# ─── slice_block_ids_per_group validation ────────────────────────────────────


def test_slice_block_ids_heterogeneous_groups():
    allocated = {0: list(range(8)), 1: [100, 101]}
    sliced = slice_block_ids_per_group(
        allocated,
        group_tokens_per_block=[512, 2048],
        start_token_idx=0,
        end_token_idx=4096,
    )
    assert sliced == [list(range(8)), [100, 101]]


def test_slice_block_ids_rejects_short_allocation():
    # 4096 tokens at tokens_per_block=512 need 8 IDs; only 2 allocated.
    # This is the signature of a grain mismatch (e.g. raw spec block
    # size used where the scheduler reports virtual blocks) and must
    # fail loudly instead of returning a silently short list.
    allocated = {0: [0, 1]}
    with pytest.raises(ValueError, match="block-ID grain"):
        slice_block_ids_per_group(
            allocated,
            group_tokens_per_block=[512],
            start_token_idx=0,
            end_token_idx=4096,
        )


def test_slice_block_ids_rejects_misaligned_range():
    with pytest.raises(ValueError, match="align"):
        slice_block_ids_per_group(
            {0: list(range(8))},
            group_tokens_per_block=[512],
            start_token_idx=100,
            end_token_idx=4096,
        )


# ─── worker-side blocks-per-chunk from EngineGroupInfo ───────────────────────


def _group_info(gid: int, tokens_per_block: int, layer: int) -> EngineGroupInfo:
    return EngineGroupInfo(
        engine_group_id=gid,
        layer_indices=(layer,),
        tokens_per_block=tokens_per_block,
    )


def test_scatter_multi_group_uses_per_group_grain(monkeypatch):
    """Attention (span 512) gets 4 blocks/chunk, Mamba (span 2048) gets 1."""
    seen: list[int] = []

    def fake_scatter(
        kv_caches,
        block_ids,
        chunks,
        blocks_per_chunk,
        skip_first_n_tokens=0,
        layout_hints=None,
        engine_kv_format=None,
    ):
        seen.append(blocks_per_chunk)

    monkeypatch.setattr(tc_base, "scatter_cpu_to_paged_kv", fake_scatter)
    kv_caches = {"a": torch.zeros(1), "b": torch.zeros(1)}
    infos = [_group_info(0, 512, 0), _group_info(1, 2048, 1)]
    tc_base.scatter_cpu_multi_group_to_paged_kv(
        kv_caches,
        block_ids=[list(range(8)), [100, 101]],
        group_chunks=[[torch.zeros(1)] * 2, [torch.zeros(1)] * 2],
        engine_group_infos=infos,
        lmcache_tokens_per_chunk=2048,
    )
    assert seen == [4, 1]


def test_gather_multi_group_uses_per_group_grain(monkeypatch):
    seen: list[int] = []

    def fake_gather(
        kv_caches,
        block_ids,
        blocks_per_chunk,
        layout_hints=None,
        engine_kv_format=None,
        out=None,
        chunk_indices=None,
        pinned_pool=None,
    ):
        seen.append(blocks_per_chunk)
        return [torch.zeros(1)]

    monkeypatch.setattr(tc_base, "gather_paged_kv_to_cpu", fake_gather)
    kv_caches = {"a": torch.zeros(1), "b": torch.zeros(1)}
    infos = [_group_info(0, 512, 0), _group_info(1, 2048, 1)]
    out = tc_base.gather_paged_kv_multi_group_to_cpu(
        kv_caches,
        block_ids=[list(range(8)), [100, 101]],
        engine_group_infos=infos,
        lmcache_tokens_per_chunk=2048,
    )
    assert seen == [4, 1]
    assert len(out) == 2


def test_missing_grain_falls_back_with_warning(monkeypatch, caplog):
    """tokens_per_block=0 -> shape-guessed grain plus a loud warning."""
    monkeypatch.setattr(
        tc_base,
        "compute_kv_layout",
        lambda kv, layout_hints=None: (512, 1, 8, "bfloat16", None),
    )
    tc_base._GRAIN_GUESS_WARNED.clear()
    seen: list[int] = []
    monkeypatch.setattr(
        tc_base,
        "scatter_cpu_to_paged_kv",
        lambda *a, **k: seen.append(a[3]),
    )
    kv_caches = {"a": torch.zeros(1)}
    infos = [_group_info(0, 0, 0)]  # engine did not report the grain
    with caplog.at_level("WARNING"):
        tc_base.scatter_cpu_multi_group_to_paged_kv(
            kv_caches,
            block_ids=[list(range(8))],
            group_chunks=[[torch.zeros(1)] * 2],
            engine_group_infos=infos,
            lmcache_tokens_per_chunk=2048,
        )
    assert seen == [4]
    assert any("guessing" in r.message for r in caplog.records)
