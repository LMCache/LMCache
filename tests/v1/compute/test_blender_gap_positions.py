# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.compute.blend.blender import LMCBlender
from lmcache.v1.compute.blend.metadata import LMCBlendCommonMetadata, LMCBlendMetadata

NUM_TOKENS = 8
HIDDEN = 4
GAPS = torch.tensor([3, 4])


def _make_blender(old_k, old_v, gaps, check_layers=(1,), ratio=0.125):
    """LMCBlender with a fake loaded buffer and identity RoPE (no vLLM model)."""
    blender = LMCBlender.__new__(LMCBlender)
    blender.gpu_connector = SimpleNamespace(
        get_kv=lambda layer_id: (old_k, old_v),
        current_gap_positions=gaps,
    )
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(rotary_emb=lambda positions, q, k: (q, k))
    )
    blender.layerwise_model = SimpleNamespace(
        vllm_model=SimpleNamespace(model=SimpleNamespace(layers=[layer, layer]))
    )
    blender.common_metadata = LMCBlendCommonMetadata(
        check_layers=list(check_layers), recomp_ratios=[ratio], thresholds=None
    )
    blender.metadata = LMCBlendMetadata(
        imp_indices=None, attn_mask=None, positions=None
    )
    return blender


def _loaded_buffer(gaps):
    old_k = torch.randn(NUM_TOKENS, HIDDEN)
    old_v = torch.randn(NUM_TOKENS, HIDDEN)
    if gaps is not None:
        # The layerwise loader zero-fills positions not covered by any segment.
        old_k[gaps] = 0.0
        old_v[gaps] = 0.0
    return old_k, old_v


def _call(blender, layer_id, k, v):
    q = torch.randn(NUM_TOKENS, HIDDEN)
    residual = torch.randn(NUM_TOKENS, HIDDEN)
    return blender.process_qkv(q, k, v, residual, layer_id, None, MagicMock())


def test_gap_rows_recomputed_before_check_layer():
    old_k, old_v = _loaded_buffer(GAPS)
    loaded_k = old_k.clone()
    blender = _make_blender(old_k, old_v, GAPS)
    k = torch.randn(NUM_TOKENS, HIDDEN)
    v = torch.randn(NUM_TOKENS, HIDDEN)

    _call(blender, 0, k, v)

    assert torch.equal(old_k[GAPS], k[GAPS])
    assert torch.equal(old_v[GAPS], v[GAPS])
    keep = torch.ones(NUM_TOKENS, dtype=torch.bool)
    keep[GAPS] = False
    assert torch.equal(old_k[keep], loaded_k[keep])


def test_gap_positions_join_recompute_set_at_check_layer():
    old_k, old_v = _loaded_buffer(GAPS)
    blender = _make_blender(old_k, old_v, GAPS)
    k = old_k.clone()
    k[GAPS] = torch.randn(len(GAPS), HIDDEN)
    k[6] += 100.0  # the single top-k pick at ratio 1/8
    v = torch.randn(NUM_TOKENS, HIDDEN)

    _, k_out, v_out, _, attn_output, _ = _call(blender, 1, k, v)

    assert blender.metadata.imp_indices.tolist() == [3, 4, 6]
    assert attn_output.shape[0] == 3
    assert torch.equal(k_out[[3, 4, 6]], k[[3, 4, 6]])
    assert torch.equal(v_out[[3, 4, 6]], v[[3, 4, 6]])


@pytest.mark.parametrize("gaps", [None, torch.tensor([], dtype=torch.long)])
def test_without_gaps_selection_is_unchanged(gaps):
    old_k, old_v = _loaded_buffer(None)
    blender = _make_blender(old_k, old_v, gaps)
    k = old_k.clone()
    k[5] += 100.0
    v = torch.randn(NUM_TOKENS, HIDDEN)

    _call(blender, 1, k, v)

    assert blender.metadata.imp_indices.tolist() == [5]
