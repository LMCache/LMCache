# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.compute.blend.blender import LMCBlender
from lmcache.v1.compute.blend.metadata import (
    LMCBlendCommonMetadata,
    LMCBlendMetadata,
)


def _identity_rotary(
    positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stand-in for vllm's rotary_emb. Real RoPE preserves shapes; that's all
    process_qkv needs in this test."""
    assert positions.shape[0] == q.shape[0] == k.shape[0]
    return q, k


def _build_fake_blender(
    *,
    num_layers: int,
    hidden_dim: int,
    check_layer: int,
    recomp_ratio: float,
    kv_per_layer: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> LMCBlender:
    """Construct an LMCBlender with all vllm-side attrs mocked.

    Skips ``__init__`` (which would walk ``vllm_model.model.layers`` etc.) and
    sets the attributes ``process_qkv`` actually reads.
    """
    blender = LMCBlender.__new__(LMCBlender)
    blender.num_layers = num_layers

    layers = []
    for _ in range(num_layers):
        layer = MagicMock()
        layer.self_attn.rotary_emb = _identity_rotary
        layers.append(layer)
    fake_vllm_model = MagicMock()
    fake_vllm_model.model.layers = layers
    blender.layerwise_model = MagicMock()
    blender.layerwise_model.vllm_model = fake_vllm_model

    blender.gpu_connector = MagicMock()
    blender.gpu_connector.get_kv = lambda lid: kv_per_layer[lid]

    blender.common_metadata = LMCBlendCommonMetadata(
        check_layers=[check_layer],
        recomp_ratios=[recomp_ratio],
        thresholds=None,
    )
    blender.metadata = LMCBlendMetadata(
        imp_indices=None, attn_mask=None, positions=None
    )
    return blender


def _make_kv(n: int, hidden_dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, hidden_dim, generator=g)


@pytest.mark.parametrize(
    "n_compute,n_retrieved,recomp_ratio",
    [
        (1292, 801, 0.15),  # production crash signature, seen on Qwen3-8B
        (919, 459, 0.15),  # H100 morning data point — exactly 2:1 ratio
        (6519, 6512, 0.15),  # #938 signature: small offset from separator counting
    ],
)
def test_process_qkv_aligns_k_with_shorter_old_k(
    n_compute: int, n_retrieved: int, recomp_ratio: float
) -> None:
    """When ``retrieve_layer`` skipped masked-prefix chunks, ``old_k`` covers
    fewer positions than ``compute_layer``'s ``k``. The trailing
    ``old_k.shape[0]`` rows of ``k`` correspond to the same token positions as
    ``old_k``.

    Without the alignment fix, ``diff_k = (k - old_k)**2`` raises
    ``RuntimeError: The size of tensor a (X) must match the size of tensor b
    (Y) at non-singleton dimension 0`` and the engine dies.

    Reproduces / regresses the family of bugs reported in #1875, #854, #938.
    """
    hidden_dim = 4096
    attn_md = MagicMock()
    attn_md.update_from_top_indices = MagicMock()

    blender = _build_fake_blender(
        num_layers=3,
        hidden_dim=hidden_dim,
        check_layer=1,
        recomp_ratio=recomp_ratio,
        kv_per_layer={
            i: (
                _make_kv(n_retrieved, hidden_dim, seed=10 + 2 * i),
                _make_kv(n_retrieved, hidden_dim, seed=11 + 2 * i),
            )
            for i in range(3)
        },
    )

    q = _make_kv(n_compute, hidden_dim, seed=20)
    k = _make_kv(n_compute, hidden_dim, seed=21)
    v = _make_kv(n_compute, hidden_dim, seed=22)
    residual = _make_kv(n_compute, hidden_dim, seed=23)

    # Layer 0 (not check_layer): passthrough — k/v/q must retain their full
    # ``n_compute`` shape. Slicing here would corrupt ``attn_metadata`` (sized
    # for ``len(input_ids)`` by ``init_attn_metadata``) and trigger CUDA OOB
    # at the layer-0 attention forward.
    q0, k0, v0, r0, ao0, _ = blender.process_qkv(
        q, k, v, residual, layer_id=0, attn_output=None, attn_metadata=attn_md
    )
    assert k0.shape == (n_compute, hidden_dim)

    # Layer 1 (check_layer): aligns ``k_for_diff`` to ``old_k`` shape, picks
    # topk in ``old_k``-local space, restricts ``q/k/v/residual`` via global
    # indices (= local + offset), updates ``attn_metadata`` with global
    # indices so subsequent layers see consistent positions.
    q1, k1, v1, r1, ao1, _ = blender.process_qkv(
        q0, k0, v0, r0, layer_id=1, attn_output=ao0, attn_metadata=attn_md
    )
    expected_topk = max(int(n_retrieved * recomp_ratio), 1)
    assert blender.metadata.imp_indices is not None
    assert blender.metadata.imp_indices.shape == (expected_topk,)
    # imp_indices stay in old_k-local space [0, n_retrieved) for the writeback.
    assert blender.metadata.imp_indices.max().item() < n_retrieved
    # attn_metadata.update_from_top_indices was called with GLOBAL indices.
    last_call_args = attn_md.update_from_top_indices.call_args_list[-1].args
    offset = n_compute - n_retrieved
    assert last_call_args[0].max().item() < n_compute
    assert last_call_args[0].min().item() >= offset
    # k1 is returned via the writeback branch: ``old_k`` with selective updates.
    assert k1.shape == (n_retrieved, hidden_dim)

    # Layer 2 (post-check, writeback path): q/k/v shapes are the topk subset
    # because layer 1's attention forward produced ``hidden_states`` of size
    # ``topk``. The writeback writes them into ``old_k[imp_indices]``.
    topk = blender.metadata.imp_indices.shape[0]
    q_l2 = _make_kv(topk, hidden_dim, seed=30)
    k_l2 = _make_kv(topk, hidden_dim, seed=31)
    v_l2 = _make_kv(topk, hidden_dim, seed=32)
    r_l2 = _make_kv(topk, hidden_dim, seed=33)
    ao_l2 = _make_kv(topk, hidden_dim, seed=34)
    _, k2, _, _, _, _ = blender.process_qkv(
        q_l2, k_l2, v_l2, r_l2, layer_id=2, attn_output=ao_l2, attn_metadata=attn_md
    )
    assert k2.shape == (n_retrieved, hidden_dim)


def test_process_qkv_unchanged_when_dims_already_match() -> None:
    """When ``k.shape[0] == old_k.shape[0]`` (the common case), the alignment
    slice is a no-op (offset == 0) and behavior matches pre-fix lmcache."""
    n = 1700
    hidden_dim = 4096
    attn_md = MagicMock()
    attn_md.update_from_top_indices = MagicMock()

    blender = _build_fake_blender(
        num_layers=2,
        hidden_dim=hidden_dim,
        check_layer=1,
        recomp_ratio=0.15,
        kv_per_layer={
            i: (
                _make_kv(n, hidden_dim, seed=40 + 2 * i),
                _make_kv(n, hidden_dim, seed=41 + 2 * i),
            )
            for i in range(2)
        },
    )

    q = _make_kv(n, hidden_dim, seed=50)
    k = _make_kv(n, hidden_dim, seed=51)
    v = _make_kv(n, hidden_dim, seed=52)
    residual = _make_kv(n, hidden_dim, seed=53)

    _, k0, _, _, ao0, _ = blender.process_qkv(
        q, k, v, residual, layer_id=0, attn_output=None, attn_metadata=attn_md
    )
    assert k0.shape == (n, hidden_dim)

    _, k1, _, _, _, _ = blender.process_qkv(
        q, k0, v, residual, layer_id=1, attn_output=ao0, attn_metadata=attn_md
    )
    assert blender.metadata.imp_indices.shape == (max(int(n * 0.15), 1),)
    assert k1.shape == (n, hidden_dim)
    # offset == 0 → update_from_top_indices was called with local == global.
    last_call_args = attn_md.update_from_top_indices.call_args_list[-1].args
    assert last_call_args[0].max().item() < n
