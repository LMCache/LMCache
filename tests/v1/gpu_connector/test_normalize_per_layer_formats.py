# SPDX-License-Identifier: Apache-2.0
"""Tests for ``normalize_and_discover_per_layer_formats``.

Uniform models return the whole normalized structure; mixed models report each
engine group's own per-layer format.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import (
    get_shape_and_dtype,
    normalize_and_discover_per_layer_formats,
)
import lmcache.lmcache_native as lmcache_native

NB, NL, BS, NH, HS = 7, 5, 3, 2, 4
DT = torch.float16
F = lmcache_native.EngineKVFormat


def _t(*shape: int) -> torch.Tensor:
    return torch.zeros(shape, dtype=DT)


def test_vllm_layer_list_uniform():
    # Per-layer-list format, every layer the same: whole list back, format per layer.
    kv = [_t(2, NB, BS, NH, HS) for _ in range(NL)]
    normalized, formats = normalize_and_discover_per_layer_formats(
        kv, (), EngineType.VLLM, {"kv_layout": "HND"}
    )
    assert len(normalized) == NL
    assert formats == [F.NL_X_TWO_NB_NH_BS_HS] * NL


def test_vllm_heterogeneous_groups():
    # Mixed-format model: a full K/V group beside a key-only MLA index group.
    kv = [_t(2, NB, BS, NH, HS) for _ in range(3)] + [_t(NB, BS, HS) for _ in range(2)]
    normalized, formats = normalize_and_discover_per_layer_formats(
        kv,
        [[0, 1, 2], [3, 4]],
        EngineType.VLLM,
        {"kv_layout": "HND"},
    )
    assert len(normalized) == 5
    assert formats[:3] == [F.NL_X_TWO_NB_NH_BS_HS] * 3
    assert formats[3:] == [F.NL_X_NB_BS_HS] * 2


def test_vllm_mixed_rank4_fused_groups():
    # Regression for #4263: blocks-first fused rank-4 groups whose trailing
    # dims differ (hybrid sliding + full attention under the vLLM 0.26 V2
    # model runner). v0.5.2's whole-set detection applied the first tensor's
    # fused_dim to every layer and died in reshape; mixed-shape sets must be
    # detected per shape bucket instead, each layer keeping its own raw
    # [NB, BS, NH, CS] tensor and reporting the fused CS format.
    kv = [_t(NB, BS, NH, 2 * HS) for _ in range(3)] + [
        _t(NB, BS, NH, 4 * HS) for _ in range(2)
    ]
    normalized, formats = normalize_and_discover_per_layer_formats(
        kv,
        [[0, 1, 2], [3, 4]],
        EngineType.VLLM,
        {"kv_layout": "HND"},
    )
    assert len(normalized) == 5
    assert tuple(normalized[0].shape) == (NB, BS, NH, 2 * HS)
    assert tuple(normalized[3].shape) == (NB, BS, NH, 4 * HS)
    assert formats == [F.NL_X_NB_NH_BS_CS] * 5


def _mla_plane_tuple_entry() -> tuple[torch.Tensor, torch.Tensor]:
    latent = torch.zeros(NB, BS, 1, 8, dtype=torch.int8)
    rope = torch.zeros(NB, BS, 1, 2, dtype=torch.float16)
    return latent, rope


@pytest.mark.parametrize(
    ("kv_caches", "layer_indices", "expected"),
    [
        pytest.param(
            [_t(NB, BS, NH, HS)],
            None,
            [((NB, BS, NH, HS), DT)],
            id="bare_tensor",
        ),
        pytest.param(
            [_mla_plane_tuple_entry()],
            None,
            [
                (
                    ((NB, BS, 1, 8), torch.int8),
                    ((NB, BS, 1, 2), torch.float16),
                )
            ],
            id="mla_plane_tuple_mixed_dtype",
        ),
        pytest.param(
            [[_t(2, 3), _t(4, 5)]],
            None,
            [(((2, 3), DT), ((4, 5), DT))],
            id="nested_sequence",
        ),
        pytest.param(
            [_t(2), _t(3), _t(4)],
            [2, 0],
            [((4,), DT), ((2,), DT)],
            id="layer_indices_subset",
        ),
        pytest.param(
            [_t(2), _t(3), _t(4)],
            None,
            [((2,), DT), ((3,), DT), ((4,), DT)],
            id="all_layers_in_order",
        ),
    ],
)
def test_get_shape_and_dtype(
    kv_caches: list[object],
    layer_indices: list[int] | None,
    expected: list[object],
) -> None:
    """Walker keys for normalize_and_discover_per_layer_formats shape buckets."""
    if layer_indices is None:
        assert get_shape_and_dtype(kv_caches) == expected
    else:
        assert get_shape_and_dtype(kv_caches, layer_indices) == expected


def test_get_shape_and_dtype_keys_differ_by_dtype() -> None:
    fp16 = _t(NB, BS, HS)
    bf16 = torch.zeros(NB, BS, HS, dtype=torch.bfloat16)
    keys = get_shape_and_dtype([fp16, fp16.clone(), bf16])
    assert keys[0] == keys[1] and keys[0] != keys[2]
