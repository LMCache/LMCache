# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

from lmcache.v1.compute.blend.blender import LMCBlender


def _make_blender(num_layers: int = 1):
    blender = LMCBlender.__new__(LMCBlender)
    blender.num_layers = num_layers
    calls: dict[str, object] = {"count": 0, "last": None}

    def fake_blend_layer(tokens, mask=None, **kwargs):
        calls["count"] += 1
        calls["last"] = (tokens, mask, kwargs)
        for _ in range(blender.num_layers + 2):
            yield None

    blender.blend_layer = fake_blend_layer
    return blender, calls


def test_blend_all_true_mask_calls_blend_layer():
    blender, calls = _make_blender()
    tokens = torch.arange(4, dtype=torch.int64)
    mask = torch.ones(4, dtype=torch.bool)

    blender.blend(tokens, mask)

    assert calls["count"] == 1


def test_blend_all_false_mask_skips():
    blender, calls = _make_blender()
    tokens = torch.arange(4, dtype=torch.int64)
    mask = torch.zeros(4, dtype=torch.bool)

    blender.blend(tokens, mask)

    assert calls["count"] == 0


def test_blend_empty_tokens_skips():
    blender, calls = _make_blender()
    tokens = torch.tensor([], dtype=torch.int64)
    mask = torch.tensor([], dtype=torch.bool)

    blender.blend(tokens, mask)

    assert calls["count"] == 0


def test_blend_mask_length_mismatch_raises():
    blender, _ = _make_blender()
    tokens = torch.arange(4, dtype=torch.int64)
    mask = torch.ones(3, dtype=torch.bool)

    with pytest.raises(ValueError, match="Mask length"):
        blender.blend(tokens, mask)
