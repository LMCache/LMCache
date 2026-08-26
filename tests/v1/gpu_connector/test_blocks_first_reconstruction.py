# SPDX-License-Identifier: Apache-2.0
"""Blocks-first (BLHNC/BLNHC) declaration-driven reconstruction tests.

Builds per-layer strided views into one buffer exactly the way vLLM
registers them under a blocks-first layout, and checks that detection
rebuilds the single cross-layer tensor -- and that every declaration /
allocation drift fails loudly instead of misclassifying.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format import detect_format
import lmcache.lmcache_native as lmcache_native

NB, NL, NH, BS, CS = 6, 3, 2, 4, 8


def make_views(order: str) -> tuple[torch.Tensor, list[torch.Tensor]]:
    inner = (NH, BS, CS) if order == "BLHNC" else (BS, NH, CS)
    buf = torch.arange(NB * NL * NH * BS * CS, dtype=torch.float32)
    buf = buf.reshape(NB, NL, *inner)
    return buf, [buf[:, layer] for layer in range(NL)]


@pytest.mark.parametrize(
    "order,expected",
    [
        ("BLHNC", lmcache_native.EngineKVFormat.NB_NL_NH_BS_CS),
        ("BLNHC", lmcache_native.EngineKVFormat.NB_NL_BS_NH_CS),
    ],
)
def test_reconstructs_cross_layer_tensor(order, expected):
    buf, views = make_views(order)
    fmt, kv = detect_format(views, EngineType.VLLM, {"kv_layout": order})
    assert fmt == expected
    assert isinstance(kv, torch.Tensor)
    assert kv.shape == buf.shape
    assert kv.data_ptr() == buf.data_ptr()
    assert torch.equal(kv, buf)


def test_view_order_does_not_matter():
    buf, views = make_views("BLHNC")
    fmt, kv = detect_format(
        list(reversed(views)), EngineType.VLLM, {"kv_layout": "BLHNC"}
    )
    assert torch.equal(kv, buf)


def test_foreign_storage_is_drift():
    _, views = make_views("BLHNC")
    views[1] = views[1].clone()
    with pytest.raises(ValueError, match="disagree"):
        detect_format(views, EngineType.VLLM, {"kv_layout": "BLHNC"})


def test_compact_per_layer_views_are_drift():
    views = [torch.zeros(NB, NH, BS, CS) for _ in range(NL)]
    with pytest.raises(ValueError, match="disagree"):
        detect_format(views, EngineType.VLLM, {"kv_layout": "BLHNC"})


def test_bare_tensor_is_rejected():
    buf, _ = make_views("BLHNC")
    with pytest.raises(ValueError, match="rank-4"):
        detect_format(buf, EngineType.VLLM, {"kv_layout": "BLHNC"})


def test_interleaved_groups_reconstruct():
    """Two groups woven inside each block: this group's layer step is
    2 * chunk (the other group's layer sits in between)."""
    inner = (NH, BS, CS)
    buf = torch.arange(NB * 2 * NL * NH * BS * CS, dtype=torch.float32)
    buf = buf.reshape(NB, 2 * NL, *inner)
    views = [buf[:, 2 * layer] for layer in range(NL)]  # even slots = ours
    fmt, kv = detect_format(views, EngineType.VLLM, {"kv_layout": "BLHNC"})
    assert fmt == lmcache_native.EngineKVFormat.NB_NL_NH_BS_CS
    assert kv.shape == (NB, NL, *inner)
    assert kv.stride(1) == 2 * NH * BS * CS
    assert torch.equal(kv, buf[:, ::2])
