# SPDX-License-Identifier: Apache-2.0
"""Tests for ``LMCacheMetadata.get_shapes``.

Pins the shape contract that memory objects are allocated with, most
importantly that fused-packed kernel groups report the true split-KV
KV_2LTD shape and agree with the pre-registration ``kv_shape`` fallback:
``cache_engine.store`` allocates before the connector's lazy layout
discovery has registered the groups manager, so a first-store shape flip
would hand the GPU connector a buffer shaped differently from its
group-derived tmp buffers (issue #4463's V3 crash).
"""

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.metadata import LMCacheMetadata
import lmcache.lmcache_native as lmc_ops

_NL, _NB, _BS, _NH, _HS = 3, 8, 4, 4, 8
_CHUNK = 16


def _metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(_NL, 2, _CHUNK, _NH, _HS),
        chunk_size=_CHUNK,
    )


def test_fused_group_shapes_are_split_kv():
    """Fused groups report [2, L, T, NH*HS], matching the kv_shape fallback
    used before registration — no first-store discontinuity."""
    md = _metadata()
    before = md.get_shapes(_CHUNK)
    assert before == [torch.Size([2, _NL, _CHUNK, _NH * _HS])]

    caches = [
        torch.randn(_NB, _BS, _NH, 2 * _HS, dtype=torch.float16) for _ in range(_NL)
    ]
    md.kv_layer_groups_manager = KVLayerGroupsManager(
        caches,
        engine_kv_formats=[lmc_ops.EngineKVFormat.NL_X_NB_BS_NH_CS] * _NL,
    )
    assert md.get_shapes(_CHUNK) == before


def test_mla_group_shapes_stay_single_plane():
    """MLA groups are kv_size 1 too but must keep their single-plane shape —
    the fused translation is keyed on the format, not on kv_size."""
    md = _metadata()
    caches = [torch.randn(_NB, _BS, _HS, dtype=torch.float16) for _ in range(_NL)]
    md.kv_layer_groups_manager = KVLayerGroupsManager(
        caches,
        engine_kv_formats=[lmc_ops.EngineKVFormat.NL_X_NB_BS_HS] * _NL,
    )
    assert md.get_shapes(_CHUNK) == [torch.Size([1, _NL, _CHUNK, _HS])]


def test_fused_group_odd_hidden_rejected():
    md = _metadata()
    fake_group = SimpleNamespace(
        shape_desc=SimpleNamespace(kv_size=1, nh=3, hs=9),
        hidden_dim_size=27,
        num_layers=_NL,
        engine_kv_format=lmc_ops.EngineKVFormat.NL_X_NB_BS_NH_CS,
    )
    md.kv_layer_groups_manager = SimpleNamespace(kernel_groups=[fake_group])
    with pytest.raises(ValueError, match="odd hidden dim"):
        md.get_shapes(_CHUNK)
