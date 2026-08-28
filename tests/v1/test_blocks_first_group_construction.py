# SPDX-License-Identifier: Apache-2.0
"""Multi-group blocks-first construction, DSV4-shaped.

Two groups with different geometry interleaved inside one blocks-first pool
(vLLM HMA), registered as per-layer views: the full normalize -> manager ->
group pointers pipeline must group by shape and read each group's block
step from its own views' stride(0).
"""

# Third Party
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import (
    get_group_data_ptrs,
    normalize_and_discover_per_layer_formats,
    resolve_block_stride_and_log_layout,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
import lmcache.lmcache_native as lmcache_native

NB = 4
# Group A: 2 layers, [NH=2, BS=8, CS=16]; Group B: 3 layers, [NH=1, BS=8, CS=32]
A_LAYERS, A_INNER = 2, (2, 8, 16)
B_LAYERS, B_INNER = 3, (1, 8, 32)
A_CHUNK = 2 * 8 * 16
B_CHUNK = 1 * 8 * 32
BLOCK_ELEMS = A_LAYERS * A_CHUNK + B_LAYERS * B_CHUNK


def build_pool() -> list[torch.Tensor]:
    """One buffer; each block holds [A0, A1, B0, B1, B2] back to back."""
    buf = torch.arange(NB * BLOCK_ELEMS, dtype=torch.float32)
    views = []
    for layer in range(A_LAYERS):
        views.append(
            buf.as_strided(
                (NB, *A_INNER),
                (BLOCK_ELEMS, 8 * 16, 16, 1),
                storage_offset=layer * A_CHUNK,
            )
        )
    b_base = A_LAYERS * A_CHUNK
    for layer in range(B_LAYERS):
        views.append(
            buf.as_strided(
                (NB, *B_INNER),
                (BLOCK_ELEMS, 8 * 32, 32, 1),
                storage_offset=b_base + layer * B_CHUNK,
            )
        )
    return views


def test_multi_group_blocks_first_constructs():
    views = build_pool()
    groups = [[0, 1], [2, 3, 4]]
    normalized, formats = normalize_and_discover_per_layer_formats(
        views, groups, EngineType.VLLM, {"kv_layout": "BLHNC"}
    )
    assert all(f == lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_CS for f in formats)
    for n, v in zip(normalized, views, strict=True):
        assert n.data_ptr() == v.data_ptr()
        assert n.shape == v.shape and n.stride() == v.stride()

    manager = KVLayerGroupsManager(
        normalized,
        engine_kv_formats=formats,
        engine_group_infos=(),
        separate_object_groups=False,
    )
    assert len(manager.kernel_groups) == 2

    for group_idx, group in enumerate(manager.kernel_groups):
        indices = list(group.layer_indices)
        fmt = formats[indices[0]]
        ptrs = get_group_data_ptrs(normalized, fmt, indices)
        assert ptrs == [normalized[i].data_ptr() for i in indices]
        # Both groups share the pool, so both see the interleaved block step.
        stride = resolve_block_stride_and_log_layout(
            normalized, fmt, layer_idx=indices[0], group_idx=group_idx
        )
        assert stride == BLOCK_ELEMS
