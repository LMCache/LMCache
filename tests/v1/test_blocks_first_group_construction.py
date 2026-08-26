# SPDX-License-Identifier: Apache-2.0
"""Multi-group blocks-first construction, DSV4-shaped.

Two groups with different geometry interleaved inside one blocks-first pool
(vLLM HMA), registered as per-layer views: the full normalize -> manager ->
group pointers pipeline must build without treating a reconstructed
cross-layer tensor as a per-layer list.
"""

# Third Party
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import (
    get_group_data_ptrs,
    normalize_and_discover_per_layer_formats,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
import lmcache.lmcache_native as lmcache_native

NB = 4
# Group A: 2 layers, [NH=2, BS=8, CS=16]; Group B: 3 layers, [NH=1, BS=8, CS=32]
A_LAYERS, A_INNER = 2, (2, 8, 16)
B_LAYERS, B_INNER = 3, (1, 8, 32)
A_CHUNK = 2 * 8 * 16
B_CHUNK = 1 * 8 * 32


def build_pool() -> list[torch.Tensor]:
    """One buffer; each block holds [A0, A1, B0, B1, B2] back to back."""
    block_elems = A_LAYERS * A_CHUNK + B_LAYERS * B_CHUNK
    buf = torch.arange(NB * block_elems, dtype=torch.float32)
    views = []
    for layer in range(A_LAYERS):
        views.append(
            buf.as_strided(
                (NB, *A_INNER),
                (block_elems, 8 * 16, 16, 1),
                storage_offset=layer * A_CHUNK,
            )
        )
    b_base = A_LAYERS * A_CHUNK
    for layer in range(B_LAYERS):
        views.append(
            buf.as_strided(
                (NB, *B_INNER),
                (block_elems, 8 * 32, 32, 1),
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
    assert all(f == lmcache_native.EngineKVFormat.NB_NL_NH_BS_CS for f in formats)
    # Each group's entries are its own reconstructed tensor.
    assert normalized[0] is normalized[1]
    assert normalized[2] is normalized[4]
    assert normalized[0] is not normalized[2]
    assert tuple(normalized[0].shape) == (NB, A_LAYERS, *A_INNER)
    assert tuple(normalized[2].shape) == (NB, B_LAYERS, *B_INNER)

    manager = KVLayerGroupsManager(
        normalized,
        engine_kv_formats=formats,
        engine_group_infos=(),
        separate_object_groups=False,
    )
    assert len(manager.kernel_groups) == 2

    for group in manager.kernel_groups:
        fmt = formats[group.layer_indices[0]]
        ptrs = get_group_data_ptrs(normalized, fmt, list(group.layer_indices))
        entry = normalized[group.layer_indices[0]]
        chunk_bytes = entry.stride(1) * entry.element_size()
        expected = [
            entry.data_ptr() + i * chunk_bytes for i in range(len(group.layer_indices))
        ]
        assert ptrs == expected
