# SPDX-License-Identifier: Apache-2.0
"""
Regression test for the D2H / H2D kv_interleaved layout consistency fix.

The per-chunk kernel (used for D2H store) and the layer-wise kernel (used for
H2D retrieve) must both honour ``PageBufferShapeDesc.kv_interleaved`` at
runtime.  Before the fix, the per-chunk kernel compiled out the interleaved
branch via a ``constexpr`` template parameter that defaulted to ``false``,
causing a layout mismatch when ``kv_size == 2`` and ``kv_interleaved == true``.

This test exercises a D2H->H2D round-trip with ``kv_size=2`` (split K/V
format, e.g. ``NL_X_TWO_NB_BS_NH_HS``) and ``kv_interleaved=true`` to
verify that data survives the trip without corruption.
"""

# Standard
import random

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
import lmcache.lmcache_native as lmcache_native

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not (torch_dev.is_available() and torch_device_type == "cuda"),
        reason="Requires CUDA backend",
    ),
]

try:
    # First Party
    import lmcache.cuda_ops as cuda_ops
except ImportError:
    cuda_ops = None


def _make_shape_desc(
    *,
    kv_size: int,
    nl: int,
    nb: int,
    bs: int,
    nh: int,
    hs: int,
    element_size: int,
    kv_interleaved: bool,
) -> "lmcache_native.PageBufferShapeDesc":
    sd = lmcache_native.PageBufferShapeDesc()
    sd.kv_size = kv_size
    sd.nl = nl
    sd.nb = nb
    sd.bs = bs
    sd.nh = nh
    sd.hs = hs
    sd.element_size = element_size
    sd.block_stride_elems = 0
    sd.kv_interleaved = kv_interleaved
    return sd


@pytest.mark.skipif(cuda_ops is None, reason="cuda_ops not built")
@pytest.mark.parametrize("num_tokens", [256, 512])
@pytest.mark.parametrize("kv_interleaved", [False, True])
@pytest.mark.parametrize(
    "engine_kv_format",
    [
        lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
    ],
)
def test_d2h_h2d_interleaved_roundtrip(num_tokens, kv_interleaved, engine_kv_format):
    """D2H (per-chunk) -> H2D (layerwise) round-trip with kv_size=2.

    When kv_interleaved=True, both the D2H and H2D kernels must use L2TD
    layout [L, 2, T, D].  Before the fix the D2H kernel always used 2LTD
    [2, L, T, D], causing silent data corruption on retrieve.
    """
    device = torch_device_type
    dtype = torch.bfloat16
    element_size = dtype.itemsize  # 2

    num_blocks = 200
    block_size = 16
    num_heads = 8
    head_size = 128
    num_layers = 4
    chunk_size = 256
    kv_size = 2  # split K/V

    # ---- engine-side paged KV cache [NL_X_TWO_NB_BS_NH_HS] ----
    # Each layer: [2, NB, BS, NH, HS]
    kv_cache = [
        torch.rand(
            [kv_size, num_blocks, block_size, num_heads, head_size],
            dtype=dtype,
            device=device,
        )
        for _ in range(num_layers)
    ]
    page_buffer_size = num_blocks * block_size

    # Pick random slots and derive the block IDs they touch
    slot_mapping = sorted(random.sample(range(page_buffer_size), num_tokens))
    block_ids_set = sorted({s // block_size for s in slot_mapping})

    # blocks_per_object: how many blocks fit in one LMCache chunk
    blocks_per_object = chunk_size // block_size
    assert blocks_per_object * block_size == chunk_size

    # We need to transfer exactly `blocks_per_object` blocks per object.
    # For simplicity, use the first blocks_per_object blocks from block_ids_set.
    block_ids_for_transfer = block_ids_set[:blocks_per_object]
    block_ids_t = torch.tensor(block_ids_for_transfer, device=device, dtype=torch.int64)

    # ---- per-layer pointer tensor ----
    kv_ptrs = torch.empty(num_layers, dtype=torch.int64, device="cpu", pin_memory=True)
    for i in range(num_layers):
        kv_ptrs[i] = kv_cache[i].data_ptr()

    # ---- shape_desc ----
    sd = _make_shape_desc(
        kv_size=kv_size,
        nl=num_layers,
        nb=num_blocks,
        bs=block_size,
        nh=num_heads,
        hs=head_size,
        element_size=element_size,
        kv_interleaved=kv_interleaved,
    )

    # ---- LMCache host object (pinned) ----
    D = num_heads * head_size
    # The flat size is kv_size * num_layers * chunk_size * D regardless of
    # the interleaving order; the kernel interprets the byte offset.
    lmcache_obj = torch.zeros(
        [kv_size * num_layers * chunk_size * D],
        dtype=dtype,
        device="cpu",
        pin_memory=True,
    )

    # ---- D2H: per-chunk kernel (store) ----
    # cuda_ops.multi_layer_block_kv_transfer(
    #     paged_buffer_ptrs_tensor, lmcache_objects_ptrs, block_ids,
    #     device, direction, shape_desc, lmcache_chunk_size,
    #     engine_kv_format, skip_prefix_n_blocks)
    cuda_ops.multi_layer_block_kv_transfer(
        kv_ptrs,  # paged_buffer_ptrs_tensor
        [lmcache_obj.data_ptr()],  # lmcache_objects_ptrs
        block_ids_t,  # block_ids
        kv_cache[0].device,  # device
        int(lmcache_native.TransferDirection.D2H),  # direction
        sd,  # shape_desc
        chunk_size,  # lmcache_chunk_size
        int(engine_kv_format),  # engine_kv_format
        0,  # skip_prefix_n_blocks
    )
    torch_dev.synchronize()

    # ---- H2D: layerwise kernel (retrieve) into fresh KV cache ----
    kv_cache_new = [
        torch.zeros(
            [kv_size, num_blocks, block_size, num_heads, head_size],
            dtype=dtype,
            device=device,
        )
        for _ in range(num_layers)
    ]
    kv_ptrs_new = torch.empty(
        num_layers, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        kv_ptrs_new[i] = kv_cache_new[i].data_ptr()

    cuda_ops.multi_layer_block_kv_transfer_layerwise(
        kv_ptrs_new,  # paged_buffer_ptrs_tensor
        [lmcache_obj.data_ptr()],  # lmcache_objects_ptrs
        block_ids_t,  # block_ids
        kv_cache_new[0].device,  # device
        int(lmcache_native.TransferDirection.H2D),  # direction
        sd,  # shape_desc
        chunk_size,  # lmcache_chunk_size
        int(engine_kv_format),  # engine_kv_format
        0,  # skip_prefix_n_blocks
    )
    torch_dev.synchronize()

    # ---- verify round-trip ----
    for layer_idx in range(num_layers):
        for block_id in block_ids_for_transfer:
            orig = kv_cache[layer_idx][:, block_id, :, :, :]
            restored = kv_cache_new[layer_idx][:, block_id, :, :, :]
            if not torch.equal(orig, restored):
                diff_mask = orig != restored
                n_diff = diff_mask.sum().item()
                pytest.fail(
                    f"Round-trip mismatch at layer={layer_idx}, "
                    f"block={block_id}: {n_diff}/{orig.numel()} elements differ "
                    f"(kv_interleaved={kv_interleaved})"
                )
