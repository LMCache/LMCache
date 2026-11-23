# SPDX-License-Identifier: Apache-2.0
"""Paged-buffer offsets past 2**31 scalars in the per-token transfer kernels.

``page_buffer_offset`` and ``key_value_offset`` turn a slot index into a
scalar offset into a layer's paged buffer. They used to compute that offset
as an int32 product (``token_idx * scalars_per_token``) before widening it, so
a pool with more than 2**31 scalars per layer wrapped to a negative offset and
the scatter wrote out of bounds (an illegal memory access that poisons the
CUDA context). Wide-row groups reach that size in practice: a 13831-scalar
hybrid aux group overflows past slot ~155k.

Each case transfers rows at the top of a pool just larger than 2**31 scalars
and checks them against direct tensor indexing (int64). The head size is odd
so the kernel moves 2-byte units, which keeps the pool at ~4 GiB per plane.
"""

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

_HS = 1025  # odd: forces 2-byte transfer units for bf16
_BS = 16
_N_TOK = 48
_DTYPE = torch.bfloat16
_INT32_MAX = 2**31 - 1


def _num_blocks() -> int:
    """Smallest block count whose top ``_N_TOK`` slots all overflow int32."""
    first_slot = _INT32_MAX // _HS + 1
    return (first_slot + _N_TOK) // _BS + 1


def _require_free_bytes(nbytes: int) -> None:
    """Skip the test when the GPU cannot hold ``nbytes`` plus headroom."""
    free, _ = torch_dev.mem_get_info()
    if free < nbytes + (1 << 30):
        pytest.skip(f"needs {nbytes / 2**30:.1f} GiB free GPU memory")


@pytest.mark.skipif(cuda_ops is None, reason="cuda_ops not built")
@pytest.mark.parametrize(
    "engine_kv_format",
    [
        lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,  # MLA / single plane
        lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,  # flash-attn NHD
    ],
    ids=["mla", "flash_attn_nhd"],
)
def test_transfer_past_int32_scalar_offset(engine_kv_format) -> None:
    """Rows whose scalar offset exceeds int32 land at the right address,
    in both directions."""
    num_blocks = _num_blocks()
    is_mla = engine_kv_format == lmcache_native.EngineKVFormat.NL_X_NB_BS_HS
    kv_size = 1 if is_mla else 2
    page_buffer_size = num_blocks * _BS
    _require_free_bytes(kv_size * page_buffer_size * _HS * _DTYPE.itemsize)

    device = torch.device(torch_device_type)
    paged_shape = (num_blocks, _BS, _HS) if is_mla else (2, num_blocks, _BS, 1, _HS)
    paged = torch.zeros(paged_shape, dtype=_DTYPE, device=device)
    planes = paged.view(kv_size, page_buffer_size, _HS)
    ptrs = torch.tensor([paged.data_ptr()], dtype=torch.long, device=device)

    slots = torch.arange(
        page_buffer_size - _N_TOK, page_buffer_size, dtype=torch.long, device=device
    )
    if int(slots[0]) * _HS <= _INT32_MAX:
        raise ValueError("test geometry does not exceed int32 offsets")
    src = torch.randn(kv_size, 1, _N_TOK, _HS, dtype=_DTYPE, device=device)

    cuda_ops.multi_layer_kv_transfer(
        src,
        ptrs,
        slots,
        device,
        page_buffer_size,
        lmcache_native.TransferDirection.H2D,
        engine_kv_format,
        block_size=_BS,
        head_size=_HS,
    )
    torch_dev.synchronize()
    for plane in range(kv_size):
        assert torch.equal(planes[plane, slots], src[plane, 0])
    # Nothing landed at the wrapped-around (low) addresses.
    assert not planes[:, :_N_TOK].any()

    back = torch.zeros_like(src)
    cuda_ops.multi_layer_kv_transfer(
        back,
        ptrs,
        slots,
        device,
        page_buffer_size,
        lmcache_native.TransferDirection.D2H,
        engine_kv_format,
        block_size=_BS,
        head_size=_HS,
    )
    torch_dev.synchronize()
    assert torch.equal(back, src)
