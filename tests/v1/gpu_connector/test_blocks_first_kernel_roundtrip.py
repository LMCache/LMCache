# SPDX-License-Identifier: Apache-2.0
"""GPU round-trip for blocks-first pools through the per-layer CS formats.

Paged pool -> LMCache staging (D2H direction of the kernel) must equal a
pure-torch gather, and writing it back into a zeroed pool (H2D) must
restore the original bytes. Exercises the inflated per-block step that
distinguishes a blocks-first pool from a layer-compact cache: the step
rides in block_stride_elems, taken from the views' stride(0).
"""

# Third Party
import pytest
import torch

# First Party
from lmcache import device_ops, torch_dev, torch_device_type
from lmcache.v1.common.custom_types import EngineType
from lmcache.v1.common.kv_format import detect_format
from lmcache.v1.common.utils import make_page_buffer_shape_desc
from lmcache.v1.platform.native_block_transfer import transfer_with_address_kernel
import lmcache.lmcache_native as lmcache_native

cuda_only = pytest.mark.skipif(
    torch_device_type != "cuda" or not torch_dev.is_available(), reason="needs CUDA"
)

NB, NL, NH, BS, CS = 8, 3, 2, 4, 16
SPT = NH * CS  # scalars per token


def make_pool(
    order: str, pad_layers: int = 0
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Blocks-first pool; ``pad_layers`` extra layer slots per block simulate
    an HMA pool shared with other groups (block step > NL * chunk)."""
    inner = (NH, BS, CS) if order == "BLHNC" else (BS, NH, CS)
    total = NL + pad_layers
    buf = torch.arange(
        NB * total * NH * BS * CS, dtype=torch.float32, device=torch_device_type
    ).reshape(NB, total, *inner)
    return buf, [buf[:, layer] for layer in range(NL)]


def torch_gather(buf: torch.Tensor, order: str, slots: torch.Tensor) -> torch.Tensor:
    """[1, NL, T, SPT] reference: token-major per layer, heads flattened."""
    blocks, offsets = slots // BS, slots % BS
    buf = buf[:, :NL]
    if order == "BLHNC":  # buf [NB, NL, NH, BS, CS]
        out = torch.stack(
            [
                buf[b, :, :, o].reshape(NL, SPT)
                for b, o in zip(blocks.tolist(), offsets.tolist(), strict=True)
            ],
            dim=1,
        )
    else:  # buf [NB, NL, BS, NH, CS]
        out = torch.stack(
            [
                buf[b, :, o].reshape(NL, SPT)
                for b, o in zip(blocks.tolist(), offsets.tolist(), strict=True)
            ],
            dim=1,
        )
    return out.unsqueeze(0)  # [1, NL, T, SPT]


@cuda_only
@pytest.mark.parametrize("order", ["BLHNC", "BLNHC"])
@pytest.mark.parametrize("pad_layers", [0, 2])
def test_kernel_roundtrip_matches_torch(order, pad_layers):
    buf, views = make_pool(order, pad_layers)
    fmt, kv = detect_format(views, EngineType.VLLM, {"kv_layout": order})
    expected_fmt = (
        lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_CS
        if order == "BLHNC"
        else lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_CS
    )
    assert fmt == expected_fmt
    assert kv[0].stride(0) == buf.stride(0)
    desc = make_page_buffer_shape_desc(
        kv, fmt, 0, NL, NB, BS, block_stride_elems=kv[0].stride(0)
    )
    blocks = [7, 1, 3]
    slots = torch.tensor(
        [block * BS + offset for block in blocks for offset in range(BS)]
    )
    staging = torch.zeros(1, NL, len(slots), SPT, dtype=buf.dtype, pin_memory=True)

    def transfer(direction: lmcache_native.TransferDirection) -> None:
        transfer_with_address_kernel(
            device_ops.multi_layer_block_kv_transfer,
            kv,
            [staging],
            blocks,
            buf.device,
            direction,
            desc,
            len(slots),
            fmt,
            0,
        )
        torch_dev.synchronize()

    transfer(lmcache_native.TransferDirection.D2H)
    assert torch.equal(staging, torch_gather(buf, order, slots).cpu())

    original = buf.clone()
    buf[blocks, :NL] = 0
    transfer(lmcache_native.TransferDirection.H2D)
    assert torch.equal(buf, original)
