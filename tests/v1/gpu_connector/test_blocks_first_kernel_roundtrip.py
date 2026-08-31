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
from lmcache import device_ops
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format import detect_format
import lmcache.lmcache_native as lmcache_native

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

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
        NB * total * NH * BS * CS, dtype=torch.float32, device="cuda"
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
    ptrs = torch.tensor([v.data_ptr() for v in kv], dtype=torch.int64, device="cuda")
    # A blocks-first pool always needs its real step: every block packs all
    # layers, so stride(0) exceeds the per-layer tight step even unpadded.
    block_stride = kv[0].stride(0)
    slots = torch.tensor([5, 6, 7, 12, 13, 14, 15, 28], device="cuda")
    staging = torch.zeros(1, NL, len(slots), SPT, dtype=torch.float32, device="cuda")

    device_ops.multi_layer_kv_transfer(
        staging,
        ptrs,
        slots,
        torch.device("cuda:0"),
        NB * BS,
        int(lmcache_native.TransferDirection.D2H),
        int(fmt),
        BS,
        CS // 2,
        0,
        block_stride,
    )
    torch.cuda.synchronize()
    ref = torch_gather(buf, order, slots.cpu())
    assert torch.equal(staging.cpu(), ref.cpu())

    # Zero the touched blocks, write back, expect original bytes restored.
    original = buf.clone()
    for b in {int(s) // BS for s in slots.cpu()}:
        buf[b, :NL].zero_()
    device_ops.multi_layer_kv_transfer(
        staging,
        ptrs,
        slots,
        torch.device("cuda:0"),
        NB * BS,
        int(lmcache_native.TransferDirection.H2D),
        int(fmt),
        BS,
        CS // 2,
        0,
        block_stride,
    )
    torch.cuda.synchronize()
    touched = sorted({int(s) // BS for s in slots.cpu()})
    for b in touched:
        used = [int(s) % BS for s in slots.cpu() if int(s) // BS == b]
        if order == "BLHNC":
            assert torch.equal(buf[b, :NL, :, used], original[b, :NL, :, used])
        else:
            assert torch.equal(buf[b, :NL, used], original[b, :NL, used])
