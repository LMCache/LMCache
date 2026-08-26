# SPDX-License-Identifier: Apache-2.0
"""GPU round-trip for the blocks-first cross-layer formats.

Paged pool -> LMCache staging (D2H direction of the kernel) must equal a
pure-torch gather, and writing it back into a zeroed pool (H2D) must
restore the original bytes. Exercises the inflated per-block step that
distinguishes NB_NL_* from the per-layer formats.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format import detect_format, get_spec
import lmcache.lmcache_native as lmcache_native
from lmcache import device_ops

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NB, NL, NH, BS, CS = 8, 3, 2, 4, 16
SPT = NH * CS  # scalars per token


def make_pool(order: str) -> tuple[torch.Tensor, list[torch.Tensor]]:
    inner = (NH, BS, CS) if order == "BLHNC" else (BS, NH, CS)
    buf = torch.arange(NB * NL * NH * BS * CS, dtype=torch.float32, device="cuda")
    buf = buf.reshape(NB, NL, *inner)
    return buf, [buf[:, layer] for layer in range(NL)]


def torch_gather(buf: torch.Tensor, order: str, slots: torch.Tensor) -> torch.Tensor:
    """[1, NL, T, SPT] reference: token-major per layer, heads flattened."""
    blocks, offsets = slots // BS, slots % BS
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
def test_kernel_roundtrip_matches_torch(order):
    buf, views = make_pool(order)
    fmt, kv = detect_format(views, EngineType.VLLM, {"kv_layout": order})
    spec = get_spec(kv, fmt)
    ptrs = torch.tensor(
        spec.data_ptrs(list(range(NL))), dtype=torch.int64, device="cuda"
    )
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
    )
    torch.cuda.synchronize()
    ref = torch_gather(buf, order, slots.cpu())
    assert torch.equal(staging.cpu(), ref.cpu())

    # Zero the touched blocks, write back, expect original bytes restored.
    original = buf.clone()
    for b in {int(s) // BS for s in slots.cpu()}:
        buf[b].zero_()
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
    )
    torch.cuda.synchronize()
    touched = sorted({int(s) // BS for s in slots.cpu()})
    for b in touched:
        used = [int(s) % BS for s in slots.cpu() if int(s) // BS == b]
        if order == "BLHNC":
            assert torch.equal(buf[b, :, :, used], original[b, :, :, used])
        else:
            assert torch.equal(buf[b, :, used], original[b, :, used])
