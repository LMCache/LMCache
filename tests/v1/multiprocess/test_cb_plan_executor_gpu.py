# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the CB retrieve plan executor's copy/compute overlap.

The executor stages each step on a dedicated copy stream while the previous
step's rope/scatter kernels run on the compute stream, with per-parity CUDA
events ordering tmp-slot reuse (step w's staging may only start once step
w-2's kernels finished reading the same slot half). These tests drive many
steps of slot-half reuse where every wave writes DIFFERENT data into the same
slots — any event-ordering bug surfaces as a bit-exact mismatch against the
sequential tensor-op reference, not as a flake.

The stream-overlap cases exercise two paged layouts: the fused-packed
HND format (K/V in the trailing 2*HS dim, kv_size 1) and the un-fused
flash-attention format (separate K/V planes, kv_size 2). The scatter-only
cases check FlashAttention, FlashInfer and MLA against logical tensor indexing.
"""

# Standard
from dataclasses import dataclass

# Third Party
import numpy as np
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
import lmcache.lmcache_native as lmcache_native

if not (torch_dev.is_available() and torch_device_type == "cuda"):
    pytest.skip(
        "CUDA is not available, skipping the test",
        allow_module_level=True,
    )

# First Party
import lmcache.cuda_ops as cuda_ops  # noqa: E402

if not hasattr(cuda_ops, "execute_cb_retrieve_plan_flat"):
    pytest.skip(
        "cuda_ops build lacks execute_cb_retrieve_plan_flat",
        allow_module_level=True,
    )

_NL, _SPC, _NH, _HS = 4, 8, 2, 16
_NB, _BS = 512, 4
_DTYPE = torch.bfloat16


@dataclass(frozen=True)
class _FmtCase:
    """Per-format geometry: chunk plane count, widths, and paged shape."""

    fmt: "lmcache_native.EngineKVFormat"
    kv_size: int  # chunk leading planes (1 = K/V fused, 2 = split)
    hidden: int  # per-plane scalars per token
    head_stride: int  # rope stride between heads in the K plane
    paged_shape: tuple  # per-layer paged tensor shape


_CASES = {
    # Fused-packed HND: [NB, NH, BS, 2*HS], K is the first HS of each head.
    "packed": _FmtCase(
        fmt=lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_TWO_HS,
        kv_size=1,
        hidden=_NH * 2 * _HS,
        head_stride=2 * _HS,
        paged_shape=(_NB, _NH, _BS, 2 * _HS),
    ),
    # Un-fused flash-attention HND: [2, NB, NH, BS, HS], separate K/V planes.
    "split": _FmtCase(
        fmt=lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        kv_size=2,
        hidden=_NH * _HS,
        head_stride=_HS,
        paged_shape=(2, _NB, _NH, _BS, _HS),
    ),
}


def _reference_scatter(
    case, host_chunks, paged_ptrs, slot_mapping, old_sts, cur_sts, cos_sin
):
    """Sequential per-chunk tensor-op reference (rope then scatter)."""
    dev = slot_mapping.device
    ramp = torch.arange(_SPC, device=dev, dtype=torch.long).repeat(_NL)
    for i, host in enumerate(host_chunks):
        buf = host.to(dev)
        k_view = buf[0].reshape(_NL * _SPC, _NH, case.head_stride)
        cuda_ops.rotary_embedding_k_fused_strided(
            old_sts[i] + ramp,
            cur_sts[i] + ramp,
            k_view,
            _HS,
            case.head_stride,
            cos_sin,
            True,
        )
        cuda_ops.multi_layer_kv_transfer(
            buf,
            paged_ptrs,
            slot_mapping[i * _SPC : (i + 1) * _SPC],
            slot_mapping.device,
            _NB * _BS,
            lmcache_native.TransferDirection.H2D,
            case.fmt,
            block_size=_BS,
            head_size=_HS,
        )
    torch_dev.synchronize()


def _run_plan(
    case,
    n_chunks,
    max_batch,
    host_chunks,
    paged_ptrs,
    slot_mapping,
    old_sts,
    cur_sts,
    cos_sin,
    slots,
):
    """Drive the production flat-table entry point with the planner's
    double-buffer wave layout. A chunk repeating the previous one in the
    same wave is a D2D row from the slot that staged it, as the planner
    emits for repeated chunks."""
    chunk_bytes = case.kv_size * _NL * _SPC * case.hidden * _DTYPE.itemsize
    spec = cuda_ops.CBGroupSpec(
        paged_kv_ptrs=paged_ptrs.data_ptr(),
        temp_buffer_ptrs=[s.data_ptr() for s in slots],
        num_layers=_NL,
        slot_tokens=_SPC,
        hidden_elems=case.hidden,
        element_size=_DTYPE.itemsize,
        engine_kv_format=case.fmt,
        page_buffer_size=_NB * _BS,
        block_size=_BS,
        head_size=_HS,
        slot_mapping_base=slot_mapping.data_ptr(),
        slot_mapping_capacity=slot_mapping.numel(),
        cos_sin_cache=cos_sin.data_ptr(),
        rot_dim=_HS,
        rope_num_kv_heads=_NH,
        rope_head_stride=case.head_stride,
        key_scalar_type=15,  # at::ScalarType::BFloat16
        is_neox=True,
    )
    wave = max_batch // 2
    staging, ropes, scatters, step_offsets = [], [], [], []
    for w0 in range(0, n_chunks, wave):
        step_idx = w0 // wave
        base = (step_idx % 2) * wave
        for j in range(min(wave, n_chunks - w0)):
            ci, slot = w0 + j, base + j
            if j == 0 or host_chunks[ci] is not host_chunks[ci - 1]:
                head = slot
                staging.append(
                    (slots[slot].data_ptr(), host_chunks[ci].data_ptr(), chunk_bytes, 0)
                )
            else:
                staging.append(
                    (slots[slot].data_ptr(), slots[head].data_ptr(), chunk_bytes, -1)
                )
            ropes.append((0, slot, old_sts[ci], cur_sts[ci]))
            scatters.append((0, slot, ci * _SPC, _SPC))
        step_offsets.append((len(staging), len(ropes), len(scatters)))
    cuda_ops.execute_cb_retrieve_plan_flat(
        slot_mapping.device,
        1 << 26,
        [spec],
        np.asarray(staging, dtype=np.int64),
        np.asarray(ropes, dtype=np.int64),
        np.asarray(scatters, dtype=np.int64),
        np.asarray(step_offsets, dtype=np.int64),
    )
    torch_dev.synchronize()


@pytest.mark.parametrize("fmt_key", sorted(_CASES))
@pytest.mark.parametrize("n_chunks,max_batch", [(12, 4), (96, 16), (44, 16)])
@pytest.mark.parametrize("repeat", [1, 3])
def test_overlap_slot_reuse_is_bit_exact(n_chunks, max_batch, fmt_key, repeat):
    """Many steps of slot-half reuse, each wave carrying different data into
    the same slots: an ordering violation (staging overwriting a slot still
    being read, or kernels reading a half-staged slot) breaks bit-exactness.
    Covers full packs, partial tail packs, and deep reuse, on both the
    fused-packed and un-fused split-K/V paged layouts. With ``repeat`` > 1
    each stored chunk fills ``repeat`` positions (D2D rows within a wave),
    each re-RoPEd by its own shift."""
    case = _CASES[fmt_key]
    dev = torch.device(torch_device_type)
    torch.manual_seed(n_chunks)

    paged_ref = [
        torch.zeros(*case.paged_shape, dtype=_DTYPE, device=dev) for _ in range(_NL)
    ]
    paged_new = [torch.zeros_like(t) for t in paged_ref]
    ptrs_ref = torch.tensor(
        [t.data_ptr() for t in paged_ref], dtype=torch.long, device=dev
    )
    ptrs_new = torch.tensor(
        [t.data_ptr() for t in paged_new], dtype=torch.long, device=dev
    )
    stored = [
        torch.randn(case.kv_size, _NL, _SPC, case.hidden, dtype=_DTYPE).pin_memory()
        for _ in range(n_chunks)
    ]
    host_chunks = [stored[i // repeat] for i in range(n_chunks)]
    slots = [
        torch.zeros(case.kv_size, _NL, _SPC, case.hidden, dtype=_DTYPE, device=dev)
        for _ in range(max_batch)
    ]
    cos_sin = torch.randn(8192, _HS, dtype=_DTYPE, device=dev)
    pos = torch.arange(0, n_chunks * _SPC, device=dev, dtype=torch.long)
    block_ids = torch.arange(_NB, device=dev, dtype=torch.long).flip(0)
    slot_mapping = block_ids[pos // _BS] * _BS + pos % _BS
    old_sts = [(i // repeat) * _SPC + 512 for i in range(n_chunks)]
    cur_sts = [i * _SPC for i in range(n_chunks)]

    _reference_scatter(
        case, host_chunks, ptrs_ref, slot_mapping, old_sts, cur_sts, cos_sin
    )
    _run_plan(
        case,
        n_chunks,
        max_batch,
        host_chunks,
        ptrs_new,
        slot_mapping,
        old_sts,
        cur_sts,
        cos_sin,
        slots,
    )

    for layer in range(_NL):
        assert torch.equal(paged_ref[layer], paged_new[layer]), (
            f"layer {layer} mismatch "
            f"(fmt={fmt_key}, n_chunks={n_chunks}, max_batch={max_batch})"
        )


@pytest.mark.parametrize(
    "fmt,axes,head_size",
    [
        (lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS, (0, 1, 2, 3, 4), 16),
        (lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS, (1, 0, 2, 3, 4), 16),
        (lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS, (0, 1, 3, 2, 4), 16),
        (lmcache_native.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS, (1, 0, 3, 2, 4), 16),
        (lmcache_native.EngineKVFormat.NL_X_NB_BS_HS, (0, 1, 2, 3, 4), 576),
        (lmcache_native.EngineKVFormat.NL_X_NB_BS_HS, (0, 1, 2, 3, 4), 66),
    ],
    ids=(
        "flash-attn-nhd",
        "flashinfer-nhd",
        "flash-attn-hnd",
        "flashinfer-hnd",
        "mla-576",
        "mla-66",
    ),
)
def test_scatter_only_writes_mapped_tokens(
    fmt: lmcache_native.EngineKVFormat, axes: tuple[int, ...], head_size: int
) -> None:
    """Scatter two chunks into permuted pages with a skipped slot and a short tail."""
    device = torch.device(torch_device_type)
    mla = fmt == lmcache_native.EngineKVFormat.NL_X_NB_BS_HS
    planes, heads = (1, 1) if mla else (2, 3)
    layers, blocks, block_size, slot_tokens = 2, 5, 4, 4
    positions = ([9, -1, 1, 18], [14, 4, 17])
    mapping = torch.tensor(positions[0] + positions[1], device=device)
    # Distinct 16-bit payloads make plane, layer, token and channel swaps observable.
    chunks = (
        torch.arange(
            2 * planes * layers * slot_tokens * heads * head_size,
            device=device,
            dtype=torch.int16,
        )
        .view(_DTYPE)
        .reshape(2, planes, layers, slot_tokens, heads, head_size)
    )
    expected = torch.full(
        (planes, layers, blocks, block_size, heads, head_size),
        -1,
        dtype=_DTYPE,
        device=device,
    )
    for chunk, slots in zip(chunks, positions, strict=True):
        for token, slot in enumerate(slots):
            if slot >= 0:
                block, offset = divmod(slot, block_size)
                expected[:, :, block, offset] = chunk[:, :, token]

    # Express physical page layout with tensor dimensions independently of the kernel.
    pages = [layer.permute(axes) for layer in expected.unbind(1)]
    if mla:
        pages = [layer.squeeze(0).squeeze(-2) for layer in pages]
    actual = [torch.full(page.shape, -1, device=device, dtype=_DTYPE) for page in pages]
    ptrs = torch.tensor([page.data_ptr() for page in actual], device=device)
    spec = cuda_ops.CBGroupSpec(
        paged_kv_ptrs=ptrs.data_ptr(),
        temp_buffer_ptrs=[chunk.data_ptr() for chunk in chunks],
        num_layers=layers,
        slot_tokens=slot_tokens,
        hidden_elems=heads * head_size,
        element_size=chunks.element_size(),
        engine_kv_format=fmt,
        page_buffer_size=blocks * block_size,
        block_size=block_size,
        head_size=head_size,
        slot_mapping_base=mapping.data_ptr(),
        slot_mapping_capacity=mapping.numel(),
        cos_sin_cache=0,
        rot_dim=0,
        rope_num_kv_heads=0,
        rope_head_stride=0,
        key_scalar_type=15,
        is_neox=True,
    )
    empty = np.empty((0, 4), dtype=np.int64)
    cuda_ops.execute_cb_retrieve_plan_flat(
        mapping.device,
        1 << 26,
        [spec],
        empty,
        empty,
        np.array([[0, 0, 0, 4], [0, 1, 4, 3]], dtype=np.int64),
        np.array([[0, 0, 2]], dtype=np.int64),
    )
    torch_dev.synchronize()
    for page, reference in zip(actual, pages, strict=True):
        assert torch.equal(page.view(torch.int16), reference.view(torch.int16))
