# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the CB retrieve plan executor's copy/compute overlap.

The executor stages each step on a dedicated copy stream while the previous
step's rope/scatter kernels run on the compute stream, with per-parity CUDA
events ordering tmp-slot reuse (step w's staging may only start once step
w-2's kernels finished reading the same slot half). These tests drive many
steps of slot-half reuse where every wave writes DIFFERENT data into the same
slots — any event-ordering bug surfaces as a bit-exact mismatch against the
sequential tensor-op reference, not as a flake.
"""

# Third Party
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "CUDA is not available, skipping the test",
        allow_module_level=True,
    )

# First Party
import lmcache.c_ops as lmc_ops  # noqa: E402

if not hasattr(lmc_ops, "execute_cb_retrieve_plan"):
    pytest.skip(
        "c_ops build lacks execute_cb_retrieve_plan",
        allow_module_level=True,
    )

_NL, _SPC, _NH, _HS = 4, 8, 2, 16
_HIDDEN = _NH * 2 * _HS  # fused packed: per-head width 2*HS
_NB, _BS = 512, 4
_FMT = lmc_ops.EngineKVFormat.NL_X_NB_NH_BS_TWO_HS
_DTYPE = torch.bfloat16


def _reference_scatter(host_chunks, paged_ptrs, slot_mapping, old_sts, cur_sts, cos_sin):
    """Sequential per-chunk tensor-op reference (rope then scatter)."""
    dev = slot_mapping.device
    ramp = torch.arange(_SPC, device=dev, dtype=torch.long).repeat(_NL)
    for i, host in enumerate(host_chunks):
        buf = host.to(dev)
        k_view = buf[0].reshape(_NL * _SPC, _NH, 2 * _HS)
        lmc_ops.rotary_embedding_k_fused_strided(
            old_sts[i] + ramp, cur_sts[i] + ramp, k_view, _HS, 2 * _HS, cos_sin, True
        )
        lmc_ops.multi_layer_kv_transfer(
            buf,
            paged_ptrs,
            slot_mapping[i * _SPC : (i + 1) * _SPC],
            slot_mapping.device,
            _NB * _BS,
            lmc_ops.TransferDirection.H2D,
            _FMT,
            block_size=_BS,
            head_size=_HS,
        )
    torch.cuda.synchronize()


def _run_plan(n_chunks, max_batch, host_chunks, paged_ptrs, slot_mapping, old_sts, cur_sts, cos_sin, slots):
    """Drive the executor with the planner's double-buffer wave layout."""
    chunk_bytes = _NL * _SPC * _HIDDEN * _DTYPE.itemsize
    spec = lmc_ops.CBGroupSpec(
        paged_kv_ptrs=paged_ptrs.data_ptr(),
        temp_buffer_ptrs=[s.data_ptr() for s in slots],
        num_layers=_NL,
        slot_tokens=_SPC,
        hidden_elems=_HIDDEN,
        element_size=_DTYPE.itemsize,
        engine_kv_format=_FMT,
        page_buffer_size=_NB * _BS,
        block_size=_BS,
        head_size=_HS,
        slot_mapping_base=slot_mapping.data_ptr(),
        slot_mapping_capacity=slot_mapping.numel(),
        cos_sin_cache=cos_sin.data_ptr(),
        rot_dim=_HS,
        rope_num_kv_heads=_NH,
        rope_head_size=_HS,
        rope_head_stride=2 * _HS,
        key_scalar_type=15,  # at::ScalarType::BFloat16
        is_neox=True,
    )
    wave = max_batch // 2
    steps = []
    for w0 in range(0, n_chunks, wave):
        step_idx = w0 // wave
        base = (step_idx % 2) * wave
        staging, ropes, scatters = [], [], []
        for j in range(min(wave, n_chunks - w0)):
            ci, slot = w0 + j, base + j
            staging.append(
                lmc_ops.StagingCopy(
                    slots[slot].data_ptr(), host_chunks[ci].data_ptr(), chunk_bytes, 0
                )
            )
            ropes.append(lmc_ops.CBRopeVar(0, slot, old_sts[ci], cur_sts[ci]))
            scatters.append(lmc_ops.CBScatterVar(0, slot, ci * _SPC, _SPC))
        steps.append(
            lmc_ops.CBRetrieveStep(staging=staging, ropes=ropes, scatters=scatters)
        )
    lmc_ops.execute_cb_retrieve_plan(
        slot_mapping.device, 1 << 26, [spec], steps
    )
    torch.cuda.synchronize()


@pytest.mark.parametrize("n_chunks,max_batch", [(12, 4), (96, 16), (44, 16)])
def test_overlap_slot_reuse_is_bit_exact(n_chunks, max_batch):
    """Many steps of slot-half reuse, each wave carrying different data into
    the same slots: an ordering violation (staging overwriting a slot still
    being read, or kernels reading a half-staged slot) breaks bit-exactness.
    Covers full packs, partial tail packs, and deep reuse."""
    dev = torch.device("cuda:0")
    torch.manual_seed(n_chunks)

    paged_ref = [
        torch.zeros(_NB, _NH, _BS, 2 * _HS, dtype=_DTYPE, device=dev)
        for _ in range(_NL)
    ]
    paged_new = [torch.zeros_like(t) for t in paged_ref]
    ptrs_ref = torch.tensor(
        [t.data_ptr() for t in paged_ref], dtype=torch.long, device=dev
    )
    ptrs_new = torch.tensor(
        [t.data_ptr() for t in paged_new], dtype=torch.long, device=dev
    )
    host_chunks = [
        torch.randn(1, _NL, _SPC, _HIDDEN, dtype=_DTYPE).pin_memory()
        for _ in range(n_chunks)
    ]
    slots = [
        torch.zeros(1, _NL, _SPC, _HIDDEN, dtype=_DTYPE, device=dev)
        for _ in range(max_batch)
    ]
    cos_sin = torch.randn(8192, _HS, dtype=_DTYPE, device=dev)
    pos = torch.arange(0, n_chunks * _SPC, device=dev, dtype=torch.long)
    block_ids = torch.arange(_NB, device=dev, dtype=torch.long).flip(0)
    slot_mapping = block_ids[pos // _BS] * _BS + pos % _BS
    old_sts = [i * _SPC + 512 for i in range(n_chunks)]
    cur_sts = [i * _SPC for i in range(n_chunks)]

    _reference_scatter(host_chunks, ptrs_ref, slot_mapping, old_sts, cur_sts, cos_sin)
    _run_plan(
        n_chunks, max_batch, host_chunks, ptrs_new, slot_mapping,
        old_sts, cur_sts, cos_sin, slots,
    )

    for layer in range(_NL):
        assert torch.equal(paged_ref[layer], paged_new[layer]), (
            f"layer {layer} mismatch (n_chunks={n_chunks}, max_batch={max_batch})"
        )
