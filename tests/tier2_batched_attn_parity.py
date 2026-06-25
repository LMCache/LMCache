#!/usr/bin/env python3
"""Tier-2 batched selective-recompute: offline attention-packing parity test.

Proves the *kernel* of the Tier-2 design (see codecsight-bench/TIER2_BATCHED_BLEND.md
§4.1): packing N requests' selective-recompute attention into ONE varlen
flash-attn call is numerically equal to running N independent single-request
calls, with NO cross-request leakage.

This isolates the part of Tier-2 most likely to hide a subtle bug (the cu_seqlens
packing + the "anchor queries attend to their own full cached KV, causally"
semantics that the LMC attention backend uses) and validates it WITHOUT lmcache,
c_ops, a model, or a vLLM server -- just flash-attn on a GPU. Run on any free GPU:

    conda run -n vllm python3 tests/tier2_batched_attn_parity.py

Semantics replicated from blender._apply_selected_indices + flash_attn.forward_contiguous:
  - per request i: A_i anchor queries, S_i full cached keys (A_i <= S_i),
  - causal=True (anchor queries treated as the tail A_i positions of the seq,
    matching the kernel's varlen-causal convention),
  - GQA: num_heads_q = G * num_heads_kv.
"""
# Standard
import sys

# Third Party
import torch

try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func
except Exception as e:  # pragma: no cover
    print(f"FAIL: cannot import flash_attn_varlen_func ({e})")
    sys.exit(2)


def single_request_attn(q, k, v, scale, causal=True):
    """One request: A queries attend to S keys (varlen, batch=1) -- mirrors the
    Tier-1 forward_contiguous path with cu_seqlens_q=[0,A], cu_seqlens_k=[0,S]."""
    A = q.shape[0]
    S = k.shape[0]
    dev = q.device
    out = torch.empty_like(q)
    flash_attn_varlen_func(
        q=q, k=k, v=v, out=out,
        cu_seqlens_q=torch.tensor([0, A], dtype=torch.int32, device=dev),
        max_seqlen_q=A,
        cu_seqlens_k=torch.tensor([0, S], dtype=torch.int32, device=dev),
        max_seqlen_k=S,
        softmax_scale=scale,
        causal=causal,
    )
    return out


def batched_attn(qs, ks, vs, scale, causal=True):
    """N requests packed into ONE varlen call. cu_seqlens_q marks anchor-query
    segments, cu_seqlens_k marks full-cached-key segments."""
    dev = qs[0].device
    A_list = [q.shape[0] for q in qs]
    S_list = [k.shape[0] for k in ks]
    q = torch.cat(qs, dim=0)
    k = torch.cat(ks, dim=0)
    v = torch.cat(vs, dim=0)
    cu_q = torch.tensor([0] + list(torch.tensor(A_list).cumsum(0)),
                        dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0] + list(torch.tensor(S_list).cumsum(0)),
                        dtype=torch.int32, device=dev)
    out = torch.empty_like(q)
    flash_attn_varlen_func(
        q=q, k=k, v=v, out=out,
        cu_seqlens_q=cu_q, max_seqlen_q=max(A_list),
        cu_seqlens_k=cu_k, max_seqlen_k=max(S_list),
        softmax_scale=scale,
        causal=causal,
    )
    # split back into per-request segments
    outs = []
    off = 0
    for A in A_list:
        outs.append(out[off:off + A])
        off += A
    return outs


def run(seed, segs, n_heads_kv=8, gqa=4, head_dim=128, dtype=torch.bfloat16):
    torch.manual_seed(seed)
    dev = "cuda"
    n_heads_q = n_heads_kv * gqa
    scale = head_dim ** -0.5
    qs, ks, vs = [], [], []
    for (S, A) in segs:
        assert A <= S
        qs.append(torch.randn(A, n_heads_q, head_dim, device=dev, dtype=dtype))
        ks.append(torch.randn(S, n_heads_kv, head_dim, device=dev, dtype=dtype))
        vs.append(torch.randn(S, n_heads_kv, head_dim, device=dev, dtype=dtype))

    serial = [single_request_attn(q, k, v, scale) for q, k, v in zip(qs, ks, vs)]
    packed = batched_attn(qs, ks, vs, scale)

    max_abs = 0.0
    for i, (a, b) in enumerate(zip(serial, packed)):
        d = (a.float() - b.float()).abs().max().item()
        max_abs = max(max_abs, d)
    return max_abs


def run_leak_check(seed=0):
    """Adversarial: perturb request 1's keys; request 0's packed output must NOT
    change (proves cu_seqlens isolates segments -- no cross-request attention)."""
    torch.manual_seed(seed)
    dev = "cuda"
    n_heads_kv, gqa, head_dim = 8, 4, 128
    n_heads_q = n_heads_kv * gqa
    scale = head_dim ** -0.5
    segs = [(2000, 300), (2500, 380)]
    qs, ks, vs = [], [], []
    for (S, A) in segs:
        qs.append(torch.randn(A, n_heads_q, head_dim, device=dev, dtype=torch.bfloat16))
        ks.append(torch.randn(S, n_heads_kv, head_dim, device=dev, dtype=torch.bfloat16))
        vs.append(torch.randn(S, n_heads_kv, head_dim, device=dev, dtype=torch.bfloat16))
    out0_before = batched_attn(qs, ks, vs, scale)[0].clone()
    # corrupt request 1's KV entirely
    ks[1] = torch.randn_like(ks[1])
    vs[1] = torch.randn_like(vs[1])
    out0_after = batched_attn(qs, ks, vs, scale)[0]
    return (out0_before.float() - out0_after.float()).abs().max().item()


def main():
    if not torch.cuda.is_available():
        print("FAIL: CUDA required for flash-attn")
        sys.exit(2)

    # bf16 flash-attn vs flash-attn: differences are pure kernel-tiling/reduction
    # order, so tolerance is tight. (Same dtype both sides -> not a precision test.)
    TOL = 5e-3
    cases = {
        "uniform N=4": [(3000, 450)] * 4,
        "ragged   N=4": [(2800, 420), (3200, 500), (1500, 230), (3600, 540)],
        "ragged   N=8": [(2800, 420), (3200, 500), (1500, 230), (3600, 540),
                          (2000, 300), (900, 140), (4096, 600), (2600, 390)],
        "full-sel N=3": [(3000, 3000), (2500, 2500), (1800, 1800)],  # A==S edge
        "tiny-anchor": [(4096, 16), (4096, 1), (2048, 8)],            # A<<S edge
    }
    ok = True
    print("=== Tier-2 batched-attention packing parity (serial vs packed) ===")
    for name, segs in cases.items():
        d = run(seed=hash(name) & 0xffff, segs=segs)
        status = "PASS" if d <= TOL else "FAIL"
        ok = ok and (d <= TOL)
        print(f"  [{status}] {name:14s}  max|Δ| = {d:.2e}  (tol {TOL:.0e})  "
              f"N={len(segs)} ΣA={sum(a for _,a in segs)} ΣS={sum(s for s,_ in segs)}")

    leak = run_leak_check()
    leak_ok = leak == 0.0
    ok = ok and leak_ok
    print(f"  [{'PASS' if leak_ok else 'FAIL'}] no-leak check  "
          f"req0 Δ when req1 KV corrupted = {leak:.2e}  (must be exactly 0)")

    print("=== " + ("ALL PASS" if ok else "FAILURES PRESENT") + " ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
