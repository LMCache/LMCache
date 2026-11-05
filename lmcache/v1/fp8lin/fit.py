# SPDX-License-Identifier: Apache-2.0
"""Fit an fp8lin coder for a model and write its coder directory.

Captures the model's KV on a few hundred documents with HuggingFace transformers,
rounds it to E4M3 the way the serving engine stores it, and fits, per layer and
head block, the least-squares prediction from the token's other heads (keys
un-rotated out of RoPE) and its previous layer, plus the residual scales.

KV conventions (``--kv``):

* ``bf16``: the serving engine keeps a bf16 KV cache; the serde rounds it to E4M3
  at per-layer scales ``headroom * amax / 448`` measured here.
* ``fp8``: the serving engine keeps an FP8 KV cache with unit scales (vLLM
  ``kv_cache_dtype="fp8"`` without calibrated scales); attention reads the FP8
  cache during capture, as in the engine.

Example::

    python -m lmcache.v1.fp8lin.fit --model Qwen/Qwen3-8B --kv bf16 \\
        --out /models/qwen3-8b-fp8lin

Text comes from ``--text`` (a UTF-8 file, split into ``--seq-len`` token windows) or,
by default, the first shard of C4 (``allenai/c4``, downloaded from the HF hub).
"""

# Standard
from pathlib import Path
import argparse
import gzip
import json

# Third Party
import numpy as np
import torch

# First Party
from lmcache import torch_dev

FP8_MAX = 448.0


def _c4_texts(n_docs: int) -> list[str]:
    # Third Party
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "allenai/c4", "en/c4-train.00000-of-01024.json.gz", repo_type="dataset"
    )
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line)["text"])
            if len(out) >= 20 * n_docs:
                break
    return out


def _docs(tok, n_docs: int, seq_len: int, text: str | None) -> list[list[int]]:
    """Token windows of exactly ``seq_len`` tokens."""
    texts = [Path(text).read_text(encoding="utf-8")] if text else _c4_texts(n_docs)
    stream: list[int] = []
    docs: list[list[int]] = []
    for t in texts:
        stream += tok(t, add_special_tokens=False)["input_ids"]
        while len(stream) >= seq_len and len(docs) < n_docs:
            docs.append(stream[:seq_len])
            stream = stream[seq_len:]
        if len(docs) >= n_docs:
            break
    if len(docs) < n_docs:
        raise ValueError(f"only {len(docs)} windows of {seq_len} tokens in the text")
    return docs


def _kv(cache, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(cache, "layers"):
        return cache.layers[layer].keys, cache.layers[layer].values
    return cache.key_cache[layer], cache.value_cache[layer]


def _to_codes(x: torch.Tensor, scale: float) -> torch.Tensor:
    y = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX)
    return y.to(torch.float8_e4m3fn).view(torch.uint8)


@torch.no_grad()
def capture(
    model, docs: list[list[int]], scales: dict[str, float] | None, fp8_cache: bool
) -> list[torch.Tensor]:
    """Per layer, uint8 E4M3 codes [n_docs * seq_len, 2C] (K post-RoPE, then V).

    With ``fp8_cache`` attention reads the E4M3-rounded cache (requires scales).
    """
    # Third Party
    from transformers.cache_utils import DynamicCache

    class _FP8Cache(DynamicCache):
        def update(self, k, v, layer_idx, *args, **kwargs):  # type: ignore[override]
            assert scales is not None
            q = lambda x, s: (_to_codes(x, s).view(torch.float8_e4m3fn).float() * s).to(
                x.dtype
            )
            return super().update(
                q(k, scales[f"L{layer_idx}.k"]),
                q(v, scales[f"L{layer_idx}.v"]),
                layer_idx,
                *args,
                **kwargs,
            )

    L = model.config.num_hidden_layers
    out: list[list[torch.Tensor]] = [[] for _ in range(L)]
    for ids in docs:
        cache = _FP8Cache() if fp8_cache else DynamicCache()
        model(
            input_ids=torch.tensor([ids], device="cuda"),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        for layer in range(L):
            k, v = _kv(cache, layer)
            k = k[0].permute(1, 0, 2).reshape(len(ids), -1)
            v = v[0].permute(1, 0, 2).reshape(len(ids), -1)
            sk = 1.0 if scales is None else scales[f"L{layer}.k"]
            sv = 1.0 if scales is None else scales[f"L{layer}.v"]
            out[layer].append(torch.cat([_to_codes(k, sk), _to_codes(v, sv)], 1).cpu())
    return [torch.cat(o) for o in out]


@torch.no_grad()
def amax_scales(model, docs: list[list[int]], headroom: float) -> dict[str, float]:
    L = model.config.num_hidden_layers
    amax = {f"L{layer}.{n}": 0.0 for layer in range(L) for n in "kv"}
    for ids in docs:
        res = model(
            input_ids=torch.tensor([ids], device="cuda"),
            use_cache=True,
            logits_to_keep=1,
        )
        for layer in range(L):
            for x, n in zip(_kv(res.past_key_values, layer), "kv", strict=True):
                amax[f"L{layer}.{n}"] = max(
                    amax[f"L{layer}.{n}"], x.float().abs().max().item()
                )
    return {k: (headroom * m / FP8_MAX if m > 0 else 1.0) for k, m in amax.items()}


def _features(
    codes: torch.Tensor,
    pos: torch.Tensor,
    th: torch.Tensor,
    H: int,
    D: int,
    vals: torch.Tensor,
) -> torch.Tensor:
    """Float features [N, 2C]: K un-rotated out of RoPE (per head), then V."""
    x = vals[codes.long()]
    a = pos.double()[:, None] * th[None]
    cos, sin = a.cos().float(), a.sin().float()
    sgn = torch.cat([-torch.ones(D // 2), torch.ones(D // 2)]).to(x.device)
    k = []
    for h in range(H):
        xh = x[:, h * D : (h + 1) * D]
        k.append(xh * cos - sgn * xh.roll(D // 2, 1) * sin)
    return torch.cat(k + [x[:, H * D :]], 1)


@torch.no_grad()
def fit_layer(
    cur: torch.Tensor,
    prev: torch.Tensor | None,
    pos: torch.Tensor,
    th: torch.Tensor,
    H: int,
    D: int,
    ridge: float = 1e-4,
    chunk: int = 8192,
) -> dict[str, np.ndarray]:
    """Fit one layer's coder from its codes and the previous layer's codes (uint8
    [N, 2C], on the GPU). Returns the arrays of a ``layer_XXX.npz``."""
    dev = cur.device
    C = H * D
    vals = torch.nan_to_num(
        torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float(), nan=0.0
    ).to(dev)
    N = cur.shape[0]
    nz = 4 * C + 1

    def z_of(sl: slice) -> torch.Tensor:
        f = _features(cur[sl], pos[sl], th, H, D, vals)
        p = (
            _features(prev[sl], pos[sl], th, H, D, vals)
            if prev is not None
            else torch.zeros_like(f)
        )
        return torch.cat([f, p], 1)

    ss = torch.zeros(4 * C, dtype=torch.float64, device=dev)
    for i in range(0, N, chunk):
        ss += z_of(slice(i, i + chunk)).double().pow(2).sum(0)
    sd = (ss / N).sqrt().clamp_min(1e-6).float()
    G = torch.zeros(nz, nz, dtype=torch.float64, device=dev)
    for i in range(0, N, chunk):
        z = z_of(slice(i, i + chunk)) / sd
        z = torch.cat([z, torch.ones_like(z[:, :1])], 1).double()
        G += z.T @ z
    W, cols_all = [], []
    for b in range(2 * H):
        h, isv = b % H, b >= H
        m = C + h * D if isv else h * D
        cols = list(range(m)) + list(range(2 * C, 4 * C)) + [nz - 1]
        tc = list(range(b * D, (b + 1) * D))
        A = G[cols][:, cols] + ridge * N * torch.eye(
            len(cols), dtype=G.dtype, device=dev
        )
        A[-1, -1] -= ridge * N
        W.append(torch.linalg.solve(A, G[cols][:, tc]).float())
        cols_all.append(np.array(cols))
    # residual scales and the per-token scale model, on the same tokens
    e2 = torch.zeros(2 * C, dtype=torch.float64, device=dev)
    preds = []
    for i in range(0, N, chunk):
        sl = slice(i, i + chunk)
        z = z_of(sl) / sd
        z = torch.cat([z, torch.ones_like(z[:, :1])], 1)
        mu_u = torch.cat([z[:, c] @ w for c, w in zip(cols_all, W, strict=True)], 1)
        mu_u = mu_u * sd[: 2 * C]
        mu = _rotate_k(mu_u, pos[sl], th, H, D)
        x = vals[cur[sl].long()]
        e2 += (x - mu).double().pow(2).sum(0)
        preds.append((sl, mu))
    sig = (e2 / N).sqrt().clamp_min(1e-6).float()
    X, Y = [], []
    for sl, mu in preds:
        x = vals[cur[sl].long()]
        r = ((x - mu) / sig).pow(2).view(-1, 2 * H, D).mean(-1)
        prev_r = torch.cumsum(r, 1) - r
        cnt = torch.arange(2 * H, device=dev).clamp_min(1)
        g1 = torch.log(prev_r / cnt + 1e-3)
        g1[:, 0] = 0
        X.append(torch.stack([torch.ones_like(g1), g1], -1))
        Y.append(torch.log(r + 1e-4))
    Xc, Yc = torch.cat(X).double(), torch.cat(Y).double()
    Wh = torch.zeros(2 * H, 4)
    for b in range(2 * H):
        Xb = Xc[:, b]
        A = Xb.T @ Xb + 1e-6 * N * torch.eye(2, dtype=torch.float64, device=dev)
        Wh[b, :2] = torch.linalg.solve(A, Xb.T @ Yc[:, b]).float().cpu()
    arrays = {f"cols{b}": cols_all[b] for b in range(2 * H)}
    arrays.update({f"W{b}": W[b].cpu().numpy() for b in range(2 * H)})
    arrays.update(
        H=np.array(H),
        D=np.array(D),
        C=np.array(C),
        th=th.cpu().numpy(),
        groups=np.array(["curK", "curV", "p"]),
        goff=np.array([0, C, 2 * C]),
        sd=sd.cpu().numpy(),
        sig=sig.cpu().numpy(),
        Wh=Wh.numpy(),
    )
    return arrays


def _rotate_k(
    mu_u: torch.Tensor, pos: torch.Tensor, th: torch.Tensor, H: int, D: int
) -> torch.Tensor:
    """Rotate the K part of un-rotated predictions back into the RoPE frame."""
    a = pos.double()[:, None] * th[None]
    cos, sin = a.cos().float(), a.sin().float()
    sgn = torch.cat([-torch.ones(D // 2), torch.ones(D // 2)]).to(mu_u.device)
    k = []
    for h in range(H):
        xh = mu_u[:, h * D : (h + 1) * D]
        k.append(xh * cos + sgn * xh.roll(D // 2, 1) * sin)
    return torch.cat(k + [mu_u[:, H * D :]], 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kv", choices=["bf16", "fp8"], default="bf16")
    ap.add_argument("--docs", type=int, default=128, help="training windows")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--headroom", type=float, default=2.0, help="bf16: scale headroom")
    ap.add_argument("--text", default=None, help="UTF-8 text file (default: C4)")
    ap.add_argument("--max-position", type=int, default=0)
    a = ap.parse_args()
    # Third Party
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = (
        AutoModelForCausalLM.from_pretrained(
            a.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    cfg = model.config
    H = cfg.num_key_value_heads
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    if D != 128:
        raise SystemExit(f"fp8lin needs head_dim 128, {a.model} has {D}")
    L = cfg.num_hidden_layers
    docs = _docs(tok, a.docs + 16, a.seq_len, a.text)
    calib, train = docs[:16], docs[16:]
    if a.kv == "bf16":
        scales = amax_scales(model, calib, a.headroom)
        codes = capture(model, train, scales, fp8_cache=False)
    else:
        scales = {f"L{layer}.{n}": 1.0 for layer in range(L) for n in "kv"}
        codes = capture(model, train, scales, fp8_cache=True)
    inv = model.model.rotary_emb.inv_freq.detach().double()
    th = torch.cat([inv, inv]).cuda()
    del model
    torch_dev.empty_cache()
    pos = torch.arange(a.seq_len, device="cuda").repeat(len(train))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for layer in range(L):
        cur = codes[layer].cuda()
        prev = codes[layer - 1].cuda() if layer else None
        np.savez(out / f"layer_{layer:03d}.npz", **fit_layer(cur, prev, pos, th, H, D))
        print(f"fp8lin fit: layer {layer + 1}/{L}", flush=True)
    meta = {
        "format": "fp8lin-coder-v1",
        "model": a.model,
        "kv": a.kv,
        "num_layers": L,
        "num_kv_heads": H,
        "head_dim": D,
        "kv_scale": {
            "k": [scales[f"L{i}.k"] for i in range(L)],
            "v": [scales[f"L{i}.v"] for i in range(L)],
        },
        "width_index": 5,
        "lane_tokens": 128,
        "max_position": a.max_position
        or int(getattr(cfg, "max_position_embeddings", 32768)),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
