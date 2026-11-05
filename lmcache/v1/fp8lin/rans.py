# SPDX-License-Identifier: Apache-2.0
"""GPU rANS entropy coder for E4M3 codes under per-value Gaussian models (Triton).

Each value has a predicted mean ``mu`` and an inverse scale ``r = 1/s`` (FP8 units,
both rounded to fp16 by the caller). Its 254 non-NaN E4M3 codes, ordered by value
(-0 just before +0), get the integer CDF::

    CDF(i) = floor(F(edge_i) * (M - 254)) + i,   F = Gaussian CDF,   M = 2**16

so every code keeps frequency >= 1 and any CDF entry costs one evaluation: there is
no pass over the alphabet to normalize. Encoder and decoder evaluate ``F`` with the
same kernel code, so their integer CDFs agree bit for bit on one GPU type.

Layout. A coded unit is a sequence of ``[N, 128]`` blocks decoded in order (the 16
head blocks of a layer); block ``b``'s predictions may depend on blocks before it.
Lane ``(q, j)`` is channel ``j`` of tokens ``[q*T, (q+1)*T)`` in every block. Its
32-bit state persists across the blocks of a layer (16-bit renormalization, at most
one word per symbol), and is flushed into two words at the end of the layer. A
layer's words are stored lane-major in decode order with a 16-bit count per lane,
so lanes of consecutive token ranges are contiguous and can be split or
concatenated (e.g. per LMCache chunk).
"""

# Standard
import math

# Third Party
import torch
import triton
import triton.language as tl

M_BITS = 16
M = 1 << M_BITS
NSYM = 254
BW = 128  # channels per block


def tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (code per symbol uint8 [254], symbol per code int32 [256] (-1 for NaN),
    bin edges in FP8 units float32 [255])."""
    vals = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    codes = [c for c in range(256) if not math.isnan(vals[c].item())]
    order = sorted(codes, key=lambda c: (vals[c].item(), 0 if c >= 128 else 1))
    v = torch.tensor([vals[c].item() for c in order], dtype=torch.float64)
    edges = torch.empty(NSYM + 1, dtype=torch.float64)
    edges[0], edges[-1] = -math.inf, math.inf
    edges[1:-1] = (v[1:] + v[:-1]) / 2  # 0 between -0 and +0
    sym = torch.full((256,), -1, dtype=torch.int32)
    sym[torch.tensor(order)] = torch.arange(NSYM, dtype=torch.int32)
    return (
        torch.tensor(order, dtype=torch.uint8, device=device),
        sym.to(device),
        edges.float().to(device),
    )


@triton.jit
def _erf(z):
    # Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7): one exp2 and one division.
    a = tl.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = t * (
        0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    y = 1.0 - y * tl.exp2(-a * a * 1.4426950408889634)
    return tl.where(z < 0, -y, y)


@triton.jit
def _cdf(i, mu, r, edges):
    """Integer CDF (int32) at symbol index i (0..254) for per-lane mu and r = 1/s."""
    e = tl.load(edges + i)
    F = 0.5 + 0.5 * _erf((e - mu) * r * 0.7071067811865476)
    F = tl.minimum(tl.maximum(F, 0.0), 1.0)
    c = tl.floor(F * 65282.0).to(tl.int32) + i  # 65282 = M - 254
    c = tl.where(i <= 0, 0, c)
    return tl.where(i >= 254, 65536, c)


@triton.jit
def _guess(mu):
    """Symbol index of the E4M3 code nearest mu, from its float bits."""
    am = tl.abs(mu)
    bits = am.to(tl.int32, bitcast=True)
    nrm = (((bits >> 23) & 0xFF) - 120) * 8 + ((bits >> 20) & 7) + ((bits >> 19) & 1)
    sub = (am * 512.0 + 0.5).to(tl.int32)
    mag = tl.minimum(tl.where(am < 0.015625, sub, nrm), 126)
    g = tl.where(mu < 0, 126 - mag, 127 + mag)
    return tl.minimum(tl.maximum(g, 0), 253)


@triton.jit(do_not_specialize=["N", "T", "CAP"])
def _enc_block(mu_ptr, r_ptr, code_ptr, sym_tab, x_ptr, p_ptr, emis_ptr, edges, N, T,
               CAP, LW: tl.constexpr):  # fmt: skip
    """Encode one [N, 128] block (fp16 mu, r; uint8 codes), tokens in reverse."""
    ch = tl.program_id(1) * LW + tl.arange(0, LW)
    q = tl.program_id(0)
    lane = (q * 128 + ch).to(tl.int64)
    x = tl.load(x_ptr + lane)
    ptr = tl.load(p_ptr + lane)
    t0 = q * T
    nt = tl.minimum(T, N - t0)
    for k in range(0, nt):
        idx = (t0 + nt - 1 - k).to(tl.int64) * 128 + ch
        mu = tl.load(mu_ptr + idx).to(tl.float32)
        r = tl.load(r_ptr + idx).to(tl.float32)
        i = tl.load(sym_tab + tl.load(code_ptr + idx).to(tl.int32))
        lo = _cdf(i, mu, r, edges)
        f = (_cdf(i + 1, mu, r, edges) - lo).to(tl.uint32)
        emit = x >= (f.to(tl.int64) << 16)
        tl.store(emis_ptr + lane * CAP + ptr, (x & 0xFFFF).to(tl.int16), mask=emit)
        ptr = tl.where(emit, ptr + 1, ptr)
        x = tl.where(emit, x >> 16, x)
        xu = x.to(tl.uint32)  # x < 2**32: 32-bit division (64-bit is emulated)
        qd = xu // f
        x = (qd * 65536 + (xu - qd * f) + lo.to(tl.uint32)).to(tl.int64)
    tl.store(x_ptr + lane, x)
    tl.store(p_ptr + lane, ptr)


@triton.jit(do_not_specialize=["N", "T", "n_words"])
def _dec_block(mu_ptr, r_ptr, out_ptr, order, x_ptr, p_ptr, words, edges, N, T,
               n_words, GALLOP: tl.constexpr, LW: tl.constexpr):  # fmt: skip
    """Decode one [N, 128] block into uint8 codes; p_ptr: absolute read positions."""
    ch = tl.program_id(1) * LW + tl.arange(0, LW)
    q = tl.program_id(0)
    lane = (q * 128 + ch).to(tl.int64)
    x = tl.load(x_ptr + lane).to(tl.uint32)
    ptr = tl.load(p_ptr + lane)
    t0 = q * T
    nt = tl.minimum(T, N - t0)
    for k in range(0, nt):
        idx = (t0 + k).to(tl.int64) * 128 + ch
        mu = tl.load(mu_ptr + idx).to(tl.float32)
        r = tl.load(r_ptr + idx).to(tl.float32)
        slot = (x & 0xFFFF).to(tl.int32)
        g = _guess(mu)
        lo_c = _cdf(g, mu, r, edges)
        hi_c = _cdf(g + 1, mu, r, edges)
        for _ in tl.static_range(GALLOP):  # one new CDF entry per step
            up = slot >= hi_c
            dn = slot < lo_c
            g2 = tl.where(up, g + 1, tl.where(dn, g - 1, g))
            cn = _cdf(tl.where(up, g2 + 1, g2), mu, r, edges)
            lo_n = tl.where(up, hi_c, tl.where(dn, cn, lo_c))
            hi_n = tl.where(up, cn, tl.where(dn, lo_c, hi_c))
            lo_c, hi_c, g = lo_n, hi_n, g2
        miss = (slot < lo_c) | (slot >= hi_c)
        if tl.sum(miss.to(tl.int32), 0) > 0:  # rare far values: binary search
            a = tl.zeros((LW,), tl.int32)
            b = tl.full((LW,), 254, tl.int32)
            ca = tl.zeros((LW,), tl.int32)
            cb = tl.full((LW,), 65536, tl.int32)
            for _ in tl.static_range(8):
                m = (a + b) // 2
                cm = _cdf(m, mu, r, edges)
                go = slot >= cm
                a = tl.where(go, m, a)
                ca = tl.where(go, cm, ca)
                b = tl.where(go, b, m)
                cb = tl.where(go, cb, cm)
            g = tl.where(miss, a, g)
            lo_c = tl.where(miss, ca, lo_c)
            hi_c = tl.where(miss, cb, hi_c)
        x = (hi_c - lo_c).to(tl.uint32) * (x >> 16) + (slot - lo_c).to(tl.uint32)
        need = x < 65536
        ok = need & (ptr < n_words)  # corrupt input can never read out of bounds
        wv = tl.load(words + ptr, mask=ok, other=0).to(tl.uint32) & 0xFFFF
        x = tl.where(need, (x << 16) | wv, x)
        ptr = tl.where(need, ptr + 1, ptr)
        tl.store(out_ptr + idx, tl.load(order + g))
    tl.store(x_ptr + lane, x.to(tl.int64))
    tl.store(p_ptr + lane, ptr)


class Tables:
    """Device-resident symbol tables shared by encoders and decoders."""

    def __init__(self, device: torch.device):
        self.order, self.sym, self.edges = tables(device)
        self.device = device


class LayerEncoder:
    """Encode one layer: push its 16 ``[N, 128]`` blocks in reverse decode order, then
    ``finish()``. Lanes cover ``T`` consecutive tokens at one channel."""

    def __init__(self, tabs: Tables, n_tokens: int, lane_tokens: int, blocks: int):
        self.t, self.N, self.T = tabs, n_tokens, lane_tokens
        self.nq = triton.cdiv(n_tokens, lane_tokens)
        lanes = self.nq * BW
        dev = tabs.device
        self.cap = blocks * lane_tokens + 2
        self.x = torch.full((lanes,), M, dtype=torch.int64, device=dev)
        self.p = torch.zeros(lanes, dtype=torch.int64, device=dev)
        self.emis = torch.empty(lanes * self.cap, dtype=torch.int16, device=dev)

    def push(self, mu16: torch.Tensor, r16: torch.Tensor, codes: torch.Tensor) -> None:
        """Encode one block. mu16, r16: fp16 [N, 128]; codes: uint8 [N, 128]."""
        _enc_block[(self.nq, BW // 32)](
            mu16, r16, codes.contiguous(), self.t.sym, self.x, self.p, self.emis,
            self.t.edges, self.N, self.T, self.cap, LW=32, num_warps=1,
        )  # fmt: skip

    def finish(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Flush the states (two words first in every lane, decode order).

        Returns:
            (words int16 [n] holding uint16 values, counts int32 [lanes]).
        """
        hi = (self.x >> 16).to(torch.int16)
        lo = (self.x & 0xFFFF).to(torch.int16)
        n = self.p
        E = self.emis.view(-1, self.cap)
        j = torch.arange(self.cap, device=n.device)[None]
        w = torch.gather(E, 1, (n[:, None] - 1 - j).clamp_min(0))  # reverse emission
        w = torch.cat([hi[:, None], lo[:, None], w], 1)
        keep = torch.cat(
            [torch.ones_like(w[:, :2], dtype=torch.bool), j < n[:, None]], 1
        )
        return w[keep], (n + 2).to(torch.int32)


class LayerDecoder:
    """Decode one layer block by block (``pull``), from its words and lane counts."""

    def __init__(self, tabs: Tables, n_tokens: int, lane_tokens: int,
                 words: torch.Tensor, counts: torch.Tensor):  # fmt: skip
        """``counts`` must already be validated (each >= 2, summing to the number
        of words); see ``ChunkCodec``."""
        self.t, self.N, self.T = tabs, n_tokens, lane_tokens
        self.nq = triton.cdiv(n_tokens, lane_tokens)
        self.words = words
        self.p = torch.cumsum(counts.long(), 0) - counts.long()
        hi = words[self.p].long() & 0xFFFF
        lo = words[self.p + 1].long() & 0xFFFF
        self.x = (hi << 16) | lo
        self.p = self.p + 2

    def pull(self, mu16: torch.Tensor, r16: torch.Tensor, out: torch.Tensor) -> None:
        """Decode one block into ``out`` (uint8 [N, 128])."""
        _dec_block[(self.nq, BW // 32)](
            mu16, r16, out, self.t.order, self.x, self.p, self.words, self.t.edges,
            self.N, self.T, self.words.numel(), GALLOP=2, LW=32, num_warps=1,
        )  # fmt: skip


def inv_scale(s: torch.Tensor) -> torch.Tensor:
    """The coded 1/s: fp32 reciprocal rounded to fp16 (shared by both sides)."""
    return (1 / s.float()).half().contiguous()
