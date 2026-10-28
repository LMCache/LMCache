# SPDX-License-Identifier: Apache-2.0
"""Lossless linear predictive coder for FP8 (E4M3) KV chunks, on the GPU.

For each layer, the 16 head blocks (K heads, then V heads, 128 channels each) are
coded in order. Block ``b`` of a token is predicted by a fitted least-squares map
from the already coded blocks of the same token (K un-rotated out of RoPE) and from
the same token's decoded previous layer. Its residual scale is a per-token scale
``sig * exp(0.5 * (w0 + w1 * g1))``, with ``g1`` the log residual energy of the
token's earlier blocks. Codes are entropy coded by :mod:`.rans`.

Chunks carry no positions in LMCache, but K is stored after RoPE. The encoder
recovers each chunk's absolute start position from the keys themselves (coding the
first tokens under every candidate start costs ~2x more bits at a wrong start) and
stores it in the header; the decoder reads it.

Encoder and decoder run the same kernels on the same inputs, so their predictions
match bit for bit on one GPU type, and every kernel computes a token's prediction
from that token's data alone (including the matmuls, see :func:`mm`), so a chunk
decodes identically whichever chunks it is batched with. Every blob carries a
checksum of the codes; a blob that fails it on decode yields ``None`` (a cache
miss) instead of bad KV.

Coder directory (written by the fitting tool)::

    meta.json         {"format": "fp8lin-coder-v1", "num_layers", "num_kv_heads",
                       "head_dim", "kv_scale": {"k": [...], "v": [...]},
                       "width_index", "lane_tokens", "max_position"}
    layer_000.npz     fitted coder of layer 0 (sd, sig, Wh, cols<b>, W<b>, th, ...)
    ...
"""

# Standard
from dataclasses import dataclass
from pathlib import Path
import json
import struct

# Third Party
import numpy as np
import torch
import triton
import triton.language as tl

# First Party
from lmcache import torch_dev

# Local
from .rans import BW, LayerDecoder, LayerEncoder, Tables

WIDTHS = [0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.3, 1.45]
MAGIC = b"F8L1"
KIND_CODED, KIND_RAW = 0, 1
_HEAD = struct.Struct("<4sBBBBHHHHIIQ")  # magic ver kind wK wV L H D lane T start sum
BT = 32  # tokens per program in the fused kernels


def e4m3_values(device: torch.device) -> torch.Tensor:
    """FP8 value of every E4M3 code (NaN codes map to 0), float32 [256]."""
    v = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    return torch.nan_to_num(v, nan=0.0).to(device)


@triton.jit(do_not_specialize=["n"])
def _checksum_kernel(x_ptr, part_ptr, n, BLOCK: tl.constexpr):
    """Partial sums of (x_i + 1) * h(i) over one block (int64, wrapping)."""
    pid = tl.program_id(0)
    i = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < n, other=0).to(tl.int64)
    h = (i * -7046029254386353131) ^ (i >> 7)  # 0x9E3779B97F4A7C15 as int64
    v = tl.where(i < n, (x + 1) * (h | 1), 0)
    tl.store(part_ptr + pid, tl.sum(v, 0))


def checksums(chunks: list[torch.Tensor]) -> list[int]:
    """Order-independent 64-bit checksums of uint8 code tensors (integer sums, so
    deterministic on any device)."""
    out = []
    block = 4096
    for c in chunks:
        x = c.reshape(-1)
        nb = triton.cdiv(x.numel(), block)
        part = torch.empty(nb, dtype=torch.int64, device=x.device)
        _checksum_kernel[(nb,)](x, part, x.numel(), BLOCK=block)
        out.append(part.sum())
    return (
        [int(v) & 0xFFFFFFFFFFFFFFFF for v in torch.stack(out).tolist()] if out else []
    )


def checksum(codes: torch.Tensor) -> int:
    """64-bit checksum of one uint8 code tensor (see :func:`checksums`)."""
    return checksums([codes])[0]


class LayerCoder:
    """Fitted coder of one layer (cross-head + previous layer)."""

    def __init__(self, path: Path, device: torch.device):
        z = np.load(path)
        self.H, self.D, self.C = int(z["H"]), int(z["D"]), int(z["C"])
        H, D, C = self.H, self.D, self.C
        if D != BW:
            raise ValueError(f"{path}: fp8lin needs head_dim {BW}, got {D}")
        self.th = torch.tensor(z["th"], dtype=torch.float64, device=device)
        go = {str(g): int(o) for g, o in zip(z["groups"], z["goff"], strict=True)}
        sd = torch.tensor(z["sd"], dtype=torch.float32, device=device)
        sig = torch.tensor(z["sig"], dtype=torch.float32, device=device)
        Wh = torch.tensor(z["Wh"], dtype=torch.float32, device=device)
        nz = len(sd) + 1
        if go["curK"] != 0 or go["curV"] != C:
            raise ValueError(f"{path}: unexpected group layout {go}")
        self.m: list[int] = []
        self.Wc: list[torch.Tensor | None] = []
        self.bias: list[torch.Tensor] = []
        self.sig: list[torch.Tensor] = []
        self.wh: list[tuple[float, float]] = []
        Wp = torch.zeros(2 * C, 2 * C, device=device)
        for b in range(2 * H):
            cols = torch.tensor(z[f"cols{b}"], dtype=torch.long)
            W = torch.tensor(z[f"W{b}"], dtype=torch.float32, device=device)
            tgt = slice(b * D, (b + 1) * D)
            sd_t = sd[tgt]
            mb = int((cols < 2 * C).sum())
            if not torch.equal(cols[:mb], torch.arange(mb)) or cols[-1] != nz - 1:
                raise ValueError(f"{path}: block {b} columns are not supported")
            rest = cols[mb:-1]
            if len(rest):
                p0 = go["p"]
                if not torch.equal(rest, torch.arange(p0, p0 + 2 * C)):
                    raise ValueError(f"{path}: block {b} side groups must be [p]")
                Wp[:, tgt] = W[mb : mb + 2 * C] / sd[p0 : p0 + 2 * C, None] * sd_t[None]
            self.m.append(mb)
            self.Wc.append(
                (W[:mb] / sd[:mb, None] * sd_t[None]).half().contiguous()
                if mb
                else None
            )
            self.bias.append((W[-1] * sd_t).contiguous())
            self.sig.append(sig[tgt].contiguous())
            self.wh.append((float(Wh[b, 0]), float(Wh[b, 1])))
        self.Wp = Wp.half().contiguous()


@triton.jit(do_not_specialize=["M"])
def _mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    sam,
    sak,
    sbk,
    sbn,
    scm,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """C = A @ B (fp16 in, fp32 accumulate, fp16 out) with a fixed tiling and a
    sequential K loop, so every output row depends only on its own inputs."""
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    rm64 = rm.to(tl.int64)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        a = tl.load(
            a_ptr + rm64[:, None] * sam + kk[None, :] * sak,
            mask=(rm[:, None] < M) & (kk[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + kk[:, None] * sbk + rn[None, :] * sbn,
            mask=(kk[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)
    tl.store(
        c_ptr + rm64[:, None] * scm + rn[None, :],
        acc.to(tl.float16),
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


def mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batch-invariant fp16 matmul: row i of ``a @ b`` is bit-identical whatever the
    number of rows of ``a`` (cuBLAS picks algorithms by shape, which would make
    predictions depend on how chunks are batched)."""
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, dtype=torch.float16, device=a.device)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))
    _mm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        BM=64,
        BN=128,
        BK=32,
        num_warps=4,
    )
    return c


def rope_tables(
    pos: torch.Tensor, th: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [N, D] float32 for positions pos [N] and per-dim inverse freqs th [D]."""
    a = pos.double()[:, None] * th[None]
    return a.cos().float().contiguous(), a.sin().float().contiguous()


@triton.jit(do_not_specialize=["N", "nb"])
def _prep(
    base_ptr,
    intra_ptr,
    bias_ptr,
    cos_ptr,
    sin_ptr,
    sig_ptr,
    racc_ptr,
    mu_out,
    r_out,
    N,
    nb,
    wh0,
    wh1,
    width,
    base_stride,
    base_off,
    HAS_INTRA: tl.constexpr,
    ROT: tl.constexpr,
    BT: tl.constexpr,
):
    """Prediction epilogue of one block: bias, RoPE (K), per-token scale -> mu, 1/s."""
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    c = tl.arange(0, 128)
    tm = t < N
    m2 = tm[:, None]
    t64 = t.to(tl.int64)
    mu_u = tl.load(
        base_ptr + t64[:, None] * base_stride + base_off + c[None, :],
        mask=m2,
        other=0.0,
    ).to(tl.float32)
    mu_u += tl.load(bias_ptr + c)[None, :]
    if HAS_INTRA:
        mu_u += tl.load(
            intra_ptr + t64[:, None] * 128 + c[None, :], mask=m2, other=0.0
        ).to(tl.float32)
    if ROT:
        cp = (c + 64) % 128
        mu_p = tl.load(
            base_ptr + t64[:, None] * base_stride + base_off + cp[None, :],
            mask=m2,
            other=0.0,
        ).to(tl.float32)
        mu_p += tl.load(bias_ptr + cp)[None, :]
        if HAS_INTRA:
            mu_p += tl.load(
                intra_ptr + t64[:, None] * 128 + cp[None, :], mask=m2, other=0.0
            ).to(tl.float32)
        cs = tl.load(cos_ptr + t64[:, None] * 128 + c[None, :], mask=m2, other=1.0)
        sn = tl.load(sin_ptr + t64[:, None] * 128 + c[None, :], mask=m2, other=0.0)
        sgn = tl.where(c < 64, -1.0, 1.0)
        mu = mu_u * cs + sgn[None, :] * mu_p * sn
    else:
        mu = mu_u
    racc = tl.load(racc_ptr + t, mask=tm, other=0.0)
    g1 = tl.log(racc / tl.maximum(nb, 1).to(tl.float32) + 1e-3)
    g1 = tl.where(nb > 0, g1, 0.0)
    s = tl.load(sig_ptr + c)[None, :] * tl.exp(0.5 * (wh0 + wh1 * g1))[:, None] * width
    off = t64[:, None] * 128 + c[None, :]
    tl.store(mu_out + off, mu.to(tl.float16), mask=m2)
    tl.store(r_out + off, (1.0 / s).to(tl.float16), mask=m2)


@triton.jit(do_not_specialize=["N"])
def _post(
    code_ptr,
    mu_ptr,
    sig_ptr,
    cos_ptr,
    sin_ptr,
    tab_ptr,
    racc_ptr,
    cur_ptr,
    N,
    cur_stride,
    cur_off,
    ROT: tl.constexpr,
    BT: tl.constexpr,
):
    """After a block is coded: residual energy (for later scales) and its un-rotated
    features (fp16) for the cross-head and next-layer predictions."""
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    c = tl.arange(0, 128)
    tm = t < N
    m2 = tm[:, None]
    t64 = t.to(tl.int64)
    off = t64[:, None] * 128 + c[None, :]
    code = tl.load(code_ptr + off, mask=m2, other=0).to(tl.int32)
    val = tl.load(tab_ptr + code)
    mu = tl.load(mu_ptr + off, mask=m2, other=0.0).to(tl.float32)
    e = (val - mu) / tl.load(sig_ptr + c)[None, :]
    r = tl.sum(e * e, 1) / 128.0
    tl.store(racc_ptr + t, tl.load(racc_ptr + t, mask=tm, other=0.0) + r, mask=tm)
    if ROT:
        cp = (c + 64) % 128
        pc = tl.load(code_ptr + t64[:, None] * 128 + cp[None, :], mask=m2, other=0)
        vp = tl.load(tab_ptr + pc.to(tl.int32))
        cs = tl.load(cos_ptr + off, mask=m2, other=1.0)
        sn = tl.load(sin_ptr + off, mask=m2, other=0.0)
        sgn = tl.where(c < 64, -1.0, 1.0)
        out = val * cs - sgn[None, :] * vp * sn  # un-rotate
    else:
        out = val
    dst = cur_ptr + t64[:, None] * cur_stride + cur_off + c[None, :]
    tl.store(dst, out.to(tl.float16), mask=m2)


@dataclass
class _Group:
    """Chunks coded in one GPU pass (same token count)."""

    idx: list[int]
    T: int


class ChunkCodec:
    """Encode/decode LMCache KV chunks given as uint8 E4M3 codes [L, T, 2C].

    Args:
        coder_dir: Directory written by the fitting tool (see module docstring).
        device: CUDA device for coding.
    """

    def __init__(self, coder_dir: str | Path, device: torch.device | str = "cuda"):
        d = Path(coder_dir)
        self.meta = json.loads((d / "meta.json").read_text())
        if self.meta.get("format") != "fp8lin-coder-v1":
            raise ValueError(f"{d}: not an fp8lin coder directory")
        self.device = torch.device(device)
        self.L = int(self.meta["num_layers"])
        self.layers = [
            LayerCoder(d / f"layer_{i:03d}.npz", self.device) for i in range(self.L)
        ]
        self.H, self.D, self.C = self.layers[0].H, self.layers[0].D, self.layers[0].C
        self.lane = int(self.meta.get("lane_tokens", 128))
        self.width_index = int(self.meta.get("width_index", 5))
        self.max_position = int(self.meta.get("max_position", 32768))
        self.tabs = Tables(self.device)
        self.vals = e4m3_values(self.device)
        self.th = self.layers[0].th
        self._staging: torch.Tensor | None = None  # pinned upload buffer

    # ------------------------------------------------------------------ coding

    def _code_layer(
        self,
        lc: LayerCoder,
        prev16: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        widths: tuple[int, int],
        true: torch.Tensor | None = None,
        dec: LayerDecoder | None = None,
        out: torch.Tensor | None = None,
    ):
        """Code one layer for N tokens. Encoder: ``true`` [N, 2C] -> blocks
        (mu16, r16, codes). Decoder: ``dec`` fills ``out`` [N, 2C]. Returns
        (blocks, un-rotated fp16 features [N, 2C])."""
        N, H, D, C = prev16.shape[0], self.H, self.D, self.C
        base = mm(prev16, lc.Wp)  # previous-layer part of every block, fp16 [N, 2C]
        racc = torch.zeros(N, device=self.device)
        cur = torch.zeros(N, 2 * C, dtype=torch.float16, device=self.device)
        grid = (triton.cdiv(N, BT),)
        blocks = []
        for b in range(2 * H):
            tgt = slice(b * D, (b + 1) * D)
            Wc = lc.Wc[b]
            intra = mm(cur[:, : lc.m[b]], Wc) if Wc is not None else base
            mu16 = torch.empty(N, D, dtype=torch.float16, device=self.device)
            r16 = torch.empty_like(mu16)
            w0, w1 = lc.wh[b]
            _prep[grid](
                base,
                intra,
                lc.bias[b],
                cos,
                sin,
                lc.sig[b],
                racc,
                mu16,
                r16,
                N,
                b,
                w0,
                w1,
                WIDTHS[widths[b >= H]],
                2 * C,
                b * D,
                HAS_INTRA=Wc is not None,
                ROT=b < H,
                BT=BT,
            )
            if true is not None:
                cb = true[:, tgt].contiguous()
                blocks.append((mu16, r16, cb))
            else:
                assert dec is not None and out is not None
                cb = torch.empty(N, D, dtype=torch.uint8, device=self.device)
                dec.pull(mu16, r16, cb)
                out[:, tgt] = cb
            _post[grid](
                cb,
                mu16,
                lc.sig[b],
                cos,
                sin,
                self.vals,
                racc,
                cur,
                N,
                2 * C,
                b * D,
                ROT=b < H,
                BT=BT,
            )
        return blocks, cur

    def _groups(self, Ts: list[int]) -> list[_Group]:
        """Batch chunks of equal length whose lanes align; others go alone."""
        groups: dict[int, _Group] = {}
        alone: list[_Group] = []
        for i, T in enumerate(Ts):
            if T % self.lane == 0:
                groups.setdefault(T, _Group([], T)).idx.append(i)
            else:
                alone.append(_Group([i], T))
        return list(groups.values()) + alone

    @torch.no_grad()
    def encode(self, chunks: list[torch.Tensor], starts: list[int]) -> list[bytes]:
        """Encode chunks (uint8 [L, T, 2C] on the codec device) at absolute start
        positions. Returns one self-describing blob per chunk."""
        out: list[bytes | None] = [None] * len(chunks)
        widths = (self.width_index, self.width_index)
        for g in self._groups([c.shape[1] for c in chunks]):
            cs = [chunks[i] for i in g.idx]
            if any(bool((c & 0x7F).eq(0x7F).any()) for c in cs):  # NaN codes: raw
                for i in g.idx:
                    out[i] = self._raw_blob(chunks[i], starts[i])
                continue
            n, T = len(g.idx), g.T
            pos = torch.cat(
                [torch.arange(T, device=self.device) + int(starts[i]) for i in g.idx]
            )
            cos, sin = rope_tables(pos, self.th)
            prev = torch.zeros(
                n * T, 2 * self.C, dtype=torch.float16, device=self.device
            )
            per_layer = []  # (words, counts) per layer for all chunks of the group
            for li, lc in enumerate(self.layers):
                true = torch.cat([c[li] for c in cs])
                blocks, prev = self._code_layer(lc, prev, cos, sin, widths, true=true)
                enc = LayerEncoder(self.tabs, n * T, self.lane, 2 * self.H)
                for mu16, r16, cb in reversed(blocks):
                    enc.push(mu16, r16, cb)
                per_layer.append(enc.finish())
            lanes = (T // self.lane) * BW
            nl = (n * T + self.lane - 1) // self.lane * BW  # lanes in the pass
            host = [
                (w.cpu().numpy(), c.to(torch.int16).cpu().numpy()) for w, c in per_layer
            ]
            sums = checksums(cs)
            for j, i in enumerate(g.idx):
                lo, hi = (j * lanes, (j + 1) * lanes) if n > 1 else (0, nl)
                blob = self._coded_blob(host, lo, hi, sums[j], starts[i], T)
                if len(blob) > self.max_blob_size(T):  # never larger than raw
                    blob = self._raw_blob(chunks[i], starts[i])
                out[i] = blob
        return [b for b in out if b is not None]

    def blob_length(self, prefix: bytes) -> int:
        """Total length of the blob that starts with ``prefix`` (its header and, for
        coded blobs, the per-layer word counts), so blobs can be read out of larger
        buffers.

        Raises:
            ValueError: ``prefix`` does not start an fp8lin blob for this coder.
        """
        h = self._parse(prefix)
        if h["kind"] == KIND_RAW:
            return h["body"] + self.L * h["T"] * 2 * self.C
        if len(prefix) < h["body"] + 4 * self.L:
            raise ValueError("fp8lin: prefix too short")
        nw = struct.unpack_from(f"<{self.L}I", prefix, h["body"])
        lanes = -(-h["T"] // self.lane) * BW
        return h["body"] + 4 * self.L + 2 * sum(lanes + n for n in nw)

    def header_size(self) -> int:
        """Bytes needed by :meth:`blob_length`."""
        return _HEAD.size + 4 * self.L

    def max_blob_size(self, T: int) -> int:
        """Upper bound on a blob for a T-token chunk: header plus raw codes."""
        return _HEAD.size + self.L * T * 2 * self.C

    def _coded_blob(
        self,
        host: list[tuple[np.ndarray, np.ndarray]],
        lo: int,
        hi: int,
        csum: int,
        start: int,
        T: int,
    ) -> bytes:
        """Blob of lanes [lo, hi) of every layer: header, word counts, then per layer
        the lane counts (u16) and words (u16)."""
        parts, nwords = [], []
        for words, counts in host:
            off = np.concatenate([[0], np.cumsum(counts.astype(np.int64))])
            w = words[off[lo] : off[hi]]
            parts += [counts[lo:hi].tobytes(), w.tobytes()]
            nwords.append(w.size)
        head = self._header(KIND_CODED, T, start, csum)
        return head + struct.pack(f"<{self.L}I", *nwords) + b"".join(parts)

    def _raw_blob(self, codes: torch.Tensor, start: int) -> bytes:
        T = codes.shape[1]
        head = self._header(KIND_RAW, T, start, checksums([codes])[0])
        return head + codes.contiguous().cpu().numpy().tobytes()

    def _header(self, kind: int, T: int, start: int, csum: int) -> bytes:
        w = self.width_index
        return _HEAD.pack(
            MAGIC, 1, kind, w, w, self.L, self.H, self.D, self.lane, T, start, csum
        )

    @torch.no_grad()
    def decode(self, blobs: list[bytes]) -> list[torch.Tensor | None]:
        """Decode blobs from :meth:`encode` into uint8 codes [L, T, 2C] on the codec
        device. A blob whose decoded codes do not match its checksum yields ``None``
        (the caller treats it as a cache miss).

        Raises:
            ValueError: A blob is not an fp8lin blob for this coder.
        """
        heads = [self._parse(b) for b in blobs]
        out: list[torch.Tensor | None] = [None] * len(blobs)
        coded = [i for i, h in enumerate(heads) if h["kind"] == KIND_CODED]
        for i, h in enumerate(heads):
            if h["kind"] == KIND_RAW:
                if len(blobs[i]) != h["body"] + self.L * h["T"] * 2 * self.C:
                    raise ValueError("fp8lin: raw blob has the wrong size")
                raw = np.frombuffer(blobs[i], np.uint8, offset=h["body"])
                out[i] = (
                    torch.from_numpy(raw.copy())
                    .to(self.device)
                    .view(self.L, h["T"], 2 * self.C)
                )
        dev_bufs = self._to_device([blobs[i] for i in coded], [heads[i] for i in coded])
        buf_of = dict(zip(coded, dev_bufs, strict=True))
        groups = self._groups([heads[i]["T"] for i in coded])
        for g in groups:
            idx = [coded[k] for k in g.idx]
            n, T = len(idx), g.T
            bodies = [self._body(blobs[i], heads[i], buf_of[i]) for i in idx]
            pos = torch.cat(
                [torch.arange(T, device=self.device) + heads[i]["start"] for i in idx]
            )
            cos, sin = rope_tables(pos, self.th)
            widths = (heads[idx[0]]["wK"], heads[idx[0]]["wV"])
            prev = torch.zeros(
                n * T, 2 * self.C, dtype=torch.float16, device=self.device
            )
            res = torch.empty(
                self.L, n * T, 2 * self.C, dtype=torch.uint8, device=self.device
            )
            for li, lc in enumerate(self.layers):
                counts = torch.cat([bd[li][0] for bd in bodies])
                words = torch.cat([bd[li][1] for bd in bodies])
                dec = LayerDecoder(self.tabs, n * T, self.lane, words, counts)
                _, prev = self._code_layer(
                    lc, prev, cos, sin, widths, dec=dec, out=res[li]
                )
            for j, i in enumerate(idx):
                out[i] = res[:, j * T : (j + 1) * T]
        ok = [i for i in range(len(out)) if out[i] is not None]
        sums = checksums([out[i] for i in ok])  # type: ignore[misc]
        for i, s_ in zip(ok, sums, strict=True):
            if s_ != heads[i]["sum"]:
                out[i] = None
        return out

    def _parse(self, blob: bytes) -> dict:
        if len(blob) < _HEAD.size:
            raise ValueError("fp8lin: blob too short")
        f = _HEAD.unpack_from(blob)
        magic, ver, kind, wK, wV, L, H, D, lane, T, start, csum = f
        if magic != MAGIC or ver != 1:
            raise ValueError("fp8lin: not an fp8lin blob")
        if (L, H, D, lane) != (self.L, self.H, self.D, self.lane):
            raise ValueError(f"fp8lin: blob for L={L} H={H} D={D} lane={lane}")
        if (
            kind not in (KIND_CODED, KIND_RAW)
            or T == 0
            or wK >= len(WIDTHS)
            or wV >= len(WIDTHS)
        ):
            raise ValueError("fp8lin: corrupt header")
        return dict(
            kind=kind, wK=wK, wV=wV, T=T, start=start, sum=csum, body=_HEAD.size
        )

    def _to_device(self, blobs: list[bytes], heads: list[dict]) -> list[torch.Tensor]:
        """Copy the int16 payloads of coded blobs to the device in one pinned transfer.

        Returns:
            Per blob, its payload (per layer: lane counts, then words) as an int16
            tensor on the device.
        """
        if not blobs:
            return []
        for b, h in zip(blobs, heads, strict=True):
            self._validate(b, h)
        offs = [h["body"] + 4 * self.L for h in heads]
        sizes = [len(b) - o for b, o in zip(blobs, offs, strict=True)]
        total = sum(sizes)
        if self._staging is None or self._staging.numel() < total:  # grow, reuse
            self._staging = torch.empty(
                total * 5 // 4, dtype=torch.uint8, pin_memory=True
            )
        host = self._staging[:total]
        hv = host.numpy()
        o = 0
        for b, off, n in zip(blobs, offs, sizes, strict=True):
            hv[o : o + n] = np.frombuffer(b, dtype=np.uint8, offset=off)
            o += n
        dev = host.to(self.device, non_blocking=True).view(torch.int16)
        torch_dev.current_stream(self.device).synchronize()  # staging is reused
        out, o = [], 0
        for n in sizes:  # every payload has an even size
            out.append(dev[o // 2 : (o + n) // 2])
            o += n
        return out

    def _validate(self, blob: bytes, h: dict) -> None:
        """Check a coded blob's structure on the host before anything touches the
        GPU: per layer, every lane count must be in [2, cap] and the counts must sum
        to the layer's word count, and the sizes must add up to the blob.

        Raises:
            ValueError: The blob is truncated or inconsistent.
        """
        lanes = -(-h["T"] // self.lane) * BW
        cap = 2 * self.H * self.lane + 2
        o = h["body"]
        if len(blob) < o + 4 * self.L:
            raise ValueError("fp8lin: truncated blob")
        nw = struct.unpack_from(f"<{self.L}I", blob, o)
        o += 4 * self.L
        if len(blob) != o + 2 * sum(lanes + n for n in nw):
            raise ValueError("fp8lin: blob size does not match its word counts")
        for n in nw:
            c = np.frombuffer(blob, dtype=np.int16, count=lanes, offset=o)
            if (
                int(c.min()) < 2
                or int(c.max()) > cap
                or int(c.astype(np.int64).sum()) != n
            ):
                raise ValueError("fp8lin: inconsistent lane counts")
            o += 2 * (lanes + n)

    def _body(
        self, blob: bytes, h: dict, payload: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per layer (counts int32 [lanes], words int16 [n]) views into a blob's
        payload on the device (see :meth:`_to_device`)."""
        nw = struct.unpack_from(f"<{self.L}I", blob, h["body"])
        lanes = -(-h["T"] // self.lane) * BW
        res, k = [], 0
        for n in nw:
            res.append(
                (
                    payload[k : k + lanes].to(torch.int32),
                    payload[k + lanes : k + lanes + n],
                )
            )
            k += lanes + n
        return res

    # ------------------------------------------------------------- positions

    @torch.no_grad()
    def find_starts(
        self, chunks: list[torch.Tensor], align: int, probe: int = 8, layer: int = 1
    ) -> list[int]:
        """Recover each chunk's absolute start position from its keys.

        Codes the first ``probe`` tokens of ``layer`` under every candidate start in
        ``range(0, max_position, align)`` (all chunks and candidates in one batched
        pass) and returns the cheapest candidate per chunk.

        Args:
            chunks: uint8 codes [L, T, 2C] per chunk, on the codec device.
            align: Candidate spacing (LMCache chunks start at multiples of the chunk
                size).
            probe: Tokens per chunk to code under each candidate.
            layer: Layer to probe (needs ``layer - 1`` as side information).

        Returns:
            One start position per chunk.
        """
        cands = torch.arange(0, self.max_position, align, device=self.device)
        nc = len(cands)
        k = min([probe] + [c.shape[1] for c in chunks])
        n = len(chunks)
        ar = torch.arange(k, device=self.device)
        pos = (cands[:, None] + ar[None]).reshape(-1).repeat(n)  # [n * nc * k]
        cos, sin = rope_tables(pos, self.th)
        cur = torch.cat([c[layer, :k].repeat(nc, 1) for c in chunks])
        prv = torch.cat([c[layer - 1, :k].repeat(nc, 1) for c in chunks])
        bits = self._ref_bits(
            self.layers[layer], self._features(prv, cos, sin), cur, cos, sin
        )
        best = bits.view(n, nc, k).sum(2).argmin(1)
        return [int(cands[j]) for j in best.tolist()]

    def _features(
        self, codes: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Un-rotated fp16 features of codes [N, 2C] (the same values ``_post``
        writes)."""
        N = codes.shape[0]
        cur = torch.empty(N, 2 * self.C, dtype=torch.float16, device=self.device)
        racc = torch.zeros(N, device=self.device)
        mu0 = torch.zeros(N, self.D, dtype=torch.float16, device=self.device)
        one = torch.ones(self.D, device=self.device)
        for b in range(2 * self.H):
            cb = codes[:, b * self.D : (b + 1) * self.D].contiguous()
            _post[(triton.cdiv(N, BT),)](
                cb,
                mu0,
                one,
                cos,
                sin,
                self.vals,
                racc,
                cur,
                N,
                2 * self.C,
                b * self.D,
                ROT=b < self.H,
                BT=BT,
            )
        return cur

    def _ref_bits(
        self,
        lc: LayerCoder,
        prev16: torch.Tensor,
        codes: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Model code length (bits) per token under the coder, Gaussian residual."""
        widths = (self.width_index, self.width_index)
        blocks, _ = self._code_layer(lc, prev16, cos, sin, widths, true=codes)
        sym = self.tabs.sym
        edges = self.tabs.edges
        total = torch.zeros(codes.shape[0], device=self.device)
        for mu16, r16, cb in blocks:
            i = sym[cb.long()].long()
            mu, r = mu16.float(), r16.float()

            def F(
                e: torch.Tensor, mu: torch.Tensor = mu, r: torch.Tensor = r
            ) -> torch.Tensor:
                return 0.5 * (1 + torch.erf((e - mu) * r * 0.7071067811865476))

            lo = torch.where(i <= 0, 0.0, F(edges[i.clamp(0, 254)]))
            hi = torch.where(i + 1 >= 254, 1.0, F(edges[(i + 1).clamp(0, 254)]))
            total += -torch.log2((hi - lo).clamp_min(1e-9)).sum(1)
        return total
