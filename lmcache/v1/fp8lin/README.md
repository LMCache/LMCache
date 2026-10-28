# fp8lin: lossless predictive coding of FP8 KV chunks

`fp8lin` stores KV chunks as E4M3 codes entropy coded on the GPU against a fitted
linear prediction. It is lossless with respect to FP8 KV: the decoded bytes are the
stored bytes. Typical size is **4.2–4.8 bits per value** (Qwen3 1.7B–14B, all layers),
i.e. **1.7–1.9× smaller than FP8** and 3.4–3.8× smaller than bf16.

Used by two serdes:

| mode | config | file |
|---|---|---|
| multiprocess (`lmcache server`) | `"serde": {"type": "fp8lin", "coder_path": ...}` on an L2 adapter | `lmcache/v1/distributed/serde/fp8lin.py` |
| in-process (deprecated) | `remote_serde: fp8lin`, `extra_config: {fp8lin.coder_path: ...}` | `lmcache/v1/storage_backend/naive_serde/fp8lin_serde.py` |

## How it codes a chunk

A chunk is `[2, L, T, H*D]` (K and V of `L` layers for `T` consecutive tokens).

```
for layer l = 0..L-1:                       (layer l-1 already decoded)
  for block b = K head 0..H-1, V head 0..H-1:
    mu  = W_b · [decoded blocks < b of this token (K un-rotated out of RoPE),
                 decoded layer l-1 of this token]           # fp16 GEMM, Triton
    s   = sig_b · exp(0.5 (w0 + w1 · log residual energy of earlier blocks))
    codes of block b  <- rANS decode under N(mu, (1.05 s)^2) over the 254 E4M3 codes
```

* **Integer CDFs without normalization.** `CDF(i) = floor(F(edge_i)·(2^16−254)) + i`
  gives every code frequency ≥ 1 and costs one Gaussian evaluation per entry.
* **Lanes.** One rANS state per (128-token group, channel); a chunk's lanes are
  contiguous, so many chunks code in one GPU pass and split cleanly.
* **Positions.** LMCache chunks carry no positions, but K is stored after RoPE.
  The encoder recovers each chunk's absolute start by coding its first tokens under
  every candidate start (a wrong start costs ~2× the bits; 33/33 chunks recovered on
  Qwen3-1.7B) and stores it in the header. `cached_positions` is used when present.

## Guarantees and limits

* **Exact.** Encoder and decoder run the same kernels on the same inputs. Every
  kernel computes a token's prediction from that token's data alone — including
  the matmuls, which use a fixed-tiling Triton GEMM instead of cuBLAS — so a chunk
  decodes identically whichever chunks it is batched with.
* **Checked.** Each blob carries a 64-bit checksum of its codes, and its structure is
  validated on the host before any GPU work; a corrupt blob is a cache miss, never
  bad KV and never an out-of-bounds GPU access (fuzzed with 300 random corruptions:
  298 detected, 2 harmless, 0 wrong).
* **Never larger than raw.** Chunks with NaN codes, or whose coded size would exceed
  the raw codes, are stored raw (header + 1 byte per value).
* **Same GPU type.** Float CDFs are bit-identical on one GPU architecture and
  software stack; encode and decode on different GPU types is not supported.
* **head_dim 128** (Qwen3, Llama-3 8B+, Mistral); not MLA models.
* **bf16 KV caches** are rounded to E4M3 at the coder's per-layer scales first
  (lossy, like the `fp8` serde); FP8 KV caches are coded losslessly.

## Fitting a coder

Once per model and KV convention, on one GPU (about 10 minutes for an 8B model):

```bash
# bf16 KV cache (vLLM default): the serde rounds to E4M3 at the fitted scales
python -m lmcache.v1.fp8lin.fit --model Qwen/Qwen3-8B --kv bf16 --out /models/qwen3-8b-fp8lin
# FP8 KV cache (vLLM kv_cache_dtype="fp8", unit scales): coded losslessly
python -m lmcache.v1.fp8lin.fit --model Qwen/Qwen3-8B --kv fp8 --out /models/qwen3-8b-fp8lin-fp8
```

It captures KV on 128 windows of 1,024 tokens (C4 by default, or `--text file`) and
writes the coder directory:

```
meta.json        {"format": "fp8lin-coder-v1", "kv": "bf16", "num_layers": 36,
                  "num_kv_heads": 8, "head_dim": 128, "kv_scale": {"k": [...], "v": [...]},
                  "width_index": 5, "lane_tokens": 128, "max_position": 40960}
layer_000.npz    per-layer least-squares maps: cols<b>, W<b> (block b), sd, sig, Wh, th
...
```

## Performance (one RTX 3080, Qwen3-1.7B, 32 chunks of 256 tokens per batch)

| | tokens/s |
|---|---|
| encode (incl. position search) | ~15k |
| decode | ~18–29k |
| bf16 recompute (same GPU) | ~8k |

On an RTX 6000 Ada, the same coder in a single 32k-token pass decodes
110–158k tokens/s (Qwen3 1.7B–14B), 5–30× faster than recomputing the prefix.

## End to end (vLLM 0.32 + `lmcache server`, fs L2, RTX 4500 Ada, Qwen3-1.7B)

`examples/serde/fp8lin`, a 6,364-token prompt (24 chunks of 256 stored), L1 cleared
before each warm request:

| L2 serde | KV cache | L2 bytes / token | warm prefetch, 24 chunks | warm = cold output |
|---|---|---|---|---|
| `fp8` | bf16 | 57,344 | 153 ms | no (E4M3 rounding) |
| `fp8lin` | bf16 | 33,610 (1.71× smaller) | 618 ms incl. JIT | no (E4M3 rounding) |
| `fp8lin` | FP8 | 33,353 (1.72× smaller) | 278 ms (546–559 ms first, JIT) | **yes**, 3/3 |

On FP8 KV caches the ``fp8`` serde casts the uint8 codes numerically
(`.to(float8_e4m3fn)`), so its restored KV is wrong: the warm output diverges at the
second character. At 1.7B on a local disk recomputing the prefix is about as fast as
either restore; the gain here is capacity: 1.72× more exact prefix per byte of L2.

The vLLM MP connector stores a chunk as `[1, L, T, H·2·D]` (each head's K and V
adjacent); the serde also accepts `[2, L, T, H·D]`.
