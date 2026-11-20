# Tier 1 — expected output

Output from a real run (macOS 15 on Apple Silicon, CPU-only, vLLM
`0.30.1.dev202609231158`). Absolute timings depend heavily on your CPU, memory
bandwidth, and the model; the **shape** is what matters: chunks are written on
the cold request and read on the warm request, and warm TTFT ≤ cold TTFT.

## Console (abridged)

```
==> Tier 1: LMCache TTFT reduction on CPU
    model:        Qwen/Qwen2.5-0.5B-Instruct
    chunk-size:   256
    shared prompt:       7346 chars
    negative prompt:     7592 chars
...
==> [1/3] cold request (prompt seen for the first time)
==> [2/3] warm request (identical prompt -> should hit LMCache)
==> [3/3] negative request (different prompt -> should NOT hit)

======================= RESULTS =======================
  cold TTFT : 5.5029s
  warm TTFT : 0.8146s
  store: 0 -> 6 chunks written (cold)
  hit  : 0 -> 6 chunks read    (warm)
  negative read delta: 6 -> 6
=======================================================
==> TTFT dropped on the cache hit (warm < cold).
==> PASS: LMCache stored KV on the cold request and served it on the warm
    request, entirely on CPU.
```

## JSONL (`ttft.jsonl` in the run's temp dir)

```json
{"label": "cold", "prompt_chars": 7346, "ttft_seconds": 5.5029, "streamed_chunks": 8}
{"label": "warm", "prompt_chars": 7346, "ttft_seconds": 0.8146, "streamed_chunks": 8}
{"label": "negative", "prompt_chars": 7592, "ttft_seconds": 6.1917, "streamed_chunks": 8}
```

## How to read it

- **`store: 0 -> 6 chunks written`** — the cold request computed KV for the
  shared prefix and LMCache stored 6 chunks. The shared prompt is ~1700 tokens,
  so at `chunk-size 256` that is six full chunks (the remainder is a partial
  chunk and is not stored).
- **`hit: 0 -> 6 chunks read`** — the warm request retrieved all 6 chunks from
  LMCache instead of recomputing them. Here that cut TTFT from 5.50s to 0.81s,
  roughly 6.8x, because prefill is the dominant cost on CPU.
- **`negative read delta: 6 -> 6`** — the different prompt read nothing new;
  hits are prefix-driven. Its TTFT (6.19s) is cold-like, as expected.
- If `warm` is not visibly faster than `cold`, that's fine on tiny
  models/machines — the read counter is the real proof. Increase the shared
  context length to widen the TTFT gap, raising `MAX_MODEL_LEN` to match.
