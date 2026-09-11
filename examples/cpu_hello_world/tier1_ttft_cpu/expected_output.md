# Tier 1 — expected output

Illustrative output. Absolute timings depend heavily on your CPU, memory
bandwidth, and the model; the **shape** is what matters: chunks are written on
the cold request and read on the warm request, and warm TTFT ≤ cold TTFT.

## Console (abridged)

```
==> Tier 1: LMCache TTFT reduction on CPU
    model:        Qwen/Qwen2.5-0.5B-Instruct
    chunk-size:   16  (demo value; production default is 256)
...
==> [1/3] cold request (prompt seen for the first time)
cold: TTFT = 1.8300s
==> [2/3] warm request (identical prompt -> should hit LMCache)
warm: TTFT = 0.4200s
==> [3/3] negative request (different prompt -> should NOT hit)

======================= RESULTS =======================
  cold TTFT : 1.83s
  warm TTFT : 0.42s
  store: 0 -> 62 chunks written (cold)
  hit  : 0 -> 62 chunks read    (warm)
  negative read delta: 62 -> 62
=======================================================
==> TTFT dropped on the cache hit (warm < cold).
==> PASS: LMCache stored KV on the cold request and served it on the warm
    request, entirely on CPU.
```

## JSONL (`ttft.jsonl` in the run's temp dir)

```json
{"label": "cold", "prompt_chars": 12040, "ttft_seconds": 1.83, "streamed_chunks": 8}
{"label": "warm", "prompt_chars": 12040, "ttft_seconds": 0.42, "streamed_chunks": 8}
{"label": "negative", "prompt_chars": 12065, "ttft_seconds": 1.79, "streamed_chunks": 8}
```

## How to read it

- **`store: 0 -> 62 chunks written`** — the cold request computed KV for the
  shared prefix and LMCache stored 62 chunks (≈ prefix_tokens / chunk_size).
- **`hit: 0 -> 62 chunks read`** — the warm request retrieved all 62 chunks
  from LMCache instead of recomputing them.
- **`negative read delta: 62 -> 62`** — the different prompt read nothing new;
  hits are prefix-driven.
- If `warm` is not visibly faster than `cold`, that's fine on tiny
  models/machines — the read counter is the real proof. Increase the shared
  context length to widen the TTFT gap.
