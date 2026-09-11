# Tier 1 — LMCache reducing TTFT on CPU with a tiny model

Runs a real language model on CPU and shows LMCache serving a previously
computed KV cache — the core value proposition — with no GPU.

The driver brings up an `lmcache server` and a CPU-only `vllm serve` wired to
it through the `LMCacheMPConnector`, then sends three requests:

| Request | Prompt | Expected |
|---------|--------|----------|
| **cold** | long shared prefix | KV computed and **stored** in LMCache |
| **warm** | *identical* prompt | KV **retrieved** from LMCache → lower TTFT |
| **negative** | different prompt | no shared prefix → **no** cache hit |

## Run

```bash
# from the repo root, after examples/cpu_hello_world/install_cpu.sh
examples/cpu_hello_world/tier1_ttft_cpu/run_demo.sh
```

Default model: **`Qwen/Qwen2.5-0.5B-Instruct`** (Apache-2.0, ~0.5 B params).
The first run downloads it from Hugging Face (a few hundred MB, no token
needed). Override with `MODEL=...` (see the license note in the parent
[README](../README.md#model--license) before shipping a different one).

## What success looks like

```
======================= RESULTS =======================
  cold TTFT : 1.83s
  warm TTFT : 0.42s
  store: 0 -> 62 chunks written (cold)
  hit  : 0 -> 62 chunks read    (warm)
  negative read delta: 62 -> 62
=======================================================
==> PASS: LMCache stored KV on the cold request and served it on the warm
    request, entirely on CPU.
```

(Numbers vary by machine; see [`expected_output.md`](expected_output.md).)

## What this proves

- **A real cache hit on CPU:** the write counter increases on the cold
  request and the read counter increases on the warm one — LMCache stored the
  KV and served it back.
- **TTFT reduction from reuse:** the warm request skips prefill for the shared
  prefix. vLLM's own prefix caching is disabled (`--no-enable-prefix-caching`),
  so LMCache is unambiguously the source of the reuse.
- **The negative control:** a different prompt does not increase the read
  counter, confirming hits are prefix-driven, not incidental.

## Why the counter, not just the clock

On a tiny model on a small CPU box, the absolute prefill saved can be modest,
so the warm-vs-cold TTFT gap may be small (or noisy). The **cache-hit counter**
(`lmcache_mp_l1_read_chunks_total`) is the authoritative proof that LMCache
served the KV. The gap grows with longer shared contexts and larger models —
which is exactly when LMCache matters most in production.

## Variations

- **Longer shared context** — raise the prompt size (edit the generator in
  `run_demo.sh`) or point at `examples/online_session/ffmpeg.txt`; the TTFT gap
  widens.
- **Disk (L2) offload** — add a local-disk backend so the cache survives
  beyond RAM. See [`examples/kv_cache_reuse/local_backends/`](../../kv_cache_reuse/local_backends/)
  and the [CPU RAM / storage-backend docs](../../../docs/source/kv_cache/storage_backends/).
- **Fuller TTFT benchmarking** — [`examples/online_session/`](../../online_session/)
  provides context sweeps and cache-flush controls against the same endpoint.

## Notes / caveats

- **CPU startup is slow.** vLLM can take a minute or more to load on CPU; the
  script waits up to 10 minutes by default (`VLLM_READY_TIMEOUT`).
- **Chunk size is lowered to 16** for the demo so short prompts still form
  several cacheable chunks. Production default is 256 — a prompt shorter than
  one chunk caches nothing.
- **Transport:** POSIX shared-memory transports are Linux-only; this demo uses
  the default engine-driven path, which works on Linux and macOS.
