---
name: hybrid-benchmarking
description: Benchmark a hybrid-attention model across the LMCache performance ladder (vLLM no-hybrid-allocator → hybrid allocator + prefix caching → hybrid allocator + LMCache) and produce a decode-throughput / TTFT / cache-hit-rate comparison. Use when asked to benchmark, demo, or quantify the value of LMCache caching for a hybrid model.
allowed-tools: Bash, Read, Grep, Glob, Write
argument-hint: "<hf-model-id> [--gpus N] [--tp N] [--output-dir DIR]"
---

# Hybrid-Model Benchmarking — the LMCache performance ladder

This skill is a **handbook**: given a target hybrid-attention model (and hardware),
calibrate a workload, run four vLLM configurations, and deliver a **Markdown report**
(`report.md`) demonstrating the LMCache performance ladder — exact launch commands, a
decode-throughput / TTFT / TPOT / hit-rate comparison, and a rationale for every tuned
number.

> This handbook was validated end-to-end on three distinct hybrid families:
> `google/gemma-4-31B-it` (sliding-window, 2×H200 TP=2), `Qwen/Qwen3.6-27B` (Mamba/GDN,
> 1×H200), and `deepseek-ai/DeepSeek-V4-Flash` (sparse-MLA + fp8 KV, 8×H200 TP=8). The
> empirical lessons from those runs are baked in below — **read the family-specific notes
> in Step 1 and the Troubleshooting table before running**; several "obvious" choices are
> wrong and cost hours.

> **Run fully autonomously end-to-end. Do NOT stop mid-run to ask for approval or
> confirmation** (operating point, calibration, "should I proceed?", etc.) — it's
> disruptive. Log each decision and keep going through all runs and the report. The only
> acceptable stop is a hard blocker you cannot resolve (e.g. the model isn't given, or
> gated weights are missing). The user will interrupt if something looks off.

## Inputs

`$ARGUMENTS`:
- **Positional 1 (required):** the HF model id, e.g. `Qwen/Qwen3.6-27B`, `google/gemma-4-31B-it`.
- `--gpus N`, `--tp N` — default: detect with `nvidia-smi -L`; tp = gpus.
- `--output-dir DIR` — default `./hybrid-bench-<model-slug>/`.

If the model isn't given, ask first. **On a shared box**, check `nvidia-smi` for other
users' processes and pick free GPUs / non-default ports; never kill processes you didn't
start.

## The ladder you are demonstrating (and what actually happens)

Run the **same** `long-doc-qa` workload against four configs:

| Run | vLLM configuration | What the run shows |
|------|--------------------|--------------------|
| **A** | hybrid allocator **off**, prefix caching **off** | Baseline. **Smallest** decode batch → **lowest** decode throughput; no cache → worst TTFT. |
| **B** | hybrid allocator **on**, prefix caching **off** | **Isolates the allocator**: packs **~8× more** tokens → **much larger** batch → **much higher** decode throughput than A. |
| **C** | hybrid allocator **on** + `--enable-prefix-caching`, no LMCache | Adds GPU prefix caching — but it's **~0 % hit** under batch saturation (see below), so it's **≈ Run B** (prefix caching *alone* doesn't help here). |
| **D** | Run C + `LMCacheMPConnector` + `lmcache server` | CPU pool serves prefixes regardless of GPU pressure → **~100 % hit** → **lowest TTFT** and **highest decode throughput** (cache hits free compute for decode). |

The A→B step isolates the hybrid allocator (capacity/decode-throughput); B→C isolates GPU
prefix caching (≈ no-op here — it's saturated); C→D isolates LMCache (the real TTFT/hit win).

> **Mandatory-allocator models collapse to 3 runs (Run A ≡ Run B).** For Mamba/GDN
> (Qwen3.5/3.6, Qwen3-Next) and DeepSeek-V4 sparse-MLA there is **no allocator-off
> baseline**: `--disable-hybrid-kv-cache-manager` is either rejected (Mamba/GDN) or accepted
> but fails at runtime (DeepSeek-V4 — its CuTe sparse-attention kernel only supports the
> C128 / block_size=8 layout). Collapse to **3 distinct runs**: baseline (no prefix
> caching) → + GPU prefix caching → + LMCache. The A→B allocator contrast does not exist;
> the headline is the cache story (C→D). The full 4-run structure above applies only to
> sliding-window hybrids (Gemma 3/4, gpt-oss), where Run A can truly disable the allocator.
>
> **Mamba/GDN turnkey config (use these to get a clean result first try):**
> - Probe the unified block size **N** (`Setting attention block size to N tokens`).
> - **Every** run: `--mamba-cache-mode align`, **`--max-num-batched-tokens 2N-1`** (NOT N —
>   N serializes prefill, see gotcha), `--max-num-seqs ≤` the #Mamba cache blocks vLLM
>   reports (else CUDA-graph capture fails; e.g. 800 for Qwen3.6-27B).
> - Run baseline `--no-enable-prefix-caching`; +PC `--enable-prefix-caching`; +LMCache adds
>   the connector + `lmcache server --chunk-size N` + `lmcache.mp.mq_timeout 900`.
> - Expected clean result: +PC ≈ baseline (0% hit under saturation, but no catastrophe at
>   2N-1); **+LMCache wins** (~97% hit, lowest TTFT). Generation not bit-exact (GDN).
>
> **Mamba/GDN gotcha — `+ GPU prefix caching` can be *much worse* than baseline (config
> pathology, verified).** Measured on Qwen3.6-27B at 0 % hit: TTFT ~13×, wall-clock ~7×
> worse than baseline. **Root cause: concurrency collapse, NOT state-checkpoint cost.**
> Prefix caching engages `mamba_cache_mode=align`, which forces every prefill chunk to a
> whole `block_size`(=N)-aligned chunk (`scheduler.py:_mamba_block_aligned_split`,
> `num_new_tokens // N * N`). If `--max-num-batched-tokens` == N (the minimum of the
> required `[N,2N)`), then any in-flight decode consumes ≥1 token of the per-step budget,
> leaving < N → a new request's block-aligned prefill rounds to **0 tokens** and can't be
> admitted → **strictly serial, Running:1, GPU KV ~4 %** (baseline: Running 24, KV ~92 %).
> **Verified by controlled A/B:** raising *only* `--max-num-batched-tokens` N→~2N restored
> Running 1→27, KV 4 %→99 %, wall-clock 3.7× / TTFT 6.8× faster (residual ~1.9× vs baseline
> = the real `align` overhead). **Lever: set `--max-num-batched-tokens` toward 2N, not N**
> (and `--max-num-seqs` ≤ #Mamba cache blocks, else CUDA-graph capture fails). LMCache runs
> the same `align` path but **hits (97 %)** → state loaded, not recomputed → only config
> that beats baseline.
> **Lesson for the skill: when a result is surprising, verify the mechanism with a
> controlled single-variable A/B before writing the cause — don't assert an unprofiled
> rationale.**

> **DeepSeek-V4 sparse-MLA + fp8 KV turnkey config (use these to get a clean result first
> try):**
> - **JIT prerequisite:** the sparse-attention kernel is JIT-compiled at startup and fails
>   on the default toolchain (*"CUDA compiler and CUDA toolkit headers are incompatible"*).
>   Export a matching `CUDA_HOME` for **every** `vllm serve` (e.g.
>   `export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH` so `nvcc`
>   matches the torch CUDA build).
> - **Required flags:** `--kv-cache-dtype fp8_ds_mla --trust-remote-code
>   --tokenizer-mode deepseek_v4 --enable-expert-parallel` (MoE). Mandatory hybrid allocator
>   (3 runs — see above).
> - **Lower util hard (≈0.35), not the usual 0.50.** fp8 KV is compressed *on GPU*, so at
>   util 0.80 the GPU pool is enormous (~12 M tokens) and swallows the whole working set →
>   nothing for LMCache to serve. Drop util until the GPU pool is *below* the working set so
>   it overflows (forces the +PC saturation and gives LMCache prefixes to serve).
> - **Big LMCache per-token storage:** LMCache stores the full sparse-MLA state (~62 KB/token,
>   `tokens_per_gb ≈ 16 800`), so the pinned pool must be large (we used `l1_size_gb 600` for
>   a ~172 GiB working set) to stay below the eviction watermark.
> - Expected clean result: +PC ≈ baseline (0 % hit under saturation); **+LMCache wins**
>   (~99 % hit, ~2× lower TTFT, ~1.6× decode throughput).

**Two findings that flip naive expectations (don't get these wrong):**

1. **For sliding-window-heavy hybrids, the hybrid allocator gives the LARGER batch, not the smaller one.** Sliding-window layers keep only a `window`-sized KV slice, so the allocator packs ~`num_layers / num_full_layers` × more tokens (≈ **8×** for Gemma-4: 50/60 sliding). So **Run A (allocator off → all layers full) has the *smallest* batch and the *lowest* decode throughput** — the opposite of the "no prefix caching → bigger batch" intuition. Verify the ratio empirically (Step 2).
2. **Run C's GPU prefix cache is ~0 % hit under batch saturation** (so Run C ≈ Run B). When the active batch's KV (`batch × (doc+output)`) ≈ the whole GPU pool, there are no free blocks to *retain* cached prefixes, so they're evicted by live traffic. GPU prefix caching is useless exactly when memory is tight — which is the gap LMCache (CPU pool) fills. This is *the* point of the demo.

## What it takes to make each criterion visible

The default `long-doc-qa` run is **prefill-bound** (huge prompt, tiny decode), which
*hides* the decode-throughput difference. You must shape the workload:

- **Decode-throughput criterion (B ≫ A):** make the run **decode-bound** with
  `--ignore-eos --ldqa-output-length <N>` (e.g. 2048 at L=24 000). Without `--ignore-eos`
  the model emits EOS after ~10 tokens → ~100 % prefill → the gap collapses to the
  prefill-throughput ratio (~1.5×). With a forced long decode, the gap reflects the real
  batch advantage (we measured **2.7×**; it grows toward the memory-bandwidth ceiling
  with longer output / shorter context).
- **Cache criterion (D TTFT ≪ C, D hit ~100 %):** the document must be a long, cacheable
  prefix (keeps A batch-starved too), and the working set must **overflow the GPU pool
  but fit the LMCache pool with margin** (Step 3b).

`--ignore-eos` makes output deterministic (identical total output tokens across runs),
so decode-throughput numbers are apples-to-apples and reproducible. **Use it.**

---

## Step 0 — Prerequisites & references

1. `lmcache --version`, `vllm --version`, `nvidia-smi -L`, `free -g` all sane. Note
   `HF_HOME` if weights live off the default cache (e.g. `/raid/...`) — export it for
   **every** command (shell env does not persist between tool calls).
   **Record exact versions for the report** (benchmarks are meaningless without them):
   ```bash
   vllm --version                                 # e.g. 0.1.dev17429+g2c9c07c85 → vLLM commit 2c9c07c85
   git -C "$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')/.." rev-parse --short HEAD 2>/dev/null
   git -C <lmcache-repo> rev-parse --short HEAD    # LMCache commit
   git -C <lmcache-repo> status --short            # note any local patch (e.g. --ignore-eos not yet upstream)
   ```
   If LMCache is running with local/un-upstreamed changes (e.g. the `--ignore-eos` /
   `--ldqa-output-length` bench patch), record the **base commit ID + "with `<patch>` patch"**;
   once the patch merges, record the merged commit directly.
2. **Read the model's recipe** under `docs/source/recipes/` (`gemma4.rst`, `qwen3_5.rst`,
   `gemma3.rst`, `gpt_oss.rst`, …) for validated flags + per-model quirks; else read
   `docs/source/mp/hybrid_models.rst` and warn the model is unvalidated.
3. Confirm the weights are present (`ls $HF_HOME/hub/models--<org>--<model>`); a gated
   download is a blocker — surface it, don't stall.

## Step 1 — Classify the model and decide Run A

| Family | Examples | Run A ("no hybrid allocator") |
|--------|----------|--------------------------------|
| **Sliding-window + full hybrid** | Gemma 3/4, gpt-oss | `--disable-hybrid-kv-cache-manager --no-enable-prefix-caching` — collapses to one full-attention group. |
| **Mamba / GDN linear-attention hybrid** | Qwen3.5/3.6, Qwen3-Next | `--disable-hybrid-kv-cache-manager` is **invalid** (manager mandatory). Run A = manager **on** + `--no-enable-prefix-caching`. Note this caveat + GDN's non-bit-exact caching in the report. |
| **Sparse-MLA + fp8 KV** | DeepSeek-V4 | `--disable-hybrid-kv-cache-manager` parses but **fails at runtime** (CuTe sparse-attn needs C128/block_size=8) — manager mandatory. Run A = manager **on** + `--no-enable-prefix-caching`. See the DeepSeek-V4 turnkey note (CUDA_HOME JIT, fp8 forces low util). |
| **Dense** (not hybrid) | Llama, Qwen3 dense, Mistral | Not a hybrid model; the allocator distinction is moot. |

Classify via the recipe, else the HF `config.json` `architectures` / `layer_types`.
**Mamba/GDN only:** probe the unified block size `N` (vLLM logs `Setting attention block
size to N tokens`) — needed for server `--chunk-size N` and vLLM `--max-num-batched-tokens`
**`2N-1`** (the max of the required `[N,2N)`; using `N` serializes prefill — see the
Mamba gotcha above), plus `--mamba-cache-mode align`. Also set `--max-num-seqs` ≤ the
number of Mamba cache blocks vLLM reports, or CUDA-graph capture fails.

## Step 2 — Probe both pools (the 8× capacity fact)

Launch the engine with `--max-model-len auto` and read vLLM's startup log:
```
INFO ... Available KV cache memory: XX.XX GiB
INFO ... GPU KV cache size: NNN,NNN tokens
INFO ... Maximum concurrency for <max_model_len> tokens per request: X.XXx
```
Do this for **both** the hybrid config (B/C) and the hybrid-off config (A) at the chosen
util. Record:
- **GPU pool tokens** for hybrid (`P_B`) and hybrid-off (`P_A`). Their ratio is the
  allocator's packing gain (≈8× for Gemma-4). `P_A` caps Run A's max request length.
- **`tokens_per_gb`** for the LMCache config: once the `lmcache server` + connector engine
  is up, read `cache_context_meta.<gpu>.kv_cache_layout.cache_size_per_token` from the
  server's `/status`; `tokens_per_gb = 1024³ / cache_size_per_token`. Pass this to the
  bench via `--tokens-per-gb-kvcache` so `num_documents` is identical across runs.

⚠️ **Use `--max-model-len auto` (or the model's true max), never a small pinned value.**
Pinning a small `max-model-len` (e.g. 16 384) cripples the sliding-window memory saving
and shrinks the effective pool ~5×. Auto resolves to the model max and gives the realistic
pool.
⚠️ **`cache_size_per_token` underestimates real on-pool storage by ~1.4×** (padding /
metadata / block-granular hybrid storage). Size the working set against *measured* usage,
not this number (Step 3b).

## Step 3 — Calibrate the operating point

### 3a. `--gpu-memory-utilization` and document length `L`
- Keep util **< 0.9** (LMCache headroom). On a big GPU the hybrid pool is huge, so you may
  need to **lower util** to make GPU memory the binding constraint and keep the working set
  affordable. (We used 0.50 on H200.)
- Pick `L` long enough that (a) it fits both pools (`L ≤ P_A`), (b) the hybrid batch
  `P_B / (L+output)` is in a sane range (a few tens — a longer `L` starves Run A harder
  and sharpens the A-vs-B gap), and (c) it's a meaningful prefix to cache. The old
  "decode 50–100× shorter than input" rule does **not** apply once you set
  `--ldqa-output-length` — pick `L` for the batch/cache story and set output separately.

### 3b. Working set (`--kv-cache-volume` → `num_documents`)
`num_documents = floor(kv_cache_volume_gb × tokens_per_gb / L)`. Require:
- **Overflow GPU:** `num_documents × L  >  P_B` (so Run B misses). Even a small overflow
  works because B's prefix cache is saturated anyway (~0 % hit).
- **Fit LMCache with margin:** estimated **real** storage `≈ 1.4 × (working_set_tokens /
  tokens_per_gb)` must sit **well under** `watermark × l1_size_gb` (use watermark 0.95 →
  target ≤ ~0.6 × `l1_size_gb`). Eviction is computed as `used / current_realized_pool`,
  so leaving margin is what guarantees Run C ~100 % hit.
- `l1_size_gb` ≤ host RAM and ≤ `/dev/shm` size; the pinned pool is non-swappable.

### 3c. Decode length, then proceed (no approval stop)
- Set **`--ldqa-output-length`** so the run is decode-bound: roughly
  `output > 2 × L × decode_tput / prefill_tput` (≈2048 at L=24 000). Longer output →
  cleaner decode-throughput gap but longer runtime (decode is slow at long context).
- **Log** the operating point (family, Run-A def, util, `L`, output, `num_documents`,
  working-set GB, `P_A`/`P_B`, `l1_size_gb`, Mamba `N`) and **proceed straight into the
  runs — do NOT pause for approval** (see the autonomy note at the top).

## Step 4 — Freeze one shared workload config

```bash
lmcache bench engine --engine-url http://localhost:8000 --workload long-doc-qa \
    --model <model> --tokens-per-gb-kvcache <pinned> --kv-cache-volume <Step 3b> \
    --ldqa-document-length <L> --ldqa-query-per-document <1-2> \
    --ldqa-num-inflight-requests <≥ hybrid batch> \
    --ldqa-output-length <Step 3c> --ignore-eos \
    --no-interactive --export-config "$OUT/shared.json"
```
`--ignore-eos` and `--ldqa-output-length` persist in the exported config. Replay every
run with `--config "$OUT/shared.json"`. Pass `--tokens-per-gb-kvcache` explicitly (not
`--lmcache-url`) so `num_documents` is byte-identical even in runs A/B (no server).

## Step 5 — Run the four runs

Start the **`lmcache server` first** (Run D needs it) so its pinned pool pre-expands while
A/B/C run — the lazy allocator pins ~80 GiB/min, and you want it fully pinned before D to
avoid mid-fill eviction:
```bash
lmcache server --port 5560 --http-port 8090 --prometheus-port 9099 \
    --l1-size-gb <Step 3b> --eviction-policy LRU --eviction-trigger-watermark 0.95 \
    [--chunk-size <N for Mamba/GDN>]
```
For each run: launch the engine, **poll `curl -sf http://localhost:8000/health`** until
ready, run the bench, **snapshot metrics (Step 6) before teardown**, then tear down by PID
(Step 5 teardown) before the next run. `<base>` = recipe's `vllm serve` flags
(`--tensor-parallel-size`, `--trust-remote-code`, model-specific).

```bash
# Run A — hybrid allocator OFF, prefix caching OFF (baseline)
vllm serve <model> <base> --gpu-memory-utilization <U> --max-model-len auto --port 8000 \
    --disable-hybrid-kv-cache-manager --no-enable-prefix-caching
#   Mamba/GDN: drop --disable-hybrid-kv-cache-manager; add --mamba-cache-mode align --max-num-batched-tokens <2N-1>

# Run B — hybrid allocator ON, prefix caching OFF (isolates the allocator)
vllm serve <model> <base> --gpu-memory-utilization <U> --max-model-len auto --port 8000 \
    --no-enable-prefix-caching [--mamba-cache-mode align --max-num-batched-tokens <2N-1>]

# Run C — hybrid ON + prefix caching, no LMCache
vllm serve <model> <base> --gpu-memory-utilization <U> --max-model-len auto --port 8000 \
    --enable-prefix-caching [--mamba-cache-mode align --max-num-batched-tokens <2N-1>]

# Run D — + LMCache.  mq_timeout 900: a fully pre-pinned large pool makes CUDA-IPC
# register_kv_caches slow (~10 min for TP=2); the default 300 s aborts the engine.
vllm serve <model> <base> --gpu-memory-utilization <U> --max-model-len auto --port 8000 \
    --enable-prefix-caching [--mamba-cache-mode align --max-num-batched-tokens <2N-1>] \
    --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.port":5560,"lmcache.mp.mq_timeout":900}}'

# Bench (identical per run)
lmcache bench engine --engine-url http://localhost:8000 --config "$OUT/shared.json" \
    --no-interactive --json --output-dir "$OUT/run-<A|B|C|D>"
```

**Run D must be measured at steady state.** Run the bench **twice** on the live D engine:
run #1 *primes* the cache (cold-fill transient), run #2 *measures*. Report run #2 and the
metrics delta over it — otherwise the cold-fill drags the hit rate down. Before D's bench,
confirm the server pool is pinned (`/status` `memory_total_bytes` ≈ `l1_size_gb`) and the
engine has **registered** (`/status` `cache_context_meta` non-empty).

**Teardown (do this between runs and at the end):** kill by **PID**, not pattern.
`pkill -f "vllm serve <model>"` self-matches the shell wrapper (exit 144) and, worse,
killing the parent `vllm serve` orphans its `EngineCore`/`Worker` subprocesses, which keep
holding GPU memory. Get the worker PIDs from `nvidia-smi --query-compute-apps=pid,...` and
kill those + the main PID. Kill the `lmcache server` cleanly too — in auto transfer mode it
parks CUDA-IPC contexts on the engine's GPUs, and an unclean kill leaves `[No data]` zombie
allocations (they clear once all referencing PIDs exit). Verify GPUs return to ~0 MiB.

## Step 6 — Collect metrics

**Bench JSON** (`bench_summary.json` → `results`): `mean_ttft_ms`, `p90/p99_ttft_ms`,
`mean_decode_speed`, `input_throughput` (prefill), `output_throughput` (**decode
throughput**), `total_output_tokens` (constant across runs thanks to `--ignore-eos`).

**Hit rate — from vLLM `:8000/metrics`** (the LMCache Prometheus endpoint is often
**disabled** in MP builds: `standalone metrics HTTP server disabled`). Snapshot before and
after each run's bench (each run is a fresh engine, so cumulative = that run):
```bash
curl -s http://localhost:8000/metrics | grep -E 'vllm:(prefix_cache|external_prefix_cache)_(hits|queries)_total'
# GPU-local hit  = prefix_cache_hits_total          / prefix_cache_queries_total
# LMCache hit    = external_prefix_cache_hits_total  / external_prefix_cache_queries_total   (Run D)
```
For Run D report the **measure-run delta** (after − before of run #2). `prefix_cache_queries_total`
can over-count at block granularity — prefer the `external_prefix_cache_*` token counters
for the LMCache hit rate.

## Step 7 — Report (`$OUT/report.md`)

Header must record **versions** (from Step 0): vLLM commit, LMCache commit (or base commit
+ "with `<patch>` patch" if running un-upstreamed changes), hardware, TP. Then three parts:
**(7a) exact launch commands** (server + 4 `vllm serve` + bench, real values,
note the prime+measure double-run for D); **(7b) comparison table** — decode throughput
(`output_throughput`), mean/p90/p99 TTFT, TPOT (`1000/mean_decode_speed`), prefill
throughput, hit rate, wall-clock, GPU pool tokens, effective batch; **(7c) rationale +
findings** — why each knob, the 8× allocator fact, C's 0 % prefix-cache hit under
saturation (Run C ≈ Run B), the decode-bound requirement, the no-eviction sizing, and any
Mamba/GDN caveat.

**Validate the criteria explicitly (the 4 runs isolate one variable each):**
- **A → B (hybrid allocator):** decode throughput jumps (≈8× batch) — the headline HMA win.
- **B → C (GPU prefix caching):** ≈ no change — prefix caching alone is ~0 % hit under saturation.
- **C → D (LMCache):** TTFT collapses (**D ≪ C**) and decode throughput rises again (freed prefill compute).
- **Hit rate:** A/B 0 (caching off), C ~0 % (on but saturated), **D ~100 %**.

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| B decode tput ≈ A (not ≫) | run is prefill-bound | add `--ignore-eos`, raise `--ldqa-output-length` until decode dominates elapsed |
| D hit rate ≪ 100 % | working set hit the eviction watermark mid-fill | pre-pin the pool before D; raise `--l1-size-gb`; shrink working set (real storage ≈1.4× estimate) |
| D engine aborts: "did not respond to register_kv_caches within 300s" | big pinned pool → slow CUDA-IPC registration | set `kv_connector_extra_config.lmcache.mp.mq_timeout` ≥ 900 |
| Registration hangs at "Wrapping N KV cache tensors for IPC" | server transfer mode ≠ connector mode | use default (auto) on both; don't set server `--supported-transfer-mode non_gpu` with an auto connector |
| Pool token count ~5× too small | `--max-model-len` pinned small | use `--max-model-len auto` |
| Run A batch huge / no memory pressure | util too high or `L` too short | lower `--gpu-memory-utilization` or raise `L` |
| `exit 144` from a kill; leaked GPU memory after teardown | `pkill -f "vllm serve <model>"` self-matched / orphaned workers | kill by PID (main + worker PIDs from `nvidia-smi`) |
| Engine load fails: "CUDA compiler and CUDA toolkit headers are incompatible" (DeepSeek-V4) | sparse-MLA JIT can't find a matching `nvcc` | `export CUDA_HOME=/usr/local/cuda-<ver> PATH=$CUDA_HOME/bin:$PATH` matching the torch CUDA build, for every `vllm serve` |
| Engine dies: "CuTe DSL split sparse-attn wrapper only supports C128 layout block_size=8" | tried to disable the allocator on DeepSeek-V4 | don't — the allocator is mandatory; use the 3-run ladder (no Run A) |
| D hit ~100 % but no +PC saturation / nothing for LMCache to serve (fp8 KV models) | fp8 GPU pool is huge, fits the whole working set on-GPU | lower `--gpu-memory-utilization` (≈0.35) until the GPU pool is below the working set |

## Flag cheat-sheet

- **Hybrid allocator off:** `--disable-hybrid-kv-cache-manager` (sliding-window/dense only; invalid for Mamba/GDN).
- **Prefix caching off / on:** `--no-enable-prefix-caching` / `--enable-prefix-caching`.
- **Deterministic decode-bound run:** `--ignore-eos --ldqa-output-length <N>` (both round-trip through `--export-config`).
- **Pool sizing:** `--max-model-len auto`; `lmcache server --l1-size-gb <GB> --eviction-trigger-watermark 0.95`; start the server early to pre-pin.
- **LMCache wiring (custom port + slow-reg timeout):** `--kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.port":5560,"lmcache.mp.mq_timeout":900}}'`.
- **Mamba/GDN extras:** `--mamba-cache-mode align`, `--max-num-batched-tokens <2N-1>`, server `--chunk-size <N>`.
- **DeepSeek-V4 sparse-MLA extras:** `export CUDA_HOME=/usr/local/cuda-<ver>` (JIT, every launch); `--kv-cache-dtype fp8_ds_mla --trust-remote-code --tokenizer-mode deepseek_v4 --enable-expert-parallel`; util ≈0.35; large `--l1-size-gb`.
- **Hit-rate metrics:** vLLM `:8000/metrics` — `prefix_cache_*` (GPU-local), `external_prefix_cache_*` (LMCache). The LMCache `:9090` Prometheus endpoint is often disabled.
- **Teardown:** by PID (main + `nvidia-smi` worker PIDs); kill the server cleanly; verify GPUs ~0 MiB.
