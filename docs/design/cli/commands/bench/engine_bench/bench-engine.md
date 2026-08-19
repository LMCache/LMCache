# `lmcache bench engine` — Design & Extension Guide

**Status:** Implemented  |  **Date:** 2026-05-05

## Overview

`lmcache bench engine` runs sustained benchmarks against an OpenAI-compatible
inference engine. It ships six workloads exercising different caching
patterns; shared modules handle request sending, stats, and live progress
while each workload controls its own scheduling.

If `--engine-url`, `--workload`, or either `--tokens-per-gb-kvcache` /
`--lmcache-url` is missing, the command drops into a guided **interactive
TUI**. `--config FILE` loads a previously-exported config and skips the TUI;
`--no-interactive` errors instead of prompting; `--export-config FILE` writes
the resolved config and exits.

```bash
# Long-document Q&A (semaphore-controlled concurrency)
lmcache bench engine --engine-url http://localhost:8000 \
    --workload long-doc-qa --tokens-per-gb-kvcache 6000

# Multi-round chat (QPS-controlled dispatch)
lmcache bench engine --engine-url http://localhost:8000 \
    --workload multi-round-chat --tokens-per-gb-kvcache 6000 \
    --mrc-qps 2.0 --mrc-duration 120

# Random prefill (all requests at once, 1-token output)
lmcache bench engine --engine-url http://localhost:8000 \
    --workload random-prefill --tokens-per-gb-kvcache 6000 --rp-num-requests 100

# Long-doc permutator (blended-prefix cache reuse stress test)
lmcache bench engine --engine-url http://localhost:8000 \
    --workload long-doc-permutator --tokens-per-gb-kvcache 6000 \
    --ldp-num-contexts 5 --ldp-num-permutations 20

# Prefix-suffix tuner (tiered KV-cache demonstrator, sequential 2-pass)
lmcache bench engine --engine-url http://localhost:8000 \
    --workload prefix-suffix-tuner --lmcache-url http://localhost:8080 \
    --psf-context-length 8000 --psf-prefix-ratio 0.8 --psf-thrash 100

# RAG answer quality (real QA data, documents cached individually)
lmcache bench engine --engine-url http://localhost:8000 \
    --workload rag-qa-quality --tokens-per-gb-kvcache 6000 \
    --rag-dataset musique --rag-num-samples 50 --rag-output cached.json
```

---

## 1. Architecture

```
lmcache/cli/commands/bench/
├── __init__.py                    # BenchCommand (CLI registration)
└── engine_bench/
    ├── config.py                  # EngineBenchConfig, auto-detection helpers
    ├── stats.py                   # RequestResult, StatsCollector, FinalStats
    ├── request_sender.py          # RequestSender (async streaming)
    ├── progress.py                # ProgressMonitor (live terminal display)
    ├── tokenizers.py              # TokenPool, single-token word pools
    ├── interactive/               # Guided TUI (schema, state, terminal)
    ├── quality/                   # Answer-quality measurement
    │   ├── dataset.py             # Sample, hub registry, schema adapters
    │   └── scoring.py             # F1, answer extraction, QualityAggregator
    └── workloads/
        ├── __init__.py            # create_workload() factory
        ├── base.py                # BaseWorkload (ABC + run loop), MetricSection
        ├── long_doc_permutator.py
        ├── long_doc_qa.py
        ├── multi_round_chat.py
        ├── prefix_suffix_tuner.py
        ├── rag_qa_quality.py
        └── random_prefill.py
```

All concrete workloads depend on `BaseWorkload`, `RequestSender`,
`StatsCollector`, and `ProgressMonitor` — never on each other.

---

## 2. Core Modules

### 2.1 `config.py`

`EngineBenchConfig` holds only general parameters (`engine_url`, `model`,
`workload`, `kv_cache_volume_gb`, `tokens_per_gb_kvcache`, `seed`,
`output_dir`, `export_csv`, `export_json`, `quiet`, `ignore_eos`).
Workload-specific configs live in their own modules.

| Function | Purpose |
|----------|---------|
| `parse_args_to_config(args)` | CLI args → fully-resolved config |
| `auto_detect_model(engine_url)` | Fetches model ID from `/v1/models` |
| `resolve_tokens_per_gb(lmcache_url, model)` | Queries LMCache `/status` for `cache_size_per_token * world_size` |

### 2.2 `stats.py`

- `RequestResult` — one request: `successful`, `ttft`, `request_latency`,
  `num_input_tokens`, `num_output_tokens`, `decode_speed`, timestamps, `error`.
- `AggregatedStats` — running totals: request counts, elapsed, mean TTFT /
  decode speed / latency, input & output throughput, token totals.
- `FinalStats` — extends it with p50/p90/p99 for TTFT, decode speed, latency.

`StatsCollector` is **thread-safe** (`threading.Lock`):

| Method | Called by |
|--------|-----------|
| `on_request_finished(result)` | RequestSender callback |
| `get_current_stats()` | ProgressMonitor (every 1s) |
| `get_final_stats()` | Orchestrator, after the benchmark |
| `reset()` | BaseWorkload, between warmup and benchmark |
| `export_csv(path)` / `export_json(path, config)` | Orchestrator |

### 2.3 `request_sender.py`

```python
OnFinishedCallback = Callable[[RequestResult, str], None]

class RequestSender:
    def __init__(self, engine_url, model, completions_mode=False,
                 on_finished=[], ignore_eos=False, extra_body={})
    async def send_request(self, request_id, messages, max_tokens=128) -> RequestResult
    async def send_warmup_request(self, request_id, messages, max_tokens=1) -> RequestResult
    async def close(self) -> None
```

Streams via `AsyncOpenAI`, measures TTFT / decode speed / latency, reads token
counts from server usage reports, then invokes every `on_finished` callback
with `(RequestResult, response_text)`. Each call is a self-contained
coroutine — concurrency is the workload's job.

`ignore_eos` and `extra_body` are merged into the request body; `extra_body`
carries options the OpenAI client has no parameter for, notably
`chat_template_kwargs`. Neither is sent when unset, so no vLLM-specific field
reaches a non-vLLM backend.

### 2.4 `progress.py`

`ProgressMonitor` runs a daemon thread redrawing every second via ANSI cursor
control, reading `StatsCollector.get_current_stats()`. Tracks in-flight count
and the last 5 log messages. No-op when `quiet=True`.

Methods: `start()`, `stop()`, `on_request_sent(id)`,
`on_request_finished(id, successful)`, `log_message(msg)`.

### 2.5 `workloads/base.py`

```python
class BaseWorkload(ABC):
    def __init__(self, request_sender, stats_collector, progress_monitor)

    # --- Must implement ---
    @abstractmethod async def warmup(self) -> None
    @abstractmethod async def step(self, time_offset: float) -> float
    @abstractmethod def log_config(self) -> None
    @abstractmethod def on_request_finished(self, request_id: str, output: str) -> None

    # --- Provided ---
    def run(self) -> None                                    # entry point (blocks)
    def request_finished(self, result, text)                 # thread-safe queue bridge
    def extra_metric_sections(self) -> list[MetricSection]   # default: []
```

**`run()` loop:**

```
log_config()  →  warmup()  →  stats_collector.reset()
loop:
    drain_finished_queue() → on_request_finished()
    next_wakeup = step(time_offset)
    if next_wakeup < 0: break
    sleep until next_wakeup
drain_finished_queue()
```

**`step()` contract:** returns the absolute time offset (from benchmark start)
at which to be called again, or a negative value when complete.

**`request_finished()`** matches `OnFinishedCallback` and is registered on the
sender by the orchestrator. It enqueues `(request_id, response_text)`; the
loop thread drains it, so workloads never handle cross-thread concerns.

**`extra_metric_sections()`** lets a workload add its own sections to the
final report. `StatsCollector` covers what every workload has in common; a
workload measuring something else returns `MetricSection(key, label, entries)`
values, rendered through the same metrics system into terminal *and* JSON.

### 2.6 `workloads/__init__.py`

`create_workload(config, args, sender, collector, monitor) -> BaseWorkload`
dispatches on `config.workload`, resolves workload config from `args`, and
returns an instance ready to `run()`.

---

## 3. End-to-End Flow

`engine_bench.command.run_engine_bench()`:

```
0. _resolve_args()      → --config file | --no-interactive | TUI | pass through
1. parse_args_to_config()   (--export-config: write JSON and return)
2. StatsCollector()
3. ProgressMonitor(collector, quiet)
4. RequestSender(engine_url, model, ignore_eos, extra_body)
5. create_workload(...) → workload
6. Wire sender callbacks:
     stats_collector.on_request_finished  |  progress_monitor.on_request_finished
     workload.request_finished
7. workload.log_config()   → before the live display starts
8. progress_monitor.start()
9. workload.run()          → blocks
10. progress_monitor.stop()  →  11. request_sender.close()
12. Emit final metrics (incl. workload.extra_metric_sections())
13. Export CSV / JSON       →  14. sys.exit(1) if any failures
```

```
Workload.step() → send_request() → RequestSender (streams SSE)
                                        │
                          on_finished callbacks
                          ├── stats_collector.on_request_finished(result)
                          ├── progress_monitor.on_request_finished(id, ok)
                          └── workload.request_finished(result, text)
                                        ↓ finished_queue
                          loop drains → workload.on_request_finished(id, text)
```

---

## 4. Workloads

### 4.1 `long-doc-qa`

Repeated questions over long synthetic documents; tests prefix reuse.

| Field | CLI arg | Default |
|-------|---------|---------|
| `document_length` | `--ldqa-document-length` | 10000 |
| `query_per_document` | `--ldqa-query-per-document` | 2 |
| `num_documents` | computed | `kv_cache_volume * tokens_per_gb / document_length` |
| `shuffle_policy` | `--ldqa-shuffle-policy` | `random` (or `tile`) |
| `num_inflight_requests` | `--ldqa-num-inflight-requests` | 3 |
| `max_output_length` | `--ldqa-max-output-length` | 128 |

**Warmup:** each document once (`max_tokens=1`). **Dispatch:**
semaphore-controlled; `step()` acquires, fires an async task, returns `0.0`.
**`on_request_finished`:** no-op. **Termination:** `-1.0` when the schedule is
exhausted and all tasks are done.

### 4.2 `multi-round-chat`

Concurrent chat users with growing conversation history.

| Field | CLI arg | Default |
|-------|---------|---------|
| `shared_prompt_length` | `--mrc-shared-prompt-length` | 2000 |
| `chat_history_length` | `--mrc-chat-history-length` | 10000 |
| `user_input_length` | `--mrc-user-input-length` | 50 |
| `output_length` | `--mrc-output-length` | 200 |
| `qps` | `--mrc-qps` | 1.0 |
| `duration` | `--mrc-duration` | 60.0 |
| `num_concurrent_users` | computed | `kv_cache_volume * tokens_per_gb / (prompt + history)` |

**Dispatch:** QPS-controlled at `1/qps` intervals, round-robin over sessions;
returns `time_offset + 0.01` to retry when the target session is busy.
**`on_request_finished`:** **stateful** — records the answer via
`Session.record_answer()`, marking the session ready for its next request.
**Termination:** `-1.0` once past `duration` with no pending tasks.

### 4.3 `random-prefill`

Raw prefill throughput; fires everything at once.

| Field | CLI arg | Default |
|-------|---------|---------|
| `request_length` | `--rp-request-length` | 10000 |
| `num_requests` | `--rp-num-requests` | 50 |

**Warmup:** none. **Dispatch:** first `step()` dispatches all requests
(`max_tokens=1`); later calls wait on `asyncio.wait(FIRST_COMPLETED)`.

### 4.4 `long-doc-permutator`

Stresses **blended** reuse by sending permutations of a fixed context set:
`[System Prompt] + [Doc_i1] + … + [Doc_iN]`. Most permutations share chunks
with earlier requests but rarely a prefix, exercising chunk-level lookup and
eviction.

| Field | CLI arg | Default |
|-------|---------|---------|
| `num_contexts` | `--ldp-num-contexts` | 5 |
| `context_length` | `--ldp-context-length` | 5000 (exact tokens) |
| `system_prompt_length` | `--ldp-system-prompt-length` | 1000 (`0` disables) |
| `num_permutations` | `--ldp-num-permutations` | 10 (capped at `N!`) |
| `vocab_size` | hardcoded in factory | 8000 |
| `num_inflight_requests` | `--ldp-num-inflight-requests` | 1 |

Each field tunes one stress axis: context boundaries, eviction pressure,
chunk homogeneity (hash collisions), prefix domination, concurrency.

**Exact lengths.** Contexts are built from a pool of words that each cost
exactly one token, so `context_length` words is `context_length` tokens in any
permutation order. Candidates come from vocabulary *keys*, covering byte-level
BPE (`Ġthe`) and SentencePiece (`▁the`) — `decode()` drops the SentencePiece
marker and would find nothing. WordPiece marks continuations rather than word
starts and is unsupported; the workload says so rather than guessing.

**Tokenizer is required** — without one a token-denominated length cannot be
honoured, so it raises rather than silently benchmarking a different operating
point. Loaded from `--model`; pass it explicitly when the engine reports a
name that is not a HF repo ID or local path.

**Permutations:** iterates `itertools.permutations` for small `N`; samples
into a `set` when `N!` far exceeds `num_permutations * 10`.

**Dispatch:** semaphore-controlled, as in 4.1. **Warmup:** one dummy request.

**`run()` override:** closes `RequestSender`'s HTTP client inside the same
`asyncio.run()` as the loop, so `asyncio.run()` cannot orphan open `httpx`
connections. The orchestrator's later `close()` then finds nothing to do.

### 4.5 `prefix-suffix-tuner`

One sequential workload run **unchanged** across three configurations to
demonstrate each cache tier:

| Baseline | Config | Overflowed tier | Expected pass-2 hits |
|----------|--------|-----------------|----------------------|
| 1 | vanilla vLLM (L0 only) | L0 (HBM) | none — cold prefill every request |
| 2 | vLLM + LMCache L1 + L2 | L1 (DRAM) | L2 prefix hits (suffix recomputed) |
| 3 | + CacheBlend | L1 (DRAM) | L2 prefix hits + CacheBlend suffix hits |

Pick `--psf-thrash` to match the tier to overflow. Request layout:

```
[prefix_i with unique-ID][random breaker][shared suffix]
```

- `num_prefixes` distinct prefixes, each opening with `PREFIX_<8-hex>` so its
  chained block hash differs from every other prefix.
- A **fresh random breaker** per request (32 tokens), defeating ordinary
  prefix caching past the prefix boundary.
- A **single shared suffix**, bit-identical across requests — the only surface
  CacheBlend can reuse, which is what the workload measures.

Bodies come from a pseudo-word pool (e.g. `"boko42"`) sampled with a different
per-component RNG offset, so chunk fingerprints don't collide across prefixes
and inflate the blend hit rate.

| Field | CLI arg | Default |
|-------|---------|---------|
| `context_length` | `--psf-context-length` | 8000 |
| `prefix_ratio` | `--psf-prefix-ratio` | 0.8, in (0.0, 1.0) |
| `thrash` | `--psf-thrash` | 20.0 GB of the targeted tier |
| `num_prefixes` | computed | `floor(thrash * 1.05 * tokens_per_gb / prefix_tokens)` |
| `prefix_tokens` | computed | `round(context_length * prefix_ratio)` |
| `suffix_tokens` | computed | remainder; errors if `< 100` |
| `breaker_tokens` | hardcoded | 32 |
| `_OVERFLOW_FACTOR` | module constant | 1.05 |

**Behavior:** strictly sequential (`step()` awaits inline). Pass 1 sends each
prefix once in pool order as warmup; pass 2 repeats in **identical order** and
is what final stats capture.
where `(i1, …, iN)` is one permutation of the `N` contexts. Most permutations
share *some* chunks with prior requests but rarely the same prefix, exercising
chunk-level cache lookup and eviction.

**Config** (`LongDocPermutatorConfig`):

| Field | CLI arg | Default | Description |
|-------|---------|---------|-------------|
| `num_contexts` | `--ldp-num-contexts` | 5 | Number of unique context documents (`N`) |
| `context_length` | `--ldp-context-length` | 5000 | Tokens per context (exact) |
| `system_prompt_length` | `--ldp-system-prompt-length` | 1000 | Shared system prompt tokens, exact (`0` disables) |
| `num_permutations` | `--ldp-num-permutations` | 10 | Distinct permutations to send (capped at `N!`) |
| `vocab_size` | (none — hardcoded in factory) | 8000 | Number of distinct single-token words contexts are sampled from |
| `num_inflight_requests` | `--ldp-num-inflight-requests` | 1 | Max concurrent in-flight requests |
| `max_output_length` | `--ldp-max-output-length` | 128 | Max tokens generated per request; `1` measures prefill alone |

**Stress axes** (each config field tunes one):

| Axis | Knob |
|------|------|
| Blended-context boundaries | `num_contexts` |
| Eviction pressure | `num_permutations` |
| Chunk homogeneity (hash collisions) | `vocab_size` |
| Prefix domination | `system_prompt_length` |
| Concurrency | `num_inflight_requests` |

**Behavior:**

- **Data generation:** Builds a deterministic pool of `vocab_size` words that
  each cost exactly one token under the model's tokenizer, generates
  `num_contexts` distinct contexts from it (each seeded independently so token
  sequences truly diverge), and enumerates permutations. Because every word is
  one token and begins a word, a context of `context_length` words is exactly
  `context_length` tokens whatever order a permutation puts them in — the
  configured lengths are exact, not estimates.
- **Tokenizer families.** Candidates are read from the vocabulary's *keys*, so
  byte-level BPE (`Ġthe`) and SentencePiece (`▁the`) are both covered;
  `decode()` would drop the SentencePiece marker and find nothing. Words are
  then joined by whichever convention that tokenizer makes exact — each word
  carrying a leading space, or plain separators for tokenizers that charge a
  token for an explicit leading space — and the total is checked before use.
  WordPiece (BERT) marks continuations rather than word starts and is not
  supported; the workload says so instead of guessing.
- **Tokenizer is required.** Without one the workload cannot honour a length
  expressed in tokens, so it raises instead of falling back to a text-level
  approximation, which would silently benchmark a different operating point
  than the flags describe. The tokenizer is loaded from `--model`, which
  defaults to the name the engine reports from `/v1/models`; pass `--model`
  explicitly when that name is not a HuggingFace repo ID or a local path.
- **Permutation enumeration:** For small `N`, iterates `itertools.permutations`
  and truncates. When `N!` is much larger than `num_permutations * 10`, samples
  random permutations into a `set` to avoid exhausting an enormous search
  space. Returns all `N!` permutations when `num_permutations >= N!`.
- **Warmup:** A single dummy request (`max_tokens=1`) to prime the engine.
- **Dispatch:** Semaphore-controlled — `step()` acquires the semaphore, fires
  an async task with the next permutation, returns `0.0` for immediate
  re-call. Once all permutations are dispatched, awaits remaining tasks via
  `asyncio.wait(FIRST_COMPLETED)`.
- **`on_request_finished`:** No-op (stateless).
- **Termination:** Returns `-1.0` when the request list is exhausted and all
  pending tasks have completed.

**`run()` override:** Unlike the other workloads, `LongDocPermutatorWorkload`
overrides `BaseWorkload.run()` to close `RequestSender`'s async HTTP client
inside the same `asyncio.run()` call as the benchmark loop. `asyncio.run()`
closes the loop on exit, which would orphan any open `httpx` connections;
closing the client here ensures clean teardown. The orchestrator's subsequent
`asyncio.run(request_sender.close())` then finds nothing to close and
completes without error.

### 4.5 `prefix-suffix-tuner` — Tiered KV-Cache Demonstrator

A single sequential workload designed to be run **unchanged** across three
LMCache configurations to demonstrate the value of each cache tier:

| Baseline | LMCache config | Targeted overflow | Expected pass-2 hits |
|----------|---------------|-------------------|----------------------|
| 1 | vanilla vLLM (L0 only) | L0 (HBM) | none — every request a cold prefill |
| 2 | vLLM + LMCache L1 + L2 | L1 (DRAM) | L2 prefix hits (suffix recomputed) |
| 3 | vLLM + LMCache L1 + L2 + CacheBlend | L1 (DRAM) | L2 prefix hits + CacheBlend suffix hits |

**Why 1.05× suffices:** with sequential dispatch and LRU in a tier of capacity
`K`, after pass 1 of `N = 1.05K` prefixes the `0.05K` oldest are evicted. Pass
2's access of prefix `0` misses, and serving it evicts prefix `0.05K` — the
very next one needed. That repeats for the whole pass, so every measured
request falls through to the next tier without overprovisioning by 2×.

`--kv-cache-volume` is unused by this workload.

### 4.6 `rag-qa-quality`

The only workload measuring **correctness** rather than speed — every other
one would report an unchanged number if the cache returned subtly wrong KV.

Documents are prefilled individually during warmup, so each is stored on its
own. The measured request then composes them:

```
[system prompt][doc_a][doc_b]…[doc_n][question]
```

Every document is therefore reused at a position it was never cached at — the
RAG serving pattern. Unlike forcing the same effect with filler tokens, the
measured request is a prompt the deployment would really receive, so a quality
change is attributable to the cache rather than to the perturbation.

| Field | CLI arg | Default |
|-------|---------|---------|
| `dataset` | `--rag-dataset` | **required** |
| `num_samples` | `--rag-num-samples` | 50 |
| `max_output_length` | `--rag-max-output-length` | 1024 |
| `doc_align_tokens` | `--rag-doc-align-tokens` | 256 (LMCache's default chunk size) |
| `template_kwargs` | `--rag-template-kwargs` | `{}` (repeatable `KEY=VALUE`) |
| `output_path` | `--rag-output` | `<output-dir>/rag_qa_quality.json` |

`--kv-cache-volume` is unused.

**Datasets** are not vendored. Named ones download from the HF Hub on first
use and cache under `HF_HOME`; any other value is a local path.

| Name | Source | Notes |
|------|--------|-------|
| `musique` | `dgslibisey/MuSiQue` | JSONL, 20 passages/question, ~2.1k tokens |
| `hotpotqa` | `hotpotqa/hotpot_qa` | Parquet (distractor/validation), needs `pyarrow` |

`load_samples` recognizes four record schemas, so most QA files on disk load
unchanged: `ctxs[].{title,text}`, MuSiQue's `paragraphs[].paragraph_text`, and
HotpotQA's `context` as either a `{title, sentences}` struct or
`[[title, [sentence, …]], …]` pairs. Records missing passages, a question, or
gold answers are skipped.

**Document alignment.** Each document is padded to a multiple of
`--rag-doc-align-tokens`, and the system prompt is padded so the
chat-template prefix plus system block is also a whole multiple. Documents
then start on a chunk boundary in both the prefill and the composite, so
their chunks hold document content alone and match. Without this they land
off-phase and nothing matches, silently.

**Set it to the deployment's LMCache chunk size** — the default 256 is
LMCache's own default, not a detected value. It is configured rather than
queried because the baseline stack a run is compared against has no LMCache
server to ask, and two runs that padded differently would no longer share
prompts. A mismatch yields partial reuse, and both runs being compared must
use the same value — it is part of the `run_fingerprint`.

Nothing here reads LMCache's own counters: the workload measures answers, and
its output is deliberately agnostic of engine-side metrics. Confirm the cache
was exercised from the engine's own `/metrics` if a run needs that evidence.

**Scoring.** The model wraps its answer in `<final_answer>…</final_answer>`;
the *last complete* region is taken, since reasoning models may echo an
example while thinking. An unterminated region counts as no answer —
generation was cut off, so scoring the reasoning before it would report an
answer never produced. Answers are scored by SQuAD-normalized token-overlap
F1, best over the gold answers.

`f1_mean` covers **parsed samples only** and is always reported beside
`parse_rate`: a high F1 over a third of the samples is a different result from
the same F1 over all of them.

**Reasoning models.** `--rag-template-kwargs` is deliberately not defaulted.
Disabling thinking is not a safe universal choice — on multi-hop QA it can
lower quality, compressing the range a cache regression must show up in.
Bounding runaway reasoning (`reasoning_effort=high`) or enabling it
(`thinking_mode=enabled`) is model-specific. Unset, the model's template
default applies.

**Behavior:** warmup sends each distinct document once behind the system block
(`max_tokens=1`; shared passages prefill once, stats discarded). `step()` is
strictly sequential: await the composite request, drain the finished queue for
its text, score.
`on_request_finished` holds a measured request's text until its step scores it.

**Comparing two stacks.** The workload reports one arm. Run it twice — once
per stack — and diff by `sample_id`. An unparsed sample's `f1` is `null`, not
`0.0`, so a diff can pair by id and skip what either run failed to score. The
files are comparable only when their `run_fingerprint` values match; the
fingerprint covers dataset, sample ids, budget, template kwargs, model, and
alignment unit.

Start each run from a clean cache state — restart the server, or
`lmcache kvcache clear --url <mp-url>` — or a previous run's composites are
still cached and the second run measures a full prefix hit instead of
per-document reuse.

---

## 5. Adding a New Workload

**1. Create `workloads/my_workload.py`:**

```python
@dataclass
class MyWorkloadConfig:
    my_param: int = 100

    def __post_init__(self) -> None:
        if self.my_param <= 0:
            raise ValueError(f"my_param must be positive, got {self.my_param}")

    @classmethod
    def resolve(cls, kv_cache_volume_gb, tokens_per_gb_kvcache, **kwargs):
        """Compute derived fields from the KV cache budget + CLI args."""


class MyWorkload(BaseWorkload):
    def __init__(self, config, request_sender, stats_collector,
                 progress_monitor, seed=42):
        super().__init__(request_sender, stats_collector, progress_monitor)

    def log_config(self) -> None: ...          # print(); runs before the display
    async def warmup(self) -> None: ...
    async def step(self, time_offset: float) -> float: ...
    def on_request_finished(self, request_id: str, output: str) -> None: ...
```

**2. Register in `workloads/__init__.py`** — add the import, extend
`_WORKLOAD_NAMES`, and add a `create_workload` branch.

**3. Add CLI args in `command.py`** — add the name to `--workload` choices and
a new argument group. All workload args must carry a short prefix (`ldqa-`,
`ldp-`, `mrc-`, `psf-`, `rag-`, `rp-`) to avoid collisions. Arguments with no
default belong in `_REQUIRED_WORKLOAD_ARGS`, and their `ConfigItem` goes in
the required phase of `interactive/schema.py` (a required item outside that
phase fails the schema test).

**4. Add tests** in `tests/.../workloads/test_my_workload.py` covering config
validation and resolution, data generation, `warmup()`, `step()` dispatch and
return values, `on_request_finished()`, and `run()` end-to-end with a mocked
sender. Add a factory case to `test_create_workload.py`.

**Key constraints:**

- `step()` must not block indefinitely — dispatch or wait briefly, then
  return; the loop handles sleeping.
- `on_request_finished()` runs on the loop thread via the queue drain, so no
  locking is needed inside a workload.
- `log_config()` uses `print()`, not `log_message()` — it runs before the
  progress monitor starts.
- Use `progress_monitor.log_message()` for runtime logging, to avoid
  corrupting the live display.
- Warmup stats are discarded (`reset()` runs after warmup).

---

## 6. Tests

```bash
pytest tests/cli/commands/bench/                       # all bench tests
pytest tests/cli/commands/bench/engine_bench/workloads # workloads
pytest tests/cli/commands/bench/engine_bench/quality   # scoring, dataset, probe
pytest tests/cli/commands/bench/test_bench_command.py  # CLI + orchestrator
```
