# `lmcache query` CLI Command Design

**Status:** Proposal  |  **Date:** 2026-03-20

## Goal
Provide a formal single-shot query interface for both the serving engine and KV cache worker, with metrics output. besides normal request query to serving engine, offers the feature to query the detailed KV cache info by the request prompt.
 

---

## Design Principles

### Single-shot, metrics-first command

`lmcache query` performs exactly one request and reports latency + result metrics
through the shared metrics framework (`BaseCommand.create_metrics()`), so users
can choose `--format terminal` or `--format json`.

### Two targets with one verb

`query` has two second-level targets:

- `query engine`: run one inference request and measure TTFT/TPOT/throughput.
- `query kvcache`: inspect cache coverage for one prompt (lookup).
 
### Script-friendly output and behavior

- `--format json` produces machine-readable metrics.
- `--output` writes the same formatted result to file.
- Exit codes: `0` success, `1` error.
- Errors go to stderr, metrics go to stdout.

### Prompt corpora support

Both subcommands accept prompt templates like `{ffmpeg}` and `{paul_graham}`,
using the shared corpora expansion mechanism described in `commands.md`.

---

## Command Overview

```text
lmcache query
├── engine    # Single inference query with latency/token metrics
└── kvcache   # Single request cache lookup or round-trip verification
```

```bash
$ lmcache query -h
usage: lmcache query [-h] {engine,kvcache} ...

Run one query and report metrics.

subcommands:
  engine      Run one inference request and report TTFT/TPOT metrics
  kvcache     Query KV cache coverage or run store-retrieve round-trip
```

---

## Commands in Detail

### `query engine`

Send one inference request to an engine HTTP endpoint and report token/latency metrics; ``--prompt`` supports placeholders, where ``{lmcache}`` loads ``lmcache/cli/documents/lmcache.txt`` and custom documents use ``--documents NAME=PATH``.
 

```bash
# Single inference query
$ lmcache query engine --url http://localhost:8000/v1 \
     --prompt "{lmcache} Summarize LMCache usage." \
     --format terminal \
     --max-tokens 128
   
================= Query Engine =================
Model:                         facebook/opt-125m
Prompt documents lmcache:                    608
Prompt query:                                  9
--------------- Latency Metrics ----------------
Input tokens:                             618.00
Output tokens:                              9.00
TTFT (ms):                                 26.88
TPOT (ms/token):                            0.91
Total latency (ms):                        35.05
Throughput (tokens/s):                   1100.64
================================================
```

#### Proposed flags besides native engine query flags

| Flag | Description |
|------|-------------|
| `--url` | Engine HTTP endpoint (`http://host:port`) |
| `--prompt` | Prompt text, supports `{documents}` templates |
| `--timeout` | Request timeout in seconds (default: 30) |
| `--documents name=path` | Register custom documents template |



#### Output metrics
 
- `prompt_tokens`, `output_tokens`, `model`
- `ttft_ms`, `tpot_ms_per_token`, `total_latency_ms`, `throughput_tokens_per_s`

`output_tokens` uses the server's `completion_tokens` when present, including
zero. The request's `max_tokens` is used only when the count is absent.

### `query kvcache`

One mode is implemented: **lookup**. Write mode (store-then-retrieve
verification) is still future work.

```bash
$ lmcache query kvcache --url http://localhost:9300 \
    --prompt "{ctx} What is the example usage of lmcache?" \
    --documents ctx=LMCache/lmcache/cli/documents/lmcache.txt \
    --model meta-llama/Llama-3.1-8B-Instruct --world-size 4

================ Query KV Cache ================
Model:          meta-llama/Llama-3.1-8B-Instruct
Prompt tokens:                               8192
Cached chunks:                                 30
Total chunks:                                  32
Cached tokens:                               7680
Cache status:                       HIT (partial)
Cache locations:         inst-0: l1/cpu, inst-1: l2/disk
tier  chunks
----------------
l1        28
l2        30
================================================
```
 

#### Flags

| Flag | Description |
|------|-------------|
| `--url` | Coordinator base URL (`http://host:port`); scheme optional |
| `--prompt` | Prompt for tokenization + lookup |
| `--model` | Tokenizer/model used to derive token IDs, and the rank fan-out key scope |
| `--world-size N` | World size behind the per-rank key fan-out (default `1`) |
| `--cache-salt SALT` | Isolation salt the request was stored under |
| `--documents name=path` | Register custom documents template |
| `--timeout SECONDS` | HTTP timeout (default `10`) |

`--world-size` and `--cache-salt` are not in the original proposal: the
coordinator cannot resolve keys without them, and a wrong fan-out reads as a
miss rather than an error.

#### Output metrics (lookup mode)

- `prompt_tokens`
- `cached_chunks_hit`
- `cached_chunks_total`
- `cached_tokens_hit`
- `cache_status` (`HIT`, `MISS`, `HIT (partial)`)
- `cached_chunk_location` (distinct `instance: tier/backend` placements)
- `cached_chunks_by_tier` table (chunks per tier over the cached prefix)

---

## API Surface and Dependencies

### `query engine`

Uses inference engine HTTP APIs (OpenAI-compatible or engine-native endpoint),
then computes CLI-side metrics from the single response stream/non-stream result.

No new dependencies required: use stdlib `urllib.request` and existing helpers.

### `query kvcache`

One `POST /directory/lookup` to the MP coordinator, in its `token_ids` form.
The coordinator expands the token sequence into one object key per complete
chunk (chunk hashes are prefix-chained, so the sequence must be the request's
full prompt from position 0), and returns each key's placements, its token IDs
and its access count. The CLI needs no chunk-size flag: the server reports the
chunk count and each chunk's tokens.

Coverage is the leading run of chunks that every rank of every object group has
a placement for. A gap makes everything behind it unreachable for that prompt,
so later resident chunks are not counted.
 

---

## Implementation

- **Single `QueryCommand`** (`BaseCommand` subclass) with second-level
  subparsers (`engine`, `kvcache`) in `lmcache/cli/commands/query.py`.
- **`query engine`:** `PromptBuilder` (`lmcache/cli/prompt.py`) expands `{name}`
  placeholders from `--documents`; top-level metrics include model plus per-slot
  token estimates (e.g. prompt documents, prompt query). `Request`
  (`lmcache/cli/request.py`) streams an OpenAI-compatible `/v1/chat/completions`
  or `/v1/completions` request; **Latency Metrics** repeats server usage (labeled
  **Input tokens**, not a duplicate client-side total).
- **`query kvcache`:** `PromptBuilder` expands `--documents`; `try_load_tokenizer`
  (shared with the engine benchmark) turns the prompt into token IDs;
  `_lookup.py` posts them and summarizes the reply into a `CoverageResult`, so
  the HTTP and summary logic stay out of the command class.
- **Errors:** `query_engine` catches `RuntimeError` / `ValueError`, prints the
  message to stderr, exits `1`; unknown `query_target` prints to stderr and exits
  `1`. `query kvcache` does the same, and additionally exits `1` when no
  tokenizer is available — reporting zero coverage silently would be
  indistinguishable from an empty cache.

---

## Phasing

| Phase | Work |
|-------|------|
| **1a** | `query engine` with prompt, max-tokens, TTFT/TPOT/throughput metrics |
| **1b** | **Done.** `query kvcache` lookup mode: prompt tokenization + prefix coverage + per-tier chunk counts |
| **future** | round-trip verification (store then retrieve) and single-instance targeting |

