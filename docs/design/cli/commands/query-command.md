# `lmcache query` CLI Command Design

**Status:** Proposal  |  **Date:** 2026-03-20

## Goal
Provide a formal single-shot query interface for the serving engine, the KV
cache worker, and the MP coordinator, with metrics output. Besides the normal
request query to a serving engine, offer the ability to query the detailed
KV cache info by the request prompt and to read the coordinator's read-only
HTTP APIs (usage, instances, quota, directory, health, prefetch, metrics)
without going through `curl`.
 

---

## Design Principles

### Single-shot, metrics-first command

`lmcache query` performs exactly one request and reports latency + result metrics
through the shared metrics framework (`BaseCommand.create_metrics()`), so users
can choose `--format terminal` or `--format json`.

### Three targets with one verb

`query` has three second-level targets:

- `query engine`: run one inference request and measure TTFT/TPOT/throughput.
- `query coordinator`: read one of the MP coordinator's read-only HTTP APIs
  (`usage`, `instances`, `health`, `directory`, `keys`, `quota`,
  `quota-config`, `prefetch`, `metrics`) and render it as an aligned
  metrics report.
- `query kvcache`: inspect cache coverage for one prompt (lookup).
 
### Script-friendly output and behavior

- `--format json` produces machine-readable metrics.
- `--output` writes the same formatted result to file.
- Exit codes: `0` success, `1` error.
- Errors go to stderr, metrics go to stdout.

### Prompt corpora support

`query engine` and `query kvcache` accept prompt templates like `{ffmpeg}`
and `{paul_graham}`, using the shared corpora expansion mechanism described
in `commands.md`. `query coordinator` takes no prompt.

---

## Command Overview

```text
lmcache query
├── engine       # Single inference query with latency/token metrics
├── coordinator  # Read one of the MP coordinator's read-only HTTP APIs
└── kvcache      # Single request cache lookup or round-trip verification
```

```bash
$ lmcache query -h
usage: lmcache query [-h] {engine,coordinator,kvcache} ...

Run one query and report metrics.

subcommands:
  engine       Run one inference request and report TTFT/TPOT metrics
  coordinator  Query the MP coordinator's read-only HTTP APIs
  kvcache      Query KV cache coverage or run store-retrieve round-trip
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

 

### `query kvcache`

Two modes under one command:

1. **Lookup mode (default):** tokenize prompt and query cache coverage.
 
```bash
# Lookup mode
$ lmcache query kvcache --url http://localhost:5555 \
    --prompt "{ctx} What is the example usage of lmcache?" \
    --documents ctx=LMCache/lmcache/cli/documents/lmcache.txt  \
    --model meta-llama/Llama-3.1-8B-Instruct

======== Query KV Cache Result ==========
Prompt tokens:                           8192
Cached chunks:                       30/32 (93.8%)
Cache locations:               [cpu=12, disk=0, ...]
Cached tokens:                         7680/8192
Cache status:                       HIT (partial)
=========================================
```
 

#### Proposed flags

| Flag | Description |
|------|-------------|
| `--url` | KV cache HTTP endpoint (`http://host:port`) |
| `--prompt` | Prompt for tokenization + lookup |
| `--model` | Tokenizer/model used to derive token IDs |
| `--documents name=path` | Register custom documents template |

#### Output metrics (lookup mode)

- `prompt_tokens`
- `cached_chunks_hit`
- `cached_chunks_total`
- `cached_chunk_location`
- `cached_tokens_hit`
- `cached_tokens_total`
- `cache_status` (`HIT`, `MISS`, `HIT (partial)`)


### `query coordinator`

Reads one of the MP coordinator's read-only HTTP APIs and prints it through
the shared metrics framework. Pick the API with `--api`; the default `--url`
is `http://127.0.0.1:9300`.

```bash
$ lmcache query coordinator --api usage

============== Coordinator: usage ==============
instance        compartment      used  capacity    ratio
--------------------------------------------------------
mp-gpu7         l1/dram      48.00 GB  64.00 GB    75.0%
mp-gpu8         l1/dram       2.00 GB  64.00 GB     3.1%
mp-gpu7         l2/fs        12.00 GB        --  unknown
(fleet-shared)  l2/s3         7.00 GB        --  unknown
================================================
```

#### Proposed flags

| Flag | Description |
|------|-------------|
| `--api` | Which API to read (`usage`, `instances`, `health`, `directory`, `keys`, `quota`, `quota-config`, `prefetch`, `metrics`) |
| `--url` | Coordinator base URL (default: `http://127.0.0.1:9300`) |
| `--instance` | Instance id; narrows `--api usage`, required for `--api prefetch` |
| `--cache-salt` | Cache salt; narrows `--api quota` to one tenant |
| `--request-id` | Prefetch request id; required for `--api prefetch` |
| `--limit` | Rows to request for `--api keys` (default: 20) |

Only reads are exposed. Mutating routes are either server-to-coordinator
plumbing (`POST /events`, `POST /instances`, heartbeats) or belong to a
command that owns the action -- e.g. quotas are written with
`lmcache quota`.


---

## API Surface and Dependencies

### `query engine`

Uses inference engine HTTP APIs (OpenAI-compatible or engine-native endpoint),
then computes CLI-side metrics from the single response stream/non-stream result.

No new dependencies required: use stdlib `urllib.request` and existing helpers.

### `query coordinator`

Reads the coordinator's read-only HTTP APIs (`/instances`, `/instances/usage`,
`/healthz`, `/directory/*`, `/quota*`, `/cache/prefetches/*`, `/metrics`).
Bindings and per-API render helpers live in
`lmcache/cli/commands/query/_coordinator.py`. Uses stdlib HTTP only.

### `query kvcache`

All `lmcache query kvcache` CLI operations go through HTTP, using either the
per-instance HTTP server or the controller HTTP server.
 

---

## Implementation

- **Single `QueryCommand`** (`BaseCommand` subclass) with second-level
  subparsers (`engine`, `coordinator`, `kvcache`) in
  `lmcache/cli/commands/query/`.
- **`query engine`:** `PromptBuilder` (`lmcache/cli/prompt.py`) expands `{name}`
  placeholders from `--documents`; top-level metrics include model plus per-slot
  token estimates (e.g. prompt documents, prompt query). `Request`
  (`lmcache/cli/request.py`) streams an OpenAI-compatible `/v1/chat/completions`
  or `/v1/completions` request; **Latency Metrics** repeats server usage (labeled
  **Input tokens**, not a duplicate client-side total).
- **`query coordinator`:** `CoordinatorApi` bindings in
  `lmcache/cli/commands/query/_coordinator.py` map `--api` to the coordinator
  URL path and a render function; the command normalizes `--url` (adds
  `http://` if missing, strips a trailing slash), validates required flags
  per API, and dispatches through `BaseCommand.create_metrics()` -- except
  `--api metrics`, which passes the Prometheus text through verbatim to
  stdout.
- **`query kvcache`:** stub; no handler yet.
- **Errors:** `query_engine` catches `RuntimeError` / `ValueError`, prints the
  message to stderr, exits `1`; `query_coordinator` exits `2` if required
  API-specific flags are missing; unknown `query_target` prints to stderr and
  exits `1`.

---

## Phasing

| Phase | Work |
|-------|------|
| **1a** | `query engine` with prompt, max-tokens, TTFT/TPOT/throughput metrics |
| **1b** | `query kvcache` lookup mode (prompt tokenization + cache coverage) |
| **1c** | `query coordinator` read-only APIs (`usage`, `instances`, `health`, `directory`, `keys`, `quota`, `quota-config`, `prefetch`, `metrics`) |
| **future** | richer query diagnostics (per-chunk detail) |

