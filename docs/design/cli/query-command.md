# `lmcache query` CLI Command Design

**Status:** Proposal  |  **Date:** 2026-03-20

## Goal
Provide a formal single-shot query interface for both the serving engine and
KV cache worker, with metrics output. besides normal request query to serving engine, offers the feature to query the detailed KV cache info by the request prompt.
 

---

## Design Principles

### Single-shot, metrics-first command

`lmcache query` performs exactly one request and reports latency + result metrics
through the shared metrics framework (`BaseCommand.create_metrics()`), so users
can choose `--format terminal` or `--format json`.

### Two targets with one verb

`query` has two second-level targets:

- `query engine`: run one inference request and measure TTFT/TPOT/throughput.
- `query kvcache`: inspect cache coverage for one prompt (lookup) or run a
  store-retrieve round-trip with correctness check.

This matches the top-level CLI command model in
[commands.md](commands.md): one verb, different backends.

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

Send one inference request to an engine HTTP endpoint and report token and
latency metrics. 
#### Proposed flags besides native engine query flags

| Flag | Description |
|------|-------------|
| `--url` | Engine HTTP endpoint (`http://host:port`) |
| `--prompt` | Prompt text, supports `{corpus}` templates |
| `--timeout` | Request timeout in seconds (default: 30) |
| `--corpus name=path` | Register custom corpus template |

#### Output metrics

- `prompt_tokens`
- `output_tokens`
- `ttft_ms`
- `tpot_ms_per_token`
- `total_latency_ms`
- `throughput_tokens_per_s`

### `query kvcache`

Two modes under one command:

1. **Lookup mode (default):** tokenize prompt and query cache coverage.
 
```bash
# Lookup mode
$ lmcache query kvcache --url http://localhost:5555 \
    --prompt "{ffmpeg} What is the example usage of ffmpeg?" \
    --model meta-llama/Llama-3.1-8B-Instruct

======== Query KV Cache Result ==========
Prompt tokens:                           8192
Cached chunks:                       30/32 (93.8%)
Cache locations:               [cpu=12, disk=0, s3=0]
Cached tokens:                         7680/8192
Cache status:                       HIT (partial)
=========================================
```

2. **Round-trip mode (`--round-trip`):** measure store/retrieve latency for a round-trip operation, then verify checksum integrity.


```bash
# Round-trip mode
$ lmcache query kvcache --url http://localhost:5555 --round-trip

==== Query KV Cache Result (round-trip) ====
Store latency (ms):                      1.23
Retrieve latency (ms):                   0.87
Checksum:                                OK
============================================
```

#### Proposed flags

| Flag | Description |
|------|-------------|
| `--url` | KV cache HTTP endpoint (`http://host:port`) |
| `--prompt` | Prompt for tokenization + lookup |
| `--model` | Tokenizer/model used to derive token IDs |
| `--round-trip` | Switch to store-retrieve verification mode |
| `--chunk-size` | Override chunk size for synthetic round-trip payload |
| `--corpus name=path` | Register custom corpus template |

#### Output metrics (lookup mode)

- `prompt_tokens`
- `cached_chunks_hit`
- `cached_chunks_total`
- `cached_chunk_location`
- `cached_tokens_hit`

- `cached_tokens_total`
- `cache_status` (`HIT`, `MISS`, `HIT (partial)`)

#### Output metrics (round-trip mode)

- `store_latency_ms`
- `retrieve_latency_ms`
- `checksum_status`

---

## API Surface and Dependencies

### `query engine`

Uses inference engine HTTP APIs (OpenAI-compatible or engine-native endpoint),
then computes CLI-side metrics from the single response stream/non-stream result.

No new dependencies required: use stdlib `urllib.request` and existing helpers.

### `query kvcache`

All `lmcache query kvcache` CLI operations go through HTTP, using either the
per-instance HTTP server or the controller HTTP server.

ZMQ remains reserved for performance-critical data-path communication between
the inference engine and LMCache (store, retrieve, prefetch), not for CLI
query operations.

Some operations (for example, end-session) are currently implemented only over
ZMQ. These require new HTTP endpoints before they can be supported by the CLI.

---

## Implementation

- **Single `QueryCommand`** (`BaseCommand` subclass) with second-level
  subparsers (`engine`, `kvcache`), implemented in
  `lmcache/cli/commands/query.py`.
- **Shared metrics integration:** always construct output via
  `self.create_metrics(title, args, width=48)` to honor `--format` and `--output`.
- **Prompt expansion:** reuse shared corpus expansion helper used by
  `bench engine`/other prompt-capable commands.
- **Transport split by target:**
  - `engine` path uses HTTP client helper.
  - `kvcache` path uses HTTP client helper.
- **Error handling:** raise command errors with concise messages; dispatcher prints
  to stderr and returns exit code `1`.

---

## Phasing

| Phase | Work |
|-------|------|
| **1a** | `query engine` with prompt, max-tokens, TTFT/TPOT/throughput metrics |
| **1b** | `query kvcache` lookup mode (prompt tokenization + cache coverage) |
| **1c** | `query kvcache --round-trip` with checksum verification metrics |
| **future** | richer query diagnostics (per-chunk detail) |

