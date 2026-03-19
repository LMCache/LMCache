# `lmcache kvcache` CLI Command Design

**Status:** Proposal  |  **Date:** 2026-03-19

## Why

Users need a way to manage KV cache state for **specific requests** from the
command line — pin a request's cache to prevent eviction, compress it, clear it,
or warm the cache for an upcoming prompt. Part of **Phase 1** of the
[CLI design](commands.md).

---

## Design Principles

`lmcache kvcache` is a **per-request management tool**. Every sub-command
operates on a specific request's KV cache, identified by request ID or token
sequence.

### All management goes through HTTP

ZMQ is reserved for **performance-critical data-path** communication between the
inference engine and LMCache (store, retrieve, prefetch). Every `lmcache kvcache`
CLI operation goes through **HTTP** — either the per-instance HTTP server or the
controller HTTP server.

Today some operations (e.g. `end-session`) only have ZMQ implementations. These
need new HTTP endpoints before the CLI can use them.

### Every sub-command targets a specific request

Every sub-command requires one of these to identify the target KV cache:

- **`--request-id <id>`** (required) — identifies the request whose KV cache
  to operate on.
- **`--start <st> --end <ed>`** (optional) — narrow the operation to a token
  range `[st, ed)` within the request. Defaults to the full sequence.

The only exception is `generate`, which creates a new request rather than
targeting an existing one.

### Pipe- and script-friendly output

- **Exit codes:** `0` = success, `1` = error, `2` = rejected (e.g. pin rejected
  due to memory pressure). Scripts branch on `$?` without parsing output.
- **`--format json`:** Structured output for piping into `jq` (already exists).
- **`--format terminal`:** Human-readable ASCII table (default, already exists).
- **`--quiet` / `-q`:** Suppress all stdout. Exit code only.
- **Stdout vs stderr:** Metrics to stdout (pipeable). Errors to stderr.

---

## Command Overview

```
lmcache kvcache
├── info           # Per-request cache state (locations, pinned status)
├── clear          # Remove a request's cached KV data
├── pin            # Pin a request's KV cache to L1/CPU (may be rejected)
├── compress       # Compress a request's KV cache in-place
├── end-session    # Clean up session state for a finished request
└── generate       # Prefill a prompt via vLLM to populate the cache
```

| Sub-command | Target | Description |
|------------|--------|-------------|
| `info` | instance | Show per-request cache state: which chunks, where stored, pinned status |
| `clear` | instance or controller | Remove cached data for a specific request |
| `pin` | instance or controller | Pin a request's KV cache to L1/CPU; may be rejected if memory pressure is too high |
| `compress` | instance or controller | Compress a request's KV cache to reduce memory footprint |
| `end-session` | instance | Remove per-request session state (token hashes, chunk tracking) |
| `generate` | instance + vLLM | Send a prompt to a vLLM endpoint to trigger prefill and populate the cache |

---

## Commands in Detail

### `info`

Show the cache state for a specific request: which chunks exist, which storage
backend holds each one, and whether they are pinned.

Each chunk becomes an entry in a "Chunks" section, keyed by range. The value
is a summary string (location + flags). This fits the existing `Metrics` API:
`metrics["chunks"].add("0:256", "[0:256]", "L1, pinned")`.

```bash
# By request ID
$ lmcache kvcache info --url http://localhost:8000 --request-id req-abc-123

===== KV Cache Info (req-abc-123) =====
Total chunks:                         32
Pinned:                                8
---------------- Chunks ---------------
[0:256]:                  L1, pinned
[256:512]:                L1, pinned
[512:768]:                L1, L2
[768:1024]:               L2
...
========================================

# Narrowed to a token range
$ lmcache kvcache info --url http://localhost:8000 \
    --request-id req-abc-123 --start 0 --end 512

# JSON for scripting
$ lmcache kvcache info --url http://localhost:8000 \
    --request-id req-abc-123 --format json
{
  "title": "KV Cache Info (req-abc-123)",
  "metrics": {
    "total_chunks": 32,
    "pinned": 8,
    "chunks": {
      "0:256": "L1, pinned",
      "256:512": "L1, pinned",
      "512:768": "L1, L2",
      "768:1024": "L2"
    }
  }
}

# Find chunks on L2
$ lmcache kvcache info --url http://localhost:8000 \
    --request-id req-abc-123 --format json \
    | jq '.metrics.chunks | to_entries[] | select(.value | contains("L2"))'
```

### `clear`

Remove cached KV data for a specific request.

```bash
$ lmcache kvcache clear --url http://localhost:8000 --request-id req-abc-123

# Clear specific backends only
$ lmcache kvcache clear --url http://localhost:8000 \
    --request-id req-abc-123 --location LocalCPUBackend

# By token range
$ lmcache kvcache clear --url http://localhost:8000 \
    --request-id req-abc-123 --start 0 --end 512

# Via controller (targets specific instance)
$ lmcache kvcache clear --url http://localhost:9000 \
    --instance-id inst-0 --request-id req-abc-123
```

| Flag | Required | Description |
|------|----------|-------------|
| `--url` | yes | Target HTTP endpoint |
| `--request-id` | yes | Target request |
| `--start`, `--end` | no | Narrow to token range `[st, ed)` |
| `--location` | no | Restrict to specific backend(s) |
| `--instance-id` | no | Target instance (controller mode) |

### `pin`

Pin a request's KV cache chunks to L1 (CPU memory) to prevent eviction. The
server **may reject** the request if CPU memory pressure is too high.

Exit codes: `0` = pinned, `2` = rejected, `1` = error.

```bash
$ lmcache kvcache pin --url http://localhost:8000 --request-id req-abc-123
$ echo $?
0

# Quiet mode for scripts
if lmcache kvcache pin -q --url http://localhost:8000 --request-id req-abc-123; then
    echo "pinned"
else
    echo "rejected or error"
fi

# By token range
$ lmcache kvcache pin --url http://localhost:8000 \
    --request-id req-abc-123 --start 0 --end 512

# Rejected case (exit code 2)
$ lmcache kvcache pin --url http://localhost:8000 --request-id req-xyz
$ echo $?
2
```

### `compress`

Compress a request's KV cache chunks in-place to reduce memory footprint.

```bash
$ lmcache kvcache compress --url http://localhost:8000 \
    --request-id req-abc-123 --method zstd

$ lmcache kvcache compress --url http://localhost:8000 \
    --request-id req-abc-123 --start 0 --end 512 --method zstd
```

| Flag | Required | Description |
|------|----------|-------------|
| `--method` | yes | Compression method (e.g. `zstd`) |
| `--request-id` | yes | Target request |
| `--start`, `--end` | no | Narrow to token range `[st, ed)` |
| `--instance-id` | no | Target instance (controller mode) |

### `end-session`

Remove per-request session state from the engine. A session tracks the
accumulated token IDs and computed chunk hashes for a request. Call this after
an inference request completes to free the associated tracking resources.

```bash
$ lmcache kvcache end-session --url http://localhost:8000 --request-id req-abc-123
```

**Note:** Today `end-session` is ZMQ-only (`END_SESSION` in `RequestType`). A new
HTTP endpoint needs to be added to the per-instance server.

### `generate`

Trigger KV cache generation by sending a prompt to a vLLM inference endpoint.
The prompt is prefilled and the resulting KV cache is stored in LMCache. Useful
for warming the cache before production traffic arrives.

This is the only sub-command that does not require `--request-id`
— it creates a new cache entry rather than targeting an existing one.

```bash
$ lmcache kvcache generate \
    --target-url http://localhost:8080/v1/completions \
    --prompt "System prompt text here..." --max-tokens 1

$ lmcache kvcache generate \
    --target-url http://localhost:8080/v1/completions \
    --prompt-file ./system_prompt.txt --max-tokens 1

# Script: warm cache and check how many chunks were stored
$ CHUNKS=$(lmcache kvcache generate --format json \
    --target-url http://localhost:8080/v1/completions \
    --prompt-file ./system_prompt.txt | jq -r '.metrics.chunks_cached')
$ echo "Cached ${CHUNKS} chunks"
```

| Flag | Required | Description |
|------|----------|-------------|
| `--target-url` | yes | vLLM-compatible completions endpoint |
| `--prompt` or `--prompt-file` | yes | Prompt text or path to file |
| `--max-tokens` | no (default: 1) | Output tokens (1 = prefill only) |
| `--model` | no | Model name (if endpoint serves multiple) |

---

## Existing API Surface & Gaps

| CLI sub-command | Existing HTTP endpoint | Gap |
|----------------|----------------------|-----|
| `info` | `/cache/kvcache/info` (metadata only) | Need per-request chunk detail endpoint |
| `clear` | `/cache/clear` (instance), `/clear` (controller) | Need request-id and token-range filtering |
| `pin` | `/pin` (controller only) | Need per-instance HTTP pin endpoint with rejection |
| `compress` | `/compress` (controller only) | Need per-instance HTTP compress endpoint |
| `end-session` | ZMQ `END_SESSION` only | Need HTTP endpoint on per-instance server |
| `generate` | N/A (client-side) | No server endpoint needed — CLI sends request to vLLM directly |

---

## Implementation

- **Single `KVCacheCommand`** (`BaseCommand` subclass) with second-level
  argparse subparsers. File: `lmcache/cli/commands/kvcache.py`.
- **HTTP only:** `_http_request()` wraps `urllib.request` (no new deps).
- **Indexing args** shared via a helper that adds `--request-id`,
  `--start`, `--end` to each subparser (except `generate`).
- **Output:** `self.create_metrics()` — use `--format json | jq` for scripting.
- **New `--quiet` / `-q` flag** on `BaseCommand`: skips `StreamHandler`.
- **Exit codes:** `0` success, `1` error, `2` rejected. Errors to stderr.

## Phasing

| Phase | Work |
|-------|------|
| **1a** | `clear` (HTTP exists), `end-session` (needs new HTTP endpoint) |
| **1b** | `info` (needs per-request endpoint), `pin` (needs per-instance endpoint) |
| **1c** | `compress` (needs per-instance endpoint), `generate` (client-side only) |
