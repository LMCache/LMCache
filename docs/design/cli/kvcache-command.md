# `lmcache kvcache` CLI Command Design

**Status:** Proposal  |  **Date:** 2026-03-19

## Why

Users need a way to manage KV cache state for **specific requests** from the
command line — pin a request's cache to prevent eviction, compress it, or clear
it. Part of **Phase 1** of the
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
├── clear          # Clear all cached KV data
├── pin            # Pin a request's KV cache to L1/CPU (may be rejected)
├── compress       # Compress a request's KV cache in-place
└── end-session    # Clean up session state for a finished request
```

| Sub-command | Target | Description |
|------------|--------|-------------|
| `info` | instance | Show per-request cache state: which chunks, where stored, pinned status |
| `clear` | instance | Clear all cached KV data |
| `pin` | instance or controller | Pin a request's KV cache to L1/CPU; may be rejected if memory pressure is too high |
| `compress` | instance or controller | Compress a request's KV cache to reduce memory footprint |
| `end-session` | instance | Remove per-request session state (token hashes, chunk tracking) |

```bash
$ lmcache kvcache -h
usage: lmcache kvcache [-h] {info,clear,pin,compress,end-session} ...

Manage KV cache state for specific requests.

subcommands:
  info          Show per-request cache state (chunks, locations, pinned status)
  clear         Clear all cached KV data
  pin           Pin a request's KV cache to L1/CPU (may be rejected)
  compress      Compress a request's KV cache in-place
  end-session   Clean up session state for a finished request
```

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

Clear **all** cached KV data on the target instance.

```bash
$ lmcache kvcache clear --url http://localhost:8000

========== KV Cache Clear ==================
Status:                                   OK
Chunks removed:                         1024
=============================================
```

### `pin`

Pin a request's KV cache chunks to L1 (CPU memory) to prevent eviction. The
server **may reject** the request if CPU memory pressure is too high.

Exit codes: `0` = pinned, `2` = rejected, `1` = error.

```bash
$ lmcache kvcache pin --url http://localhost:8000 --request-id req-abc-123

======== KV Cache Pin (req-abc-123) ========
Status:                                   OK
Chunks pinned:                            32
=============================================
$ echo $?
0

# Quiet mode for scripts
if lmcache kvcache pin -q --url http://localhost:8000 --request-id req-abc-123; then
    echo "pinned"
else
    echo "rejected or error"
fi

# Narrowed to a token range
$ lmcache kvcache pin --url http://localhost:8000 \
    --request-id req-abc-123 --start 0 --end 512

# Rejected case (exit code 2)
$ lmcache kvcache pin --url http://localhost:8000 --request-id req-xyz

======== KV Cache Pin (req-xyz) =============
Status:                             REJECTED
Reason:              L1 memory pressure (91%)
=============================================
$ echo $?
2
```

### `compress`

Compress a request's KV cache chunks in-place to reduce memory footprint.

```bash
$ lmcache kvcache compress --url http://localhost:8000 \
    --request-id req-abc-123 --method zstd

===== KV Cache Compress (req-abc-123) ======
Status:                                   OK
Method:                                 zstd
Chunks compressed:                        32
=============================================

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

=== KV Cache End Session (req-abc-123) =====
Status:                                   OK
=============================================
```

**Note:** Today `end-session` is ZMQ-only (`END_SESSION` in `RequestType`). A new
HTTP endpoint needs to be added to the per-instance server.

---

## Existing API Surface & Gaps

### Usable today (no new endpoints needed)

| CLI sub-command | Existing endpoint | Notes |
|----------------|------------------|-------|
| `clear` | `DELETE /cache/clear` (instance), `POST /clear` (controller) | Existing endpoints work as-is |

### Needs new HTTP endpoints

| CLI sub-command | What exists today | New endpoint needed |
|----------------|------------------|---------------------|
| `info` | `GET /cache/kvcache/info` returns layer metadata only | Per-request chunk detail: given a request ID, return chunk ranges, locations, and pinned status |
| `pin` | `POST /pin` on controller only | Per-instance `POST /cache/pin` that accepts request-id, returns OK or REJECTED with reason |
| `compress` | `POST /compress` on controller only | Per-instance `POST /cache/compress` that accepts request-id + method |
| `end-session` | ZMQ `END_SESSION` only (no HTTP) | Per-instance `POST /cache/end-session` that accepts request-id |

---

## Implementation

- **Single `KVCacheCommand`** (`BaseCommand` subclass) with second-level
  argparse subparsers. File: `lmcache/cli/commands/kvcache.py`.
- **HTTP only:** `_http_request()` wraps `urllib.request` (no new deps).
- **Indexing args** shared via a helper that adds `--request-id`,
  `--start`, `--end` to each subparser.
- **Output:** `self.create_metrics()` — use `--format json | jq` for scripting.
- **New `--quiet` / `-q` flag** on `BaseCommand`: skips `StreamHandler`.
- **Exit codes:** `0` success, `1` error, `2` rejected. Errors to stderr.

## Phasing

| Phase | Work |
|-------|------|
| **1a** | `clear` (HTTP exists), `end-session` (needs new HTTP endpoint) |
| **1b** | `info` (needs per-request endpoint), `pin` (needs per-instance endpoint) |
| **1c** | `compress` (needs per-instance endpoint) |
