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
CLI operation goes through the **MP HTTP server**
(`lmcache/v1/multiprocess/http_server.py`).

Today some operations (e.g. `end-session`) only have ZMQ implementations. These
need new HTTP endpoints on the MP HTTP server before the CLI can use them.

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
| `pin` | instance | Pin a request's KV cache to L1/CPU; may be rejected if memory pressure is too high |
| `compress` | instance | Compress a request's KV cache to reduce memory footprint |
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

> **Status: needs further design — will not be implemented yet.**
> The output format, filtering options, and server-side endpoint are TBD.
> The sketch below is a placeholder to illustrate intent.

Show the cache state for a specific request: which chunks exist, which storage
backend holds each one, and whether they are pinned.

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

All CLI operations target the **MP HTTP server**
(`lmcache/v1/multiprocess/http_server.py`).

### Usable today (no new endpoints needed)

| CLI sub-command | Existing MP HTTP endpoint | Notes |
|----------------|--------------------------|-------|
| `clear` | `POST /api/clear-cache` | Clears all L1 cache. Works as-is. |

### Needs new MP HTTP endpoints

| CLI sub-command | What exists today | New endpoint needed on MP HTTP server |
|----------------|------------------|---------------------------------------|
| `info` | No per-request HTTP endpoint | `GET /api/kvcache-info?request_id=...` returning chunk ranges, locations, pinned status |
| `pin` | ZMQ only (no HTTP) | `POST /api/pin` accepting request-id, returning OK or REJECTED with reason |
| `compress` | ZMQ only (no HTTP) | `POST /api/compress` accepting request-id + method |
| `end-session` | ZMQ `END_SESSION` only (no HTTP) | `POST /api/end-session` accepting request-id |

---

## Implementation

- **Single `KVCacheCommand`** (`BaseCommand` subclass) with second-level
  argparse subparsers. File: `lmcache/cli/commands/kvcache.py`.
- **MP HTTP only:** `_http_request()` wraps `urllib.request` (no new deps).
  All requests target the MP HTTP server.
- **Indexing args** shared via a helper that adds `--request-id`,
  `--start`, `--end` to each subparser.
- **Output:** `self.create_metrics()` — use `--format json | jq` for scripting.
- **New `--quiet` / `-q` flag** on `BaseCommand`: skips `StreamHandler`.
- **Exit codes:** `0` success, `1` error, `2` rejected. Errors to stderr.

## Phasing

| Phase | Work |
|-------|------|
| **1a** | `clear` (HTTP exists), `end-session` (needs new HTTP endpoint) |
| **1b** | `pin` (needs per-instance endpoint), `compress` (needs per-instance endpoint) |
| **future** | `info` (needs further design — deferred) |
