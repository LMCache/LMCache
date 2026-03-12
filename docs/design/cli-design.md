# LMCache CLI Design

**Status:** Proposal  |  **Date:** 2026-03-11

## Why

Today users must remember `python3 -m lmcache.v1.multiprocess.http_server ...` and
similar module paths. We need a single `lmcache` command as the front door to all
LMCache functionality.

## Command Overview

```
lmcache
├── server                          # Launch LMCache server (ZMQ + HTTP)
├── describe {kvcache,engine}       # Rich status view of a running endpoint
├── ping     {kvcache,engine}       # Pure liveness check (OK/FAIL)
├── query    {kvcache,engine}       # Single-shot query with metrics
├── bench    {kvcache,engine}       # Sustained performance benchmarking
└── kvcache  {clear,end-session}    # KV cache management actions
```

| Verb | Question it answers | Weight |
|------|-------------------|--------|
| `ping` | Is it alive? | Single-shot, instant (OK/FAIL) |
| `query` | What happens when I send one request? | Single-shot, with metrics |
| `describe` | What is this thing? | Rich status dashboard |
| `bench` | How fast is it? | Multi-iteration, metrics-heavy |
| `kvcache` | Mutate cache state | Clear, end-session, evict (future) |

All client commands use a unified `--url` flag:
- KV cache targets: `--url localhost:5555` (ZMQ)
- Engine targets: `--url http://localhost:8000` (HTTP)

---

## Commands in Detail

### `lmcache server`

Replaces `python3 -m lmcache.v1.multiprocess.http_server`. Runs in foreground,
Ctrl-C to stop. HTTP frontend is enabled by default; use `--no-http` to run
ZMQ-only.

```bash
lmcache server \
    --engine-type blend --host 0.0.0.0 --port 5555 \
    --l1-size-gb 60 --eviction-policy LRU \
    --no-http  # opt out of HTTP frontend
```

Server args are composed from existing helpers: `add_mp_server_args()`,
`add_storage_manager_args()`, `add_prometheus_args()`, `add_telemetry_args()`,
`add_http_frontend_args()`.

### `lmcache describe`

```bash
$ lmcache describe kvcache --url localhost:5555

============ LMCache KV Cache Service ============
Health:                                  OK
ZMQ endpoint:                            tcp://localhost:5555
HTTP endpoint:                           http://localhost:8000
Engine type:                             blend
Chunk size:                              256
L1 capacity (GB):                        60.0
L1 used (GB):                            42.3 (70.5%)
Eviction policy:                         LRU
Cached objects:                          1024
Uptime:                                  2h 14m 32s
==================================================

$ lmcache describe engine --url http://localhost:8000

================ Inference Engine ================
Model:                                   meta-llama/Llama-3.1-70B-Instruct
Max context (tokens):                    131072
Status:                                  healthy
Running requests:                        3
==================================================
```

`describe kvcache` gathers data from multiple ZMQ request types (`NOOP` for debug
info, `GET_CHUNK_SIZE` for chunk size) and `/api/status` (HTTP) to build a
consolidated view.

### `lmcache ping`

Pure liveness check for both targets. Returns OK/FAIL with round-trip time,
measuring only the network round-trip excluding local Python overhead.

**`ping kvcache`** -- single `NOOP` round-trip over ZMQ:
```bash
$ lmcache ping kvcache --url localhost:5555

======= Ping KV Cache =======
Status:                  OK
Round trip time (ms):    0.42
==============================

```

**`ping engine`** -- single `/api/healthcheck` round-trip over HTTP:
```bash
$ lmcache ping engine --url http://localhost:8000

======== Ping Engine =========
Status:                  OK
Round trip time (ms):    12.3
==============================
```

### `lmcache query`

Single-shot query with detailed metrics. Use this to test a specific request
and see what happened.

**`query engine`** -- single inference request with TTFT/TPOT. Supports `{corpus}`
templates for realistic long-context prompts:
```bash
$ lmcache query engine --url http://localhost:8000 \
    --prompt "{ffmpeg} What is the example usage of ffmpeg?" --max-tokens 128

========== Query Engine Result ==========
Prompt tokens:                           8192
  Corpus 'ffmpeg':                       8186
  Query:                                 6
Output tokens:                           128
-----------Latency Metrics---------------
TTFT (ms):                               892.3
TPOT (ms/token):                         11.8
Total latency (ms):                      2403.7
Throughput (tokens/s):                   53.2
=========================================
```

**`query kvcache`** -- query KV cache state for specific keys or tokens:
```bash
# Check if a specific token sequence is cached (lookup)
$ lmcache query kvcache --url localhost:5555 \
    --prompt "{ffmpeg} What is the example usage of ffmpeg?" \
    --model meta-llama/Llama-3.1-8B-Instruct

======== Query KV Cache Result ==========
Prompt tokens:                           8192
Cached chunks:                           30/32 (93.8%)
Cached tokens:                           7680/8192
Cache status:                            HIT (partial)
=========================================

# Store-retrieve round-trip with latency and correctness
$ lmcache query kvcache --url localhost:5555 --round-trip

==== Query KV Cache Result (round-trip) ====
Store latency (ms):                      1.23
Retrieve latency (ms):                   0.87
Checksum:                                OK
============================================
```

### `lmcache bench`

**`bench kvcache`** -- exercises store/retrieve/lookup over ZMQ. Includes a
correctness check: each retrieved KV cache chunk is checksummed against the original
stored data to verify integrity under load.

```bash
$ lmcache bench kvcache --url localhost:5555 --duration 30

========= Bench KV Cache Result (30s) =========
--------------Operations (ops/s)----------------
Store:                                   41.3
Retrieve:                                127.3
Lookup:                                  281.7
-----------------Hit Rate-----------------------
L1:                                      92.3%
L2:                                      67.8%
---------------Bandwidth (GB/s)-----------------
L1 read:                                 12.4
L1 write:                                8.7
L2 read:                                 2.1
L2 write:                                1.4
--------------Correctness-----------------------
Checksums:                               5060/5060 OK
================================================
```

Use `--verify-only` to run the correctness check without reporting throughput
(useful in CI), or `--no-verify` to skip checksums for pure throughput measurement.

**`bench engine`** -- **superset of `vllm bench serve`**. Same CLI args, same output
format, plus an extra LMCache KV cache metrics section:

```bash
# vllm bench serve compatible -- just swap the command name
$ lmcache bench engine \
    --url http://localhost:8000 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name random --random-input-len 7500 --random-output-len 200 \
    --num-prompts 30 --request-rate 1 --ignore-eos

============ Serving Benchmark Result ============
Successful requests:                     30
Benchmark duration (s):                  31.34
Total input tokens:                      224970
Total generated tokens:                  6000
Request throughput (req/s):              0.96
Output token throughput (tok/s):         191.44
Total Token throughput (tok/s):          7369.36
---------------Time to First Token----------------
Mean TTFT (ms):                          313.41
Median TTFT (ms):                        272.83
P99 TTFT (ms):                           837.32
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.84
Median TPOT (ms):                        8.72
P99 TPOT (ms):                           11.35
----------LMCache KV Cache Performance------------
KV cache hit rate (L1):                  92.3%
KV cache hit rate (L2):                  67.8%
L1 read bandwidth:                       12.4 GB/s
L1 write bandwidth:                      8.7 GB/s
Avg tokens saved by cache (per req):     6420
Cache-assisted TTFT savings (est.):      58.2%
==================================================
```

LMCache-specific additions on top of vLLM args: `--url` (replaces `--port`),
`--prompt` with `{corpus}` templates, `--corpus name=path` for custom corpora.

### `lmcache kvcache`

```bash
$ lmcache kvcache clear --url localhost:5555

========== KV Cache Clear ==========
Status:                              OK
Objects removed:                     1024
====================================

$ lmcache kvcache end-session --url localhost:5555 <request_id>

======== KV Cache End Session ========
Status:                              OK
Request ID:                          <request_id>
======================================
```

---

## Prompt Corpora

`query engine`, `bench engine`, and `query kvcache` support `{name}` in `--prompt`
to expand built-in text corpora (e.g., `{paul_graham}` ~12k tokens, `{ffmpeg}`
~8k tokens). Custom corpora: `--corpus my_doc=./file.txt`. Built-in corpora ship
in `lmcache/cli/corpora/`.

## Implementation Notes

### Architecture

- **Auto-discovery:** Commands discovered via `pkgutil.iter_modules()` on the
  `commands/` package. Drop a new file in `commands/`, define
  `register_command(subparsers)`, done.
- **`CommandRegistrar` protocol:** Each command module exposes
  `register_command(subparsers)` which adds a subparser and sets
  `parser.set_defaults(func=handler)`.
- **`send_request()` helper:** Creates a temporary `MessageQueueClient`, submits
  a ZMQ request, waits with timeout (default 5s), tears down. All ZMQ commands
  use this. Extended to handle HTTP targets alongside ZMQ.
- **Framework:** `argparse` with subparsers (no new deps). Reuses existing
  `add_*_args()` helpers.
- **`--url` flag:** Unified connection flag with auto-detection
  (`localhost:5555` → ZMQ, `http://localhost:8000` → HTTP).

### File layout

```
lmcache/cli/
├── __init__.py
├── __main__.py          # Auto-discovery + dispatch
├── base.py              # send_request(), add_url_arg(), CommandRegistrar
├── commands/
│   ├── server.py        # lmcache server
│   ├── describe.py      # lmcache describe {kvcache,engine}
│   ├── ping.py          # lmcache ping {kvcache,engine}
│   ├── query.py         # lmcache query {kvcache,engine}
│   ├── bench.py         # lmcache bench {kvcache,engine}
│   └── kvcache.py       # lmcache kvcache {clear,end-session}
└── corpora/             # Built-in prompt corpora
```

### Other notes

- **Entry point:** `lmcache = "lmcache.cli.main:main"` in `pyproject.toml`.
- **`bench engine`:** Wraps `vllm.benchmarks`, then queries `/api/status` for
  cache metrics.
- **`query kvcache`:** Tokenizes `--prompt` using the model's tokenizer, then
  performs a lookup over ZMQ to check which chunks are cached.

## Phasing

| Phase | Scope |
|-------|-------|
| **1** | `server`, `ping kvcache`, `kvcache clear`, `kvcache end-session`, `describe kvcache`, entry point |
| **2** | `ping engine`, `query engine`, `query kvcache`, `bench engine`, `bench kvcache`, `describe engine`, corpora |
| **3** | `kvcache evict` (future) |

Existing `lmcache_server` entry point kept as a deprecated alias for 2 minor releases.
