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
├── coordinator                     # Launch the mp coordinator (HTTP)
├── describe {kvcache,engine}       # Rich status view of a running endpoint
├── ping     {kvcache,engine}       # Pure liveness check (OK/FAIL)
├── query    {kvcache,engine}       # Single-shot query with metrics
├── bench    {engine,server,l2}    # Sustained performance benchmarking
└── kvcache  {clear,end-session}    # KV cache management actions
```

| Verb | Question it answers | Weight |
|------|-------------------|--------|
| `ping` | Is it alive? | Single-shot, instant (OK/FAIL) |
| `query` | What happens when I send one request? | Single-shot, with metrics |
| `describe` | What is this thing? | Rich status dashboard |
| `bench` | How fast is it? | Multi-iteration, metrics-heavy |
| `kvcache` | Mutate cache state | Clear, end-session, evict (future) |

All client commands use a `--url` flag pointing to the **LMCache HTTP server**
(e.g. `--url http://localhost:8000`).

---

## Commands in Detail

### `lmcache server`

Replaces `python3 -m lmcache.v1.multiprocess.http_server`. Runs in foreground,
Ctrl-C to stop. HTTP frontend is enabled by default; use `--no-http` to run
ZMQ-only.

```bash
lmcache server \
    --engine-type blend --host 0.0.0.0 --port 5555 \
    --max-gpu-workers 2 \
    --l1-size-gb 60 --eviction-policy LRU \
    --no-http  # opt out of HTTP frontend
```

Server args are composed from existing helpers: `add_mp_server_args()`,
`add_storage_manager_args()`, `add_prometheus_args()`, `add_telemetry_args()`,
`add_http_frontend_args()`.

### `lmcache coordinator`

Replaces `python3 -m lmcache.v1.mp_coordinator`. Runs the mp coordinator's
FastAPI/HTTP app in the foreground (Ctrl-C to stop). The coordinator tracks mp
server instances in a registry and evicts those whose heartbeats lapse.

```bash
lmcache coordinator \
    --host 0.0.0.0 --port 9300 \
    --instance-timeout 30 \
    --health-check-interval 10
```

Config resolves from `MPCoordinatorConfig.from_env()` (the
`LMCACHE_MP_COORDINATOR_*` environment variables); any CLI flag that is supplied
overrides the corresponding field. Each flag defaults to unset so env-only
deployments keep working. See
[../v1/mp_coordinator/README.md](../v1/mp_coordinator/README.md).

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
info, `GET_CHUNK_SIZE` for chunk size) and `/status` (HTTP) to build a
consolidated view.

### `lmcache ping`

Pure liveness check for both targets. Returns OK/FAIL with round-trip time,
measuring only the network round-trip excluding local Python overhead.

**`ping kvcache`** -- pings the LMCache server process via HTTP `/healthcheck`:
```bash
$ lmcache ping kvcache --url http://localhost:8080

======= Ping KV Cache =======
Status:                  OK
Round trip time (ms):    0.42
==============================

```

**`ping engine`** -- pings the vLLM server process via HTTP `/health`:
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

**`bench server`** -- end-to-end sanity test for a running LMCache MP cache
server (ZMQ + HTTP). For each sequence in ``[--start, --end)`` the tool runs a
cold pass (``LOOKUP`` miss → ``STORE``) and a warm pass (``LOOKUP`` hit →
``RETRIEVE``), then cross-checks per-chunk checksums against the server's HTTP
API. Exercises the full RPC path
(``REGISTER_KV_CACHE → GET_CHUNK_SIZE → LOOKUP → QUERY_PREFETCH_STATUS →
RETRIEVE → STORE → END_SESSION``).

Supports two run modes via ``--mode``:

- **``gpu``** (default) -- allocates real CUDA tensors and uses CUDA IPC
  (handle transfer path).
- **``cpu``** -- allocates POSIX-SHM-backed tensors; the server maps the same
  physical pages for zero-copy STORE/RETRIEVE (data transfer path).

The transfer path can be overridden explicitly with ``--transfer-mode
{auto,handle,data}``. ``auto`` keeps the historical mapping: gpu→handle,
cpu→data. Note: ``--transfer-mode handle`` on CPU is not yet implemented.

```bash
$ lmcache bench server \
    --rpc-url tcp://localhost:5555 \
    --url http://localhost:8080 \
    --start 0 --end 2

Connecting to LMCache MP Server at tcp://localhost:5555 (mode=gpu, transfer=auto) ...
Server chunk_size = 256
Resolved KV shape spec: (2,1024,16,8,128):float16:32
[seq=0] LOOKUP cold:  0/2 chunks hit (1.82 ms)
[seq=0] STORE:        2 chunks stored (1.74 ms)
[seq=0] LOOKUP warm:  2/2 chunks hit (1.31 ms)
[seq=0] RETRIEVE:     2 chunks retrieved (1.48 ms)
[seq=0] CHECKSUM MATCH OK
[seq=1] ...
```

With ``--end`` unset, the loop runs forever; stop with ``Ctrl-C``. The KV
tensor layout is controlled by ``--kvcache-shape-spec`` (see
``lmcache/v1/kv_layer_groups.py``); see :doc:`bench_server` in the user guide
for the full flag list.

**`bench l2`** -- store / lookup / load throughput benchmark against an
``L2AdapterInterface`` implementation (no MP server required). Implemented at
``lmcache/cli/commands/bench/l2_adapter_bench/``; see the
``docs/source/cli/bench_l2.rst`` user guide for full options.

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

- **Explicit registration:** Each command inherits from `BaseCommand` (in
  `commands/base.py`) and is registered in `commands/__init__.py`'s
  `ALL_COMMANDS` list. See [framework-and-metrics.md](framework-and-metrics.md).
- **`send_request()` helper:** Creates a temporary `MessageQueueClient`, submits
  a ZMQ request, waits with timeout (default 5s), tears down. All ZMQ commands
  use this. Extended to handle HTTP targets alongside ZMQ.
- **Framework:** `argparse` with subparsers (no new deps). Reuses existing
  `add_*_args()` helpers.
- **`--url` flag:** Configured per-subcommand (ZMQ vs HTTP semantics vary).

### File layout

```
lmcache/cli/
├── __init__.py
├── main.py              # main() entry point
├── metrics/             # Metrics system (see framework-and-metrics.md)
├── commands/
│   ├── __init__.py      # ALL_COMMANDS registry
│   ├── base.py          # BaseCommand ABC
│   ├── mock.py          # lmcache mock  (example/test command)
│   ├── server.py        # lmcache server
│   ├── coordinator.py   # lmcache coordinator
│   ├── describe.py      # lmcache describe {kvcache,engine}
│   ├── ping.py          # lmcache ping {kvcache,engine}
│   ├── query.py         # lmcache query {kvcache,engine}
│   ├── bench/           # lmcache bench {engine,server,l2}
│   │   ├── __init__.py          # BenchCommand + dispatch
│   │   ├── engine_bench/        # lmcache bench engine
│   │   ├── server_bench/        # lmcache bench server
│   │   └── l2_adapter_bench/    # lmcache bench l2
│   └── kvcache.py       # lmcache kvcache {clear,end-session}
├── config.py            # CLIConfig (centralized config system)
└── corpora/             # Built-in prompt corpora
```

### Other notes

- **Entry point:** `lmcache = "lmcache.cli.main:main"` in `pyproject.toml`.
- **`bench engine`:** Wraps `vllm.benchmarks`, then queries `/status` for
  cache metrics.
- **`query kvcache`:** Tokenizes `--prompt` using the model's tokenizer, then
  performs a lookup over ZMQ to check which chunks are cached.

## Phasing

| Phase | Scope |
|-------|-------|
| **0** | CLI framework (explicit registration, `Metrics`), `mock` example command, entry point — see [framework-and-metrics.md](framework-and-metrics.md) |
| **1** | **`server`** (done), `ping kvcache`, `kvcache clear`, `kvcache end-session`, `describe kvcache` |
| **2** | `ping engine`, `query engine`, `query kvcache`, `bench engine`, `bench server`, `bench l2`, `describe engine`, corpora |
| **3** | `kvcache evict` (future) |

Existing `lmcache_server` entry point kept as a deprecated alias for 2 minor releases.
