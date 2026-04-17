# `lmcache bench engine` — Design & Extension Guide

**Status:** Implemented  |  **Date:** 2026-03-27

## Overview

`lmcache bench engine` runs sustained benchmarks against an OpenAI-compatible
inference engine. It supports multiple workload types that exercise different
caching patterns, each controlling its own request scheduling while shared
modules handle request sending, stats collection, and real-time progress
display.

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
    --workload random-prefill --tokens-per-gb-kvcache 6000 \
    --rp-num-requests 100
```

---

## 1. Architecture

### File Layout

```
lmcache/cli/commands/bench/
├── __init__.py                    # BenchCommand (CLI registration + orchestrator)
└── engine_bench/
    ├── __init__.py                # Package marker
    ├── config.py                  # EngineBenchConfig, auto-detection helpers
    ├── stats.py                   # RequestResult, StatsCollector, FinalStats
    ├── request_sender.py          # RequestSender (async streaming)
    ├── progress.py                # ProgressMonitor (real-time terminal display)
    └── workloads/
        ├── __init__.py            # create_workload() factory
        ├── base.py                # BaseWorkload (ABC with run loop)
        ├── long_doc_qa.py         # LongDocQAConfig + LongDocQAWorkload
        ├── multi_round_chat.py    # MultiRoundChatConfig + Session + MultiRoundChatWorkload
        └── random_prefill.py      # RandomPrefillConfig + RandomPrefillWorkload
```

### Module Dependency Graph

```
BenchCommand (orchestrator)
  ├── config.py          → EngineBenchConfig
  ├── stats.py           → StatsCollector
  ├── progress.py        → ProgressMonitor
  ├── request_sender.py  → RequestSender
  └── workloads/
       ├── __init__.py   → create_workload() factory
       └── base.py       → BaseWorkload (used by all concrete workloads)
```

All concrete workloads depend on `BaseWorkload`, `RequestSender`,
`StatsCollector`, and `ProgressMonitor` — but never on each other.

---

## 2. Core Modules

### 2.1 `config.py` — Configuration

```python
@dataclass
class EngineBenchConfig:
    engine_url: str
    model: str                    # auto-detected if not provided
    workload: str                 # "long-doc-qa", "multi-round-chat", "random-prefill"
    kv_cache_volume_gb: float
    tokens_per_gb_kvcache: int    # auto-resolved via --lmcache-url or explicit
    seed: int
    output_dir: str
    export_csv: bool
    export_json: bool
    quiet: bool
```

Key functions:

| Function | Purpose |
|----------|---------|
| `parse_args_to_config(args) -> EngineBenchConfig` | Converts CLI args to fully-resolved config |
| `auto_detect_model(engine_url) -> str` | Fetches model ID from `/v1/models` |
| `resolve_tokens_per_gb(lmcache_url, model_name) -> int` | Queries LMCache `/api/status` for `cache_size_per_token * world_size` |

`EngineBenchConfig` contains only general parameters. Workload-specific
configs live in their own modules and are resolved by the workload factory.

### 2.2 `stats.py` — Stats Collection

```python
@dataclass
class RequestResult:
    request_id: str
    successful: bool
    ttft: float                   # seconds (time to first token)
    request_latency: float        # seconds (total request time)
    num_input_tokens: int
    num_output_tokens: int
    decode_speed: float           # tokens/second
    submit_time: float            # absolute timestamp
    first_token_time: float       # absolute timestamp
    finish_time: float            # absolute timestamp
    error: str                    # empty if successful

@dataclass
class AggregatedStats:
    total_requests: int
    successful_requests: int
    failed_requests: int
    elapsed_time: float
    mean_ttft_ms: float
    mean_decode_speed: float
    mean_request_latency_ms: float
    input_throughput: float       # tokens/second
    output_throughput: float      # tokens/second
    total_input_tokens: int
    total_output_tokens: int

@dataclass
class FinalStats(AggregatedStats):
    p50_ttft_ms: float            # plus p90, p99
    p50_decode_speed: float       # plus p90, p99
    p50_request_latency_ms: float # plus p90, p99
```

`StatsCollector` is **thread-safe** (uses `threading.Lock`):

| Method | Called by | Description |
|--------|-----------|-------------|
| `on_request_finished(result)` | RequestSender callback | Records a completed request |
| `get_current_stats() -> AggregatedStats` | ProgressMonitor (every 1s) | Returns a snapshot for live display |
| `get_final_stats() -> FinalStats` | Orchestrator (after benchmark) | Computes percentiles |
| `reset()` | BaseWorkload (between warmup/benchmark) | Clears warmup stats |
| `export_csv(path)` | Orchestrator | Writes per-request CSV |
| `export_json(path, config)` | Orchestrator | Writes summary JSON |

### 2.3 `request_sender.py` — Async Streaming

```python
OnFinishedCallback = Callable[[RequestResult, str], None]

class RequestSender:
    def __init__(self, engine_url, model, completions_mode=False, on_finished=[])
    async def send_request(self, request_id, messages, max_tokens=128) -> RequestResult
    async def send_warmup_request(self, request_id, messages, max_tokens=1) -> RequestResult
    async def close(self) -> None
```

- Uses `AsyncOpenAI` for streaming chat/completions.
- Measures TTFT, decode speed, total latency per request.
- Extracts token counts from server usage reports.
- After each request (success or failure), invokes all `on_finished`
  callbacks with `(RequestResult, response_text)`.

Each `send_request` call is a **self-contained coroutine** — concurrency is
controlled externally by the workload (semaphore, QPS, or fire-all-at-once).

### 2.4 `progress.py` — Real-Time Display

```python
class ProgressMonitor:
    def __init__(self, stats_collector, quiet=False)
    def start(self) -> None          # starts daemon thread
    def stop(self) -> None           # stops thread, prints final state
    def on_request_sent(request_id)  # increments in-flight count
    def on_request_finished(request_id, successful)  # decrements in-flight count
    def log_message(message)         # adds to rolling log (last 5 lines)
```

Runs a daemon thread that redraws every second using ANSI cursor control.
Reads aggregated stats from `StatsCollector.get_current_stats()`. Tracks
in-flight count and rolling log messages internally. No-op when `quiet=True`.

### 2.5 `workloads/base.py` — Base Workload

```python
class BaseWorkload(ABC):
    def __init__(self, request_sender, stats_collector, progress_monitor)

    # --- Must implement ---
    @abstractmethod async def warmup(self) -> None
    @abstractmethod async def step(self, time_offset: float) -> float
    @abstractmethod def log_config(self) -> None
    @abstractmethod def on_request_finished(self, request_id: str, output: str) -> None

    # --- Provided by base class ---
    def run(self) -> None                         # entry point (blocks)
    def request_finished(self, result, text)      # thread-safe queue bridge
```

**`run()` loop** (in base class):

```
log_config()           ← print workload config (before progress monitor starts)
warmup()               ← workload-specific warmup
stats_collector.reset()
loop:
    drain_finished_queue() → on_request_finished()
    next_wakeup = step(time_offset)
    if next_wakeup < 0: break
    sleep until next_wakeup
drain_finished_queue()   ← final drain
```

**`step()` contract:**

- Returns the **absolute time offset** (from benchmark start) when the loop
  should call `step()` again. The loop sleeps until that time.
- Returns a **negative value** to signal the workload is complete.
- The loop calls `_drain_finished_queue()` before each `step()`, which
  calls `on_request_finished()` for any completed requests.

**Callback bridge — `request_finished()`:**

This method matches the `OnFinishedCallback` signature and is registered
on `RequestSender._on_finished` by the orchestrator. It enqueues
`(request_id, response_text)` onto a `queue.Queue`. The loop thread drains
this queue and calls `on_request_finished()` from a single thread, so
workload implementations do not need to handle cross-thread concerns.

### 2.6 `workloads/__init__.py` — Factory

```python
def create_workload(config, args, request_sender, stats_collector, progress_monitor) -> BaseWorkload
```

Dispatches on `config.workload` string to the appropriate workload module.
Resolves workload-specific config from `args`, constructs the workload
instance, and returns it ready to `run()`.

---

## 3. End-to-End Flow

The orchestrator in `BenchCommand._bench_engine()` wires everything together:

```
1. parse_args_to_config(args)     → EngineBenchConfig
2. StatsCollector()
3. ProgressMonitor(stats_collector, quiet)
4. RequestSender(engine_url, model)
5. create_workload(config, args, sender, collector, monitor) → workload
6. Wire callbacks on sender:
     - stats_collector.on_request_finished(result)
     - progress_monitor.on_request_finished(request_id, successful)
     - workload.request_finished(result, response_text)
7. workload.log_config()          → print config to terminal
8. progress_monitor.start()       → start live display
9. workload.run()                 → blocks until benchmark complete
10. progress_monitor.stop()
11. request_sender.close()
12. Emit final metrics (CLI metrics system)
13. Export CSV / JSON
14. sys.exit(1) if any failures
```

### Callback Wiring

```
Workload.step()
    │
    ├── send_request()  ──────────────────┐
    │                                     │
    └── progress_monitor.on_request_sent()│
                                          ▼
                                   RequestSender
                                   (streams SSE, collects stats)
                                          │
                              on_finished callbacks:
                              ├── stats_collector.on_request_finished(result)
                              ├── progress_monitor.on_request_finished(id, ok)
                              └── workload.request_finished(result, text)
                                          │
                                          ▼
                                   finished_queue
                                          │
                              loop drains → workload.on_request_finished(id, text)
```

---

## 4. Existing Workloads

### 4.1 `long-doc-qa` — Long Document Q&A

Tests KV cache reuse by asking repeated questions over long documents.

**Config** (`LongDocQAConfig`):

| Field | CLI arg | Default | Description |
|-------|---------|---------|-------------|
| `document_length` | `--ldqa-document-length` | 10000 | Tokens per document |
| `query_per_document` | `--ldqa-query-per-document` | 2 | Questions per document |
| `num_documents` | computed | — | `kv_cache_volume * tokens_per_gb / document_length` |
| `shuffle_policy` | `--ldqa-shuffle-policy` | `random` | `random` or `tile` |
| `num_inflight_requests` | `--ldqa-num-inflight-requests` | 3 | Max concurrent requests |

**Behavior:**

- **Warmup:** Sends each document once sequentially (`max_tokens=1`)
- **Dispatch:** Semaphore-controlled — `step()` acquires semaphore, fires
  async task, returns `0.0` (immediate re-call). Semaphore released when
  task completes.
- **`on_request_finished`:** No-op (stateless).
- **Termination:** Returns `-1.0` when schedule exhausted and all tasks done.

### 4.2 `multi-round-chat` — Multi-Round Chat

Simulates concurrent chat users with growing conversation history.

**Config** (`MultiRoundChatConfig`):

| Field | CLI arg | Default | Description |
|-------|---------|---------|-------------|
| `shared_prompt_length` | `--mrc-shared-prompt-length` | 2000 | System prompt tokens |
| `chat_history_length` | `--mrc-chat-history-length` | 10000 | Pre-filled history tokens |
| `user_input_length` | `--mrc-user-input-length` | 50 | Tokens per query |
| `output_length` | `--mrc-output-length` | 200 | Max tokens per response |
| `qps` | `--mrc-qps` | 1.0 | Queries per second |
| `duration` | `--mrc-duration` | 60.0 | Benchmark duration (seconds) |
| `num_concurrent_users` | computed | — | `kv_cache_volume * tokens_per_gb / (prompt + history)` |

**Behavior:**

- **Warmup:** Sends one request per session sequentially (`max_tokens=1`)
- **Dispatch:** QPS-controlled — `step()` dispatches at `1/qps` intervals
  using round-robin session scheduling. Returns `global_index * interval`.
  If the target session is busy, returns `time_offset + 0.01` to retry
  after queue drain.
- **`on_request_finished`:** **Stateful** — records the response in the
  session's conversation history via `Session.record_answer()`, which
  marks the session as ready for its next request.
- **Termination:** Returns `-1.0` when `time_offset >= duration` and all
  pending tasks are complete.

**Session state:** Each `Session` holds a system prompt, pre-filled history,
and a growing list of `(query, answer)` exchanges. `build_messages()` constructs
the full OpenAI message list including all prior context.

### 4.3 `random-prefill` — Prefill Speed Testing

Tests raw prefill throughput by firing all requests simultaneously.

**Config** (`RandomPrefillConfig`):

| Field | CLI arg | Default | Description |
|-------|---------|---------|-------------|
| `request_length` | `--rp-request-length` | 10000 | Tokens per request |
| `num_requests` | `--rp-num-requests` | 50 | Number of requests |

**Behavior:**

- **Warmup:** None.
- **Dispatch:** Fire-all-at-once — first `step()` dispatches all requests
  as concurrent async tasks with `max_tokens=1`, returns `0.0`. Subsequent
  `step()` calls wait via `asyncio.wait(FIRST_COMPLETED)`.
- **`on_request_finished`:** No-op (stateless).
- **Termination:** Returns `-1.0` when all tasks are complete.

---

## 5. Adding a New Workload

### Step 1: Create the workload module

Create `workloads/my_workload.py` with:

```python
from dataclasses import dataclass
from lmcache.cli.commands.bench.engine_bench.workloads.base import BaseWorkload

@dataclass
class MyWorkloadConfig:
    """Workload-specific config fields with defaults."""
    my_param: int = 100

    def __post_init__(self) -> None:
        # Validate all fields
        if self.my_param <= 0:
            raise ValueError(f"my_param must be positive, got {self.my_param}")

    @classmethod
    def resolve(cls, kv_cache_volume_gb, tokens_per_gb_kvcache, **kwargs):
        """Compute derived fields from the KV cache budget + CLI args."""
        # Example: compute a count from the cache budget
        computed_count = max(1, int(kv_cache_volume_gb * tokens_per_gb_kvcache / kwargs["my_param"]))
        return cls(my_param=kwargs["my_param"], ...)


class MyWorkload(BaseWorkload):
    def __init__(self, config, request_sender, stats_collector, progress_monitor, seed=42):
        super().__init__(request_sender, stats_collector, progress_monitor)
        self._config = config
        # ... generate data, build schedule, etc.

    def log_config(self) -> None:
        """Print workload config. Called BEFORE progress monitor starts."""
        print(f"Workload: my-workload\n  my_param: {self._config.my_param}")

    async def warmup(self) -> None:
        """Run warmup requests (or no-op)."""

    async def step(self, time_offset: float) -> float:
        """Dispatch logic. Return next wakeup time, or negative when done."""

    def on_request_finished(self, request_id: str, output: str) -> None:
        """Handle completed request. No-op for stateless, or record state."""
```

### Step 2: Register in the factory

In `workloads/__init__.py`, add the import and dispatch:

```python
from lmcache.cli.commands.bench.engine_bench.workloads.my_workload import (
    MyWorkloadConfig, MyWorkload,
)

_WORKLOAD_NAMES = (..., "my-workload")

def create_workload(...):
    ...
    if config.workload == "my-workload":
        wl_config = MyWorkloadConfig.resolve(
            kv_cache_volume_gb=config.kv_cache_volume_gb,
            tokens_per_gb_kvcache=config.tokens_per_gb_kvcache,
            my_param=args.mw_my_param,
        )
        return MyWorkload(wl_config, request_sender, stats_collector, progress_monitor, seed=config.seed)
    ...
```

### Step 3: Add CLI args

In `bench/__init__.py`, inside `_register_engine()`:

1. Add `"my-workload"` to the `--workload` choices list.
2. Add a new argument group with **prefixed** arg names:

```python
mw_group = parser.add_argument_group("my-workload workload options")
mw_group.add_argument("--mw-my-param", type=int, default=100, help="...")
```

All workload-specific args must be prefixed with a short workload identifier
(e.g., `ldqa-`, `mrc-`, `rp-`, `mw-`) to avoid name collisions.

### Step 4: Add tests

Create `tests/cli/commands/bench/engine_bench/workloads/test_my_workload.py`
with tests for:

- Config validation (`__post_init__` raises on invalid values)
- Config resolution (`resolve()` computes derived fields correctly)
- Workload data generation
- `warmup()` behavior (async)
- `step()` dispatch logic and return values (async)
- `on_request_finished()` behavior
- `run()` end-to-end with mocked `RequestSender`

Add factory tests to `test_create_workload.py`.

### Key Design Constraints

- **`step()` must not block indefinitely.** It should dispatch or wait
  briefly and return. The loop handles sleeping between calls.
- **`on_request_finished()` runs on the loop thread** (via queue drain),
  not the async sender thread. No locking needed within the workload.
- **`log_config()` prints via `print()`**, not `log_message()`, because
  it runs before the progress monitor starts. Use ANSI colors for
  readability.
- **Use `progress_monitor.log_message()`** for all runtime logging during
  the benchmark to avoid corrupting the terminal display.
- **Warmup stats are discarded** — `stats_collector.reset()` is called
  after warmup, so warmup request metrics don't affect final results.

---

## 6. Tests

```bash
# All bench tests (~190 tests)
pytest -xvs tests/cli/commands/bench/

# Specific workload
pytest -xvs tests/cli/commands/bench/engine_bench/workloads/test_long_doc_qa.py
pytest -xvs tests/cli/commands/bench/engine_bench/workloads/test_multi_round_chat.py
pytest -xvs tests/cli/commands/bench/engine_bench/workloads/test_random_prefill.py

# Factory
pytest -xvs tests/cli/commands/bench/engine_bench/workloads/test_create_workload.py

# CLI registration + orchestrator
pytest -xvs tests/cli/commands/bench/test_bench_command.py
```
