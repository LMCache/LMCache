# Trace Recording (`lmcache trace`)

Trace recording captures LMCache's operational stream during a real run so
the same workload can be **replayed** later for testing, regression hunting,
and benchmarking — without needing vLLM or, eventually, a GPU. This
document covers the **capture** half (PR1).  The replay tooling
(`lmcache trace info|replay`, `lmcache bench trace-replay`) is built on
top of this format and lands in PR2.

For configuration reference see [README.md](README.md). For event metadata
contracts see [EVENTS.md](EVENTS.md).

---

## 1. Goals and non-goals

### Goals

- Capture every public `StorageManager` API call to a single binary file.
- **Off** by default; near-zero overhead when off (one boolean check per
  decorated call).
- Recording is opt-in via a single CLI flag on `lmcache server`.
- The on-disk format is forwards-extensible: future trace **levels**
  (`mq`, `gpu`) can land without breaking the file layout or replay CLI.
- The decorator (`@enable_tracing`) is reusable on any future public API
  layer; instrumenting MQ handlers later requires no new event types or
  format changes.

### Non-goals (deferred to follow-up PRs)

- **No KV tensor data is captured.** Replay exercises bookkeeping and
  controller logic; payloads at replay time are zeros.
- **No MQ-, MPCacheEngine-, or GPU-copy-level capture.** Those layers
  carry GPU IPC handles and require a swappable GPU-copy abstraction
  that is out of scope.
- **No runtime enable/disable.** Capture is configured at server
  startup. Runtime toggling via the HTTP admin server can be layered on
  later by flipping the trace gate without touching the format.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  StorageManager.<method>      (+ context-manager publishes) │
│      │                                                      │
│      │  @enable_tracing()                                   │
│      ▼                                                      │
│  publish_call_event(qualname, raw_args)  ───── gated ─────► (no-op)
│      │                                          │           │
│      │  if _tracing_enabled                     │           │
│      ▼                                                      │
│  EventBus.publish(Event(TRACE_CALL, …))                     │
│      │                                                      │
│      │ (drain thread)                                       │
│      ▼                                                      │
│  StorageTraceRecorder._on_trace_call()                      │
│      │                                                      │
│      │  codecs.encode_args() + msgspec.msgpack              │
│      ▼                                                      │
│  trace file:  [Header][Record][Record]…                     │
└─────────────────────────────────────────────────────────────┘
```

Three pieces, each in its own module under
`lmcache/v1/mp_observability/trace/`:

| Module | Responsibility |
|--------|----------------|
| `decorator.py` | `@enable_tracing`, the trace gate, `publish_call_event` helper for context managers |
| `codecs.py` | Per-type encode/decode registry shared by recorder and (PR2) replay driver |
| `format.py` | `Header` + `Record` msgspec structs; length-prefixed framing |
| `recorder.py` | `TraceRecorder` ABC + `StorageTraceRecorder` `EventSubscriber` |
| `reader.py` | Streaming `TraceReader` (used by `trace info` / replay in PR2) |
| `lifecycle.py` | `maybe_initialize_trace_recorder` server-side wiring helper |

---

## 3. Capture: the `@enable_tracing` decorator

### Single unified event

All captured calls publish the same `EventType.TRACE_CALL` event:

```python
Event(
    event_type = EventType.TRACE_CALL,
    timestamp  = <time.time() stamped by EventBus.publish()>,
    metadata   = {
        "qualname": "lmcache.v1.distributed.storage_manager.StorageManager.reserve_write",
        "args":     {"keys": [...], "layout_desc": {...}, "mode": "new"},
        "t_mono":   <time.monotonic() captured inside publish_call_event>,
    },
)
```

One event type instead of one per method keeps the EventBus enum and
subscriber dispatch table small and lets new traced methods land
without schema bumps. The `qualname` field discriminates ops; the
`args` dict carries everything needed to reissue the call at replay
time.

`t_mono` is sampled inside `publish_call_event` (the publish-time
path), not on the EventBus drain thread. Otherwise the `t_mono` and
`t_wall` values recorded per call would drift by the drain queue
latency, rendering relative-timing analyses off by up to a frame's
worth of processing time.

### Signature-driven argument capture

The decorator binds `inspect.signature(func)` **once at decoration
time**:

```python
@enable_tracing()
def reserve_write(self, keys, layout_desc, mode): ...
```

On call, the wrapper:

1. Checks the gate (`_tracing_enabled`). If off, jumps straight to the
   real function — overhead is one bool load.
2. If on, runs `sig.bind_partial(*args, **kwargs).apply_defaults()`,
   filters to the configured `capture` / `redact` set (default:
   everything except `self` / `cls`), and publishes the event.

This keeps the per-call cost when enabled to a single signature bind +
dict-comprehension. No per-method instrumentation code; adding a new
traced method on either side reduces to slapping the decorator on.

### Entry-only

The decorator publishes on entry only; outputs and exceptions are not
captured. Replay re-runs the method and observes the live outcome
itself, so recorded outcomes would be redundant. Halving the event
volume also keeps the file smaller. Return values like `PrefetchHandle`
do not need to be correlated across records because the public
`StorageManager` API takes no handle as an input — later calls
reference keys, not handles.

### Context managers

`StorageManager.read_prefetched_results` is a `@contextmanager`
generator. The decorator cannot wrap it (it would publish the call to
the wrapper, not to `__enter__`). Instead the method calls
`publish_call_event(...)` manually at enter and exit, gated on
`is_tracing_enabled()`. The qualnames carry the
`.read_prefetched_results.__enter__` / `.__exit__` suffixes so replay
can re-enter the context manager faithfully.

### Why the decorator publishes raw values

The decorator publishes raw Python values; codec encoding happens later
on the EventBus drain thread inside the recorder. This keeps the
decorator import-cheap: it has no dependency on `codecs.py`, so adding
a new codec cannot pull the decorator's dependency graph.

All codec-targeted types (`ObjectKey`, `MemoryLayoutDesc`,
`PrefetchHandle`, `torch.Size`, `torch.dtype`) live in
`lmcache/v1/distributed/api.py`. Keeping them in a leaf module means
`codecs.py` imports them eagerly and registers every codec at import
time, with no cycle-break machinery.

---

## 4. The trace gate

A single module-level boolean in `decorator.py`:

```python
_tracing_enabled: bool = False
```

Flipped on inside `TraceRecorder.__init__` (after the file is open) and
off inside `TraceRecorder.close()`. A bool is sufficient: capture is
single-process; cross-thread visibility is not required for
correctness — at worst a few events are missed during the toggle
window. The bool sits at the head of every decorated call's hot path,
so the disabled cost is one attribute load.

---

## 5. Recorder

`StorageTraceRecorder(TraceRecorder)` subscribes to `TRACE_CALL` on the
EventBus. Subscriber callbacks already run on the EventBus drain
thread, so the recorder is off the request path by construction.
Encoding (codec + msgspec) and disk I/O happen inline in the callback;
adding a second worker thread would be premature optimization.

### Lifecycle

| Phase | Action |
|-------|--------|
| `__init__(output_path)` | Open file (unbuffered), capture `t_mono_start` / `t_wall_start`, flip the gate on. **Header write is deferred.** |
| `attach_storage_config(cfg)` | First call: serialize the StorageManagerConfig, hash it, write the header. Idempotent; subsequent calls are silently ignored. |
| `_on_trace_call(event)` | If the header has not been written, write a placeholder (empty config) header, then append the encoded record. |
| `close()` / `shutdown()` | Idempotent. Writes the placeholder header if neither attach nor any record ran. Flushes, fsyncs, closes the fd; flips the gate off. |

The "deferred header" design exists because the header carries the
serialized `StorageManagerConfig`, which is generally **longer** than
any placeholder. Writing a placeholder up front and seeking back to
overwrite it would land the new (longer) header bytes on top of any
records that landed in the meantime, corrupting the file. Deferring
the write avoids the in-place rewrite entirely and guarantees the file
is always readable regardless of whether `attach_storage_config` is
ever called.

### Failure modes

- **Codec error** (unknown arg type): the record is dropped, a
  WARNING is logged, `dropped_count` is incremented. The recorder
  continues. Losing a record is preferable to taking down the EventBus
  drain thread.
- **OSError on write**: same — drop and count.
- **fsync failure on close**: logged with `exc_info`; the close path
  still completes.

`dropped_count` is exposed as a property for tests and (future)
metrics integration.

### Shutdown contract

The recorder relies on `EventBus.stop()` to flush and close the file.
The chain is:

```
<server shutdown>
  → event_bus.stop()
      → _drain_all()                       (process queued events)
      → subscriber.shutdown() per sub      (EventBus contract)
          → TraceRecorder.close()          (flush + fsync + close fd)
```

All three cache-server entry points already invoke `event_bus.stop()`
in their shutdown paths:

- `server.py :: run_cache_server` — in the `KeyboardInterrupt`
  handler.
- `blend_server_v2.py :: run_cache_server` — same.
- `http_server.py :: lifespan` — in the FastAPI lifespan teardown
  branch.

`close()` is idempotent; calling it directly (for tests) and then
letting `shutdown()` fire is safe. The trace gate is flipped off
inside `close()`, so any events that race the shutdown after the
final drain become cheap no-ops in the publisher.

---

## 6. On-disk format

```
[ 4-byte big-endian length ][ msgpack Header  ]
[ 4-byte big-endian length ][ msgpack Record  ]
[ 4-byte big-endian length ][ msgpack Record  ]
...
```

Length-prefixed frames keep the reader simple and let truncated tails
(SIGKILL, fs buffer loss) be detected and recovered cleanly.

### `Header`

| Field | Type | Purpose |
|-------|------|---------|
| `magic` | `bytes` (`LMCT`) | Sanity check; reader rejects non-matching files |
| `format_version` | `int` (1) | Bumped on incompatible **framing** layout changes (length prefix, struct shape). Reader rejects unknown versions |
| `level` | `str` (`storage`) | Trace level discriminator. Future `mq` / `gpu` levels will share this format and use this field for replay-driver dispatch |
| `trace_schema_version` | `int` (1) | Bumped on incompatible changes to the captured API surface (e.g. a traced method's args change, a codec wire form changes). Owned by the trace subsystem, not tied to `lmcache.__version__`; reader rejects mismatches |
| `t_mono_start` | `float` | `time.monotonic()` at recorder construction; record `t_mono` is relative to this |
| `t_wall_start` | `float` | `time.time()` at construction, for absolute correlation with external logs |
| `sm_config_json` | `str` | JSON dump of `StorageManagerConfig` at record time, or empty string if attach was skipped |
| `sm_config_digest` | `str` | SHA-256 of `sm_config_json`. Replay drivers use this to detect mismatched configurations |

### `Record`

A single homogeneous shape across all ops; `qualname` discriminates.

| Field | Type | Purpose |
|-------|------|---------|
| `t_mono` | `float` | Seconds since `Header.t_mono_start` |
| `t_wall` | `float` | Wall-clock `time.time()` at the moment `EventBus.publish()` ran |
| `qualname` | `str` | Fully-qualified call-site name |
| `args` | `dict[str, Any]` | Codec-encoded argument dict |

The single-shape design means new traced ops are purely additive on
both write and read: no per-op msgspec class, no `Union` dispatch.

---

## 7. Codec registry

The `args` dict needs to round-trip values that msgpack does not
natively understand: `ObjectKey`, `MemoryLayoutDesc`, `PrefetchHandle`,
`torch.Size`, `torch.dtype`. A small per-type registry handles this:

```python
register_codec(t, TypeCodec(tag, encode, decode))
```

Encode wraps non-native values in `{"__t__": tag, "v": payload}` so
the decoder recognizes them. The same registry is used by the recorder
(encode-only in PR1) and by the replay driver (decode-only in PR2),
ensuring the read and write halves cannot drift apart.

Tuples are tagged separately so they decode back as tuples instead of
lists. `torch.Size` is a tuple subclass, so codec lookup checks the
exact type **before** the generic `isinstance(v, tuple)` branch.

Unknown types fail loudly (`TypeError`) rather than silently dropping
fields — silent drops would let bugs masquerade as test successes at
replay time.

---

## 8. CLI surface

`lmcache server` gains two new flags in the existing `Observability`
arg group:

| Flag | Description |
|------|-------------|
| `--trace-level {storage}` | **Primary enable flag.** Currently only `storage` is supported. |
| `--trace-output FILE` | Output path. Optional; if omitted while `--trace-level` is set, a timestamped file under `$TMPDIR` is minted (`lmcache-trace-<pid>-<UTC>.lct`) and its path is logged at INFO. |

Both flags flow through `ObservabilityConfig` and are consumed by
`maybe_initialize_trace_recorder`, called from `run_cache_server` in
both `multiprocess/server.py` and `multiprocess/blend_server_v2.py`.
When `--trace-level` is unset, the helper returns `None` and no
recorder is registered — true zero overhead.

The complementary `lmcache trace info|replay` and `lmcache bench
trace-replay` subcommands are deferred to PR2; they read the format
defined here.

---

## 9. Extensibility seams

Future MQ / GPU trace levels reuse this design without breaking the
file format:

1. **`level` header field** — the same `Header` carries the
   discriminator; replay dispatches on it.
2. **Reusable decorator** — apply `@enable_tracing` to MQ handlers or
   `MPCacheEngine` methods. No new event type, no format change. The
   `qualname` string differentiates them.
3. **Polymorphic recorder** — `TraceRecorder` ABC accepts new
   subclasses with different `get_subscriptions()` mappings (or, more
   likely, the same `TRACE_CALL` mapping with a different `level`
   passed to the base).
4. **Codec registry** — new arg types slot in by calling
   `register_codec`. No format bump. Keep newly-traced argument types
   in `lmcache/v1/distributed/api.py` (or another leaf module) so
   `codecs.py` can import them without pulling in modules that import
   the trace decorator.

---

## 10. Test coverage

Three test files under `tests/v1/mp_observability/trace/`:

- `test_decorator.py` — gate on/off; zero-overhead semantics when off;
  arg capture; `capture` / `redact` filters; entry-only on exception.
- `test_codecs.py` — round-trip every registered type, including
  primitives, tuples, `torch.Size`, `torch.dtype`,
  `MemoryLayoutDesc`, `ObjectKey`, `PrefetchHandle`. Unknown-type and
  unknown-tag error paths.
- `test_recorder.py` — header round-trip with and without
  `attach_storage_config`; gate flip on init / off on close;
  publish-via-EventBus end-to-end (codec encode → file → reader → codec
  decode); truncated-tail tolerance; bad-magic rejection;
  `dropped_count` increments on unencodable args.
