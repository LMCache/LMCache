# `kv_load_policy.py`: Per-Request Load vs. Recompute

The MP server can decide, per request, whether to serve a lookup or report a
miss so the engine recomputes the prompt. Loading is not always cheaper: a
hit that lives only in a slow L2 (disk, S3, remote) can take longer to fetch
than a prefill, and a busy storage tier is better left to requests that gain
more from it. This module is the plug point for that decision; the default
keeps today's behaviour (serve every lookup).

## Why the server, before the prefetch

A LOOKUP is a prefetch: `LookupModule.lookup` submits a task that pulls the
hit chunks from L2 into L1 and read-locks them, and the connector only sees
the hit count after that finishes. A decision made later (e.g. in the engine
connector) can only skip the L1→GPU copy, the cheap part. Deciding in
`lookup()` before `submit_prefetch_task` skips the L2 fetch, the L1 space and
the locks.

```
engine connector                           server: LookupModule.lookup
  └─ LOOKUP(token_ids, request_configs ─────►  hash chunks, begin session
            + engine_computed_tokens hint)     policy.should_load(ctx)?
                                                 yes → submit_prefetch_task  (unchanged)
                                                 no  → register empty job,
                                                       early_exit_reason=
                                                       "load_policy_recompute"
  ◄── QUERY_PREFETCH_STATUS → hit chunks ──────  0 when declined
  └─ 0 hits → engine prefills the prompt
```

The trade-off: before the prefetch the server does not know the hit length.
The policy sees the request, not the hit. Policies that need the hit length
("serve only if ≥ N chunks hit") need a second hook inside the prefetch
controller between its existence lookup and its L2 load phase; that is future
work.

## What a declined lookup does

- No prefetch is submitted, so nothing is fetched and nothing is locked.
  `query_prefetch_status` returns 0; the connector sees a plain miss and never
  calls `free_lookup_locks` for it.
- `MP_LOOKUP_PREFETCH_END` carries `early_exit_reason="load_policy_recompute"`
  and the real `requested_tokens`, so hit-rate metrics count the declined
  tokens as unserved (other early exits report 0 because nothing could be
  served).
- The engine treats the miss like any other and stores the prompt KV. Chunks
  already resident in L1 are rejected cheaply at `L1Manager.reserve_write`
  (`KEY_NOT_WRITABLE`); chunks only in L2 are written to L1 again.

## Interface

```python
class KVLoadPolicy(ABC):
    def __init__(self, configs: Mapping[str, Any]) -> None: ...
    @abstractmethod
    def should_load(self, ctx: KVLoadContext) -> bool: ...
```

`KVLoadContext`:

| Field | Meaning |
|---|---|
| `request_id`, `model_name`, `chunk_size` | identify the lookup |
| `num_lookup_tokens` | chunk-aligned tokens the lookup covers (upper bound on the hit) |
| `request_configs` | the request's `lmcache.*` entries as sent, engine hints included |

Policies are pure (no I/O) and run on the lookup handler thread. Each lookup
is decided once; the connector re-polls the result, not the policy. With
several MP servers each decides independently and the connector takes the
minimum hit, so one declining server makes the request a miss everywhere;
the existing tail-lock release covers the others.

## The engine hint

An engine integration may add `lmcache.hint.engine_computed_tokens`, the
tokens the engine already holds in its own prefix cache, to the
`request_configs` of a LOOKUP. `request_configs` is IPC metadata, not cache
identity (`IPCCacheServerKey.request_configs` has `compare=False`), so the
hint does not change which chunks match. The server does not interpret the
key; it passes `request_configs` through to the policy, which reads whatever
hints it needs.

| Integration | Sets the hint |
|---|---|
| `integration/vllm/lmcache_mp_connector.py` | scheduler-time lookups; not eager-prefetch lookups, which run before the engine checks its prefix cache |
| `integration/sglang/unified_lmcache_mp_connector.py` | every lookup (`local_hit_tokens`) |
| `integration/tensorrt_llm/tensorrt_mp_adapter.py` | every lookup (`num_computed_tokens`) |
| `integration/sglang/multi_process_adapter.py` | no (local hit not known at lookup) |

Clients can also send `lmcache.*` keys, so a policy must treat hints as
advisory.

## Configuration

| Where | Key | Value |
|---|---|---|
| `lmcache server` | `--kv-load-policy` | `DEFAULT` (built-in) or `package.module:ClassName` |
| `lmcache server` | `--runtime-plugin-config` | JSON passed to the policy constructor |
| request config (forwarded by the engine) | `lmcache.skip_load` | `true` → `DEFAULT` declines this request |

```bash
lmcache server --kv-load-policy my_pkg.policies:ShortPromptsRecompute \
  --runtime-plugin-config '{"my_pkg.min_tokens": 1024}'
```

A custom policy class must be importable in the server process. It receives
the full extra config and decides itself whether to honour
`lmcache.skip_load`.

## Non-scope

- The CacheBlend lookup (`CB_UNIFIED_LOOKUP`) does not consult the policy.
- The in-process (non-MP) engine connectors keep their own retrieve
  thresholds (e.g. `min_retrieve_tokens`).
- The reply stays a hit count. A `(hit, served)` reply would let the connector
  skip re-storing declined chunks and keep cache-hit stats exact; add it if the
  re-store cost shows up.
