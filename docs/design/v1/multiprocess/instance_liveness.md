# Worker Liveness Tracking and Reaping

## Problem

The MP server has one-way health monitoring today: vLLM adapters PING the
server, but the server doesn't monitor adapters. When a vLLM worker dies
silently, `gpu_contexts[instance_id]` and its KV-cache CUDA IPC handles linger
until something triggers `unregister_kv_cache` — which never happens on a crash.

The other artifacts of a dead vLLM (in-flight `_prefetch_jobs` holding L1 read
locks, pending sessions) are bounded already: L1 read locks have a TTL
(`read_ttl_seconds`, default 300s in `lmcache/v1/distributed/config.py`) after
which `TTLLock.is_locked()` returns False and the chunk becomes evictable under
LRU. So the chunk-capacity issue self-resolves within the TTL window. This
design does **not** add a separate cleanup path for prefetch jobs — see "Out of
scope" below.

## Goals

- Server detects dead workers via missed PINGs and reaps `gpu_contexts`.
- Single steady-state timeout (no warmup/active modes, no Nth-heartbeat
  heuristic).
- No false eviction during long warmup (large MoE / CUDA graph capture).
- Loud failure (logs) over silent recovery. Metrics/events deferred to a
  follow-up PR.

## Out of scope

- **Scheduler liveness tracking and prefetch-job reap.** L1 lock TTL bounds the
  chunk-pinning window without it. The remaining concern is the `_prefetch_jobs`
  Python dict-entry leak (the `server.py:208` TODO), addressed separately by a
  TTL sweep over `submit_time`. That PR is independent and not blocked by this
  one.
- Sub-process partial-failure detection.
- Replacing the existing client-side `_health_event` gate.
- Auto-resurrection on a `PING`-returns-False. Operator restarts.
- Cross-host failover.

## Design

Two coordinated changes:

1. **Worker adapter eager-starts the heartbeat.** Move
   `_ensure_heartbeat_started()` out of the worker's lazy hot paths
   (`submit_store_request` / `submit_retrieve_request`) into
   `register_kv_caches`, after the REGISTER ack. The first worker PING fires
   within one heartbeat interval of registration, independent of warmup or
   first-traffic timing. The scheduler adapter's heartbeat stays lazy-started on
   first lookup — it carries an `instance_id=0` sentinel and is purely a
   server-health probe; there's no reap path driven by scheduler PINGs.
2. **Server tracks worker liveness; reaper evicts past a single timeout.**
   Worker PINGs carry their `instance_id`. The server records `last_seen`, and a
   periodic reaper thread calls `unregister_kv_cache(instance_id)` for entries
   past `reap_after_seconds`.

## Wire protocol

### `PING`

Old: `payload_classes=[]`, response `bool`. New: `payload_classes=[int]`
(`instance_id`), response `bool`.

| Sender            | `instance_id`                                              |
|-------------------|------------------------------------------------------------|
| Worker adapter    | random 63-bit positive int generated once at adapter `__init__` (e.g. `random.getrandbits(63)`); stable for the adapter's lifetime |
| Scheduler adapter | `0` (sentinel — scheduler PINGs are pure server-health probes; no tracking) |

**Why not `os.getpid()`.** The MP server can serve multiple vLLM deployments —
including containerized replicas where each replica has its own PID namespace.
Two replicas with worker PID 1 (or any low PID) would collide on the same
`instance_id` server-side, producing silently-wrong behavior: registrations
overwriting each other, liveness refreshed by the wrong worker, reaps targeting
the wrong context. A random 63-bit int makes collisions astronomically unlikely
(~4 billion adapters before a 50% birthday collision) without needing external
coordination. The same change applies to every existing request that carries
`instance_id` — `REGISTER_KV_CACHE`, `STORE`, `RETRIEVE`, `UNREGISTER_KV_CACHE`,
and now `PING`. The server's view of `instance_id` is fully opaque; nothing on
the server side parses it as a PID.

Server response semantics:
- `instance_id == 0`: return `True` (sentinel; no tracking).
- `instance_id` is a known live worker: refresh `last_seen`, return `True`.
- Unknown non-zero `instance_id`: return `False`. The worker was reaped or never
  registered. Adapter clears `_health_event`; subsequent submits short-circuit
  through existing degraded-mode logic. No auto-recovery — operator restarts the
  vLLM process.

Backward compat: arity change loud-fails on either side via msgspec
deserialization error. Release notes call out the wire bump.

## Server-side state

```python
@dataclass
class _InstanceLiveness:
    last_seen: float       # time.monotonic() everywhere
    registered_at: float

class MPCacheEngine:
    _liveness: dict[int, _InstanceLiveness]   # keyed by worker instance_id
                                              # (opaque random 63-bit int)
    _liveness_lock: threading.Lock
```

`time.monotonic()` everywhere — wall-clock jumps (NTP, suspend) must not fire
false reaps.

## Server-side handler changes

| Handler              | Change |
|----------------------|--------|
| `register_kv_cache`  | If `instance_id` already in `gpu_contexts` (improbable with random IDs but cheap insurance), run `unregister_kv_cache` first. Then install context and overwrite `_liveness[instance_id]` unconditionally. |
| `unregister_kv_cache`| Idempotent. Pop from `_liveness`. |
| `store`, `retrieve`  | Refresh `_liveness[instance_id].last_seen`. **Keep the existing `assert instance_id in gpu_contexts`** — see "Why we keep the assert" below. |
| `ping`               | New signature `def ping(self, instance_id: int) -> bool`. Returns `True` for `instance_id == 0` (no-op for sentinel). For non-zero: if in `_liveness`, refresh `last_seen` and return `True`; else log WARNING and return `False`. |
| Engine `close()`     | Stop reaper before closing storage manager / event bus. |

### Reaper

`PeriodicThread`, runs every `reaper_interval` seconds. Snapshot stale
candidates **under `_liveness_lock`** (avoids dict-changed-during-iter), then do
cleanup outside the lock with a CAS-style re-check:

```python
def _execute(self):
    now = time.monotonic()
    with self._liveness_lock:
        stale = [iid for iid, st in self._liveness.items()
                 if now - st.last_seen > self._reap_after]
    for iid in stale:
        self._reap_worker(iid, now)

def _reap_worker(self, iid, now):
    with self._liveness_lock:
        st = self._liveness.get(iid)
        if st is None or now - st.last_seen <= self._reap_after:
            return                               # raced with a refresh
        del self._liveness[iid]
    self.unregister_kv_cache(iid)                # idempotent
    logger.warning("Reaped worker %d (%.1fs since last contact)",
                   iid, now - st.last_seen)
```

The reaper runs on its own thread, **not** the BLOCKING handler pool. The
existing handlers it can race (`unregister_kv_cache`, `store`, `retrieve`) are
idempotent on success and assert-fail-fast on missing state.

#### Why we keep the assert in `store`/`retrieve`

The original draft of this design proposed converting `assert instance_id in
self.gpu_contexts` to `if … raise ValueError(...)` to satisfy the coding
standard. Final review caught that the MQ layer (`mq.py:419-482`) silently
swallows handler exceptions: it `logger.exception`s and never sends an error
response, so the adapter's `MessagingFuture` would hang forever (`futures.py`
exposes only `set_result`, no `set_exception`). Converting `assert` → `raise`
would replace one bug (stripped under `python -O`) with another (silently hung
adapter futures). The proper fix is to extend the MQ layer with error-response
plumbing — out of scope here.

We therefore **keep the assert** for now and document it as a known
standards-violation deferred to a follow-up. In practice the path is rare: the
reaper only fires after 30s+ of silence, so a "dead worker that's actually still
alive enough to issue store/retrieve" is a partition-recovery edge case where
assert-failing the worker is honest behavior.

Lock discipline: `_liveness_lock` is only held across short dict reads and the
`del`. No I/O, no other locks.

### Defaults and missed-ping tolerance

| Knob                                 | Default | Where                              |
|--------------------------------------|---------|------------------------------------|
| `LMCACHE_MP_REAP_AFTER_SECONDS`      | 30      | env via existing `MPServerConfig` argparse path; passed into `MPCacheEngine.__init__` |
| `LMCACHE_MP_REAPER_INTERVAL_SECONDS` | 10      | same plumbing as above             |
| Adapter heartbeat interval           | 10      | existing `DEFAULT_HEARTBEAT_INTERVAL` |
| Adapter cold-start grace (failures)  | 2       | adapter constant                   |

**Single missed ping must not reap.** With the defaults, the implicit "missed
heartbeats before reap" is `reap_after / heartbeat = 3` — the worker has to fall
silent for three consecutive intervals before the reaper acts. A one-off ping
failure (transient GIL stall during an NCCL all-reduce, momentary zmq blip,
brief network hiccup) is absorbed: the next successful ping refreshes
`last_seen` well within the deadline.

Operator tuning rule: `reap_after >= 3 × heartbeat_interval`. We do **not**
enforce this in code (the server doesn't know the adapter's heartbeat interval),
but we document it and the defaults satisfy it. Setting `reap_after` close to
`heartbeat_interval` would re-introduce single-miss reaping; the design assumes
operators don't.

Detection window: best ≈ 30s (death just before tick), worst ≈ 40s (death just
after a tick — needs the next reaper tick to notice). The adapter clears its own
`_health_event` after `~3 × interval = 30s` of unreachability — symmetric
degradation under network partition.

## Adapter-side changes

`lmcache/integration/vllm/vllm_multi_process_adapter.py`:

### `HeartbeatThread`
- Constructor takes `instance_id: int`, plumbed into `send_ping`.
- `_health_event` starts set in adapter `__init__` (existing behavior); the
  first tick may run before any successful ping, so the cold-start branch treats
  "previously-set health event" as the entry condition.
- Tracks `_consecutive_failures: int` (resets to 0 on any successful ping) and
  `_first_success_seen: bool` (latched True on the first healthy ping; never
  reset).
- Per tick, dispatched on `send_ping`'s `Optional[bool]` result:
  - `True` (healthy): reset failures; latch `_first_success_seen`; set
    `_health_event`.
  - `None` (transient — timeout / exception):
    - If `_health_event` already cleared: leave cleared, increment counter.
    - Cold-start (`not _first_success_seen`): increment counter; clear
      `_health_event` only when counter ≥ 2 (absorbs a single GIL stall during
      warmup).
    - Steady-state (`_first_success_seen`): clear `_health_event` immediately on
      any single transient failure.
  - `False` (terminal — server explicitly returned False, i.e. "I don't know
    your instance_id"): clear `_health_event` immediately **and stop the
    heartbeat thread**. Operator must restart vLLM. No auto-recovery.

### `send_ping`
New signature `send_ping(mq_client, timeout, instance_id) -> Optional[bool]`.
The three return values are distinct so the heartbeat thread can tell them
apart:

- `True` — server responded with `True` (known live worker, or sentinel `0`
  echo).
- `False` — server responded with `False` (unknown instance_id; terminal).
- `None` — `TimeoutError`, deserialization error, or other transient exception.
  Counted toward cold-start grace; does not stop the thread.

The conflation of "server said False" with "I caught an exception" was the
ambiguity the previous `bool`-only signature had; using `Optional[bool]` keeps
the wire response a plain bool while letting `send_ping` synthesize a third "no
answer" result for the caller.

### `LMCacheMPWorkerAdapter`
- `__init__`: replace `self.instance_id = os.getpid()` with `self.instance_id =
  random.getrandbits(63)`. Generated once, stable for the adapter's lifetime.
  Used for every request that carries `instance_id` (REGISTER_KV_CACHE, STORE,
  RETRIEVE, UNREGISTER_KV_CACHE, PING).
- After REGISTER ack in `register_kv_caches`, call
  `_ensure_heartbeat_started()`. On REGISTER failure, no heartbeat thread is
  created.
- Drop the lazy `_ensure_heartbeat_started()` calls in `submit_store_request` /
  `submit_retrieve_request`.
- `shutdown`: stop the heartbeat first, then existing UNREGISTER + close.
- `_ensure_heartbeat_started`: pass `instance_id=self.instance_id`.

### `LMCacheMPSchedulerAdapter`
- `_ensure_heartbeat_started`: pass `instance_id=0` (sentinel — server treats as
  untracked health probe). Keep the existing lazy-start behavior in
  `maybe_submit_lookup_request` — the scheduler heartbeat is a purely diagnostic
  server-health probe on the scheduler's side, not load-bearing for the reaper
  (which only tracks workers). No new shutdown method, no lifecycle changes. The
  connector lives in the vLLM repo and we deliberately do not touch it.

### Tests that patched `_ensure_heartbeat_started`
`tests/v1/multiprocess/test_free_locks.py` and similar — patch before
construction or target `HeartbeatThread.start` directly.

## PING-from-unknown: explicit failure, not silent recovery

When the server has no `_liveness[instance_id]` entry (e.g. the worker was
reaped after a network blip exceeded the deadline), it returns `False`. The
adapter's heartbeat thread treats this as terminal: clear `_health_event`, stop
pinging, log ERROR. We don't auto-resurrect because:
- The server can't reconstruct kv_caches/layout from a PING.
- Auto-recovery hides reliability problems.

`register_kv_cache` is the worker's auto-recovery path — but only at process
startup. The adapter doesn't call it again in response to PING returning False.

## Edge cases

1. **Worker re-registers after reap.** Each adapter generates a fresh random
   `instance_id` at `__init__`, so a restarted worker registers under a new ID —
   the prior `_liveness` entry is reaped naturally under the old ID. The
   defensive cleanup in `register_kv_cache` is only a safety net for the
   (vanishingly unlikely) duplicate-random case.
2. **NCCL stop-the-world during warmup spikes one PING latency.** Cold-start
   grace (2 consecutive failures) plus server-side `missed_heartbeats=3`
   (derived from defaults) absorb a single miss on either side.
3. **Reaper races a legitimate request.** Either the request refreshes
   `last_seen` first (no reap), or `gpu_contexts[iid]` is gone when the handler
   runs and the existing assert fires server-side. The MQ layer swallows the
   `AssertionError` and the adapter's future hangs (see edge case 8 + "Why we
   keep the assert"). The reaper window is rare enough in practice (≥30s of
   silence before reap) that this race is dominated by the partition-recovery
   edge case.
4. **CAS re-check** inside `_reap_worker` closes the snapshot-vs-cleanup race
   when a PING refreshes between snapshot and del.
5. **Wire skew.** Old PING shape against new server (or vice versa) raises a
   deserialization error in the protocol layer; loud failure on both sides.
   Release-notes entry.
6. **Server restart.** Adapters' next PING returns `False`; heartbeats stop;
   adapters degrade. They do not auto-recover. Operator restarts every connected
   vLLM process. Sized for sidecar deployments.
7. **Engine `close()` ordering.** `MPCacheEngine.close()` stops the reaper
   thread first, then proceeds with `storage_manager.close()` and the existing
   `gpu_contexts.clear()`. If the reaper fires between `del self._liveness[iid]`
   and `unregister_kv_cache(iid)` while another shutdown step concurrently
   clears `gpu_contexts`, `unregister_kv_cache`'s missing-id branch logs a
   benign warning and returns — no crash, just noise. Stopping the reaper first
   eliminates the noise.
8. **Pre-existing `assert` in store/retrieve.** Kept as-is; see "Why we keep the
   assert" — the MQ layer doesn't propagate handler exceptions today, and
   converting to `raise` without that fix would create silent hung futures.
   Tracked as a deferred follow-up alongside MQ exception propagation.
9. **Pre-existing in-flight CUDA race in `unregister_kv_cache`.**
   `torch.cuda.empty_cache()` can race with handler memcpy_async if a peer
   thread held a local reference. Out of scope; same race exists on clean
   unregister today.
10. **Reaper deletes liveness, then a brand-new `register_kv_cache` arrives with
    the same `instance_id` before reaper calls `unregister_kv_cache`.**
    Astronomical with 63-bit randomness (≪ 1 in 2^63 per run). If it happens,
    the reaper would clobber the new worker's freshly-installed `gpu_contexts`
    entry. Acknowledged but not closed in code — the cost (widening
    `_liveness_lock` to cover a GPU `empty_cache` call) is not worth the
    probability. Document it here so future maintainers don't rediscover it.

## Test plan

### Unit (server, `tests/v1/multiprocess/`)
1. Reap fires after `reap_after` with no pings — assert `gpu_contexts[iid]` is
   gone and the WARNING log line was emitted.
2. PING refreshes deadline.
3. CAS race: PING arrives between snapshot and `_reap_worker` → no reap.
4. PING from unknown `instance_id` returns False, logs WARNING, no reap.
5. PING with sentinel `instance_id=0` returns True, no `_liveness` mutation.
6. Concurrent PING + reaper: no exceptions, no double cleanup.
7. `engine.close()` stops reaper before storage manager close.
8. Wire-skew rejection (old empty-payload PING against new server).

### Unit (adapter)
9. Worker `register_kv_caches` starts the heartbeat (not lazy on first
   store/retrieve).
10. Scheduler heartbeat keeps existing lazy start on first lookup; PINGs carry
    `instance_id=0` and the server returns True without tracking.
11. Cold-start grace: two consecutive failures flip; one success latches
    `_first_success_seen`; afterward a single failure flips immediately.
12. PING returning False (terminal): heartbeat stops; subsequent submits
    short-circuit through degraded mode.
13. Update tests that patch `_ensure_heartbeat_started` to patch before
    construction or target `HeartbeatThread.start`.
14. Worker `shutdown` stops the heartbeat thread before UNREGISTER + close.

### End-to-end smoke
15. Start MP server + vLLM with separate engine and worker processes. `kill -9`
    the worker. Within ~40s:
    - Server logs the worker reap with the killed worker's `instance_id`.
    - `report_status` no longer lists the dead worker.
16. Slow-warmup regression: model with warmup >60s — adapter `_health_event`
    stays set throughout warmup (validates eager-start + cold-start grace).

## Files touched

- `lmcache/integration/vllm/vllm_multi_process_adapter.py` — worker eager-start
  in `register_kv_caches`, cold-start grace, `send_ping(instance_id)` signature,
  worker `instance_id = random.getrandbits(63)` (replaces `os.getpid()`; fixes
  container / PID-namespace collisions), scheduler `instance_id=0` sentinel
  (lazy-start preserved), worker `shutdown` heartbeat-stop hook.
- (No vLLM connector changes. The connector lives in the vLLM repo and is
  intentionally untouched — daemon-thread heartbeat dies with the process, so no
  shutdown plumbing is required from this side.)
- `lmcache/v1/multiprocess/protocols/controller.py` — `PING` payload becomes
  `[int]`.
- `lmcache/v1/multiprocess/server.py` — `_liveness`, `_liveness_lock`, reaper
  class, `_reap_worker`, `ping` signature, `register_kv_cache` defensive
  cleanup, `store`/`retrieve` liveness refresh, engine close ordering (stop
  reaper first inside `MPCacheEngine.close()`).
- Tests as listed.
- Release notes: PING wire bump, server-restart fan-out behavior.

All new symbols follow `docs/coding_standards.md`: full type hints (no `Any`, no
bare `Optional`), full docstrings, no bool params, no `assert` for runtime
checks.

## Follow-up (separate PRs)

- **`_prefetch_jobs` TTL sweeper** — addresses `server.py:208` TODO. Periodic
  scan over `_prefetch_jobs.values()`, drop entries whose `submit_time` is older
  than e.g. `read_ttl_seconds * 1.5`. Catches the Python dict-entry leak
  independently of liveness tracking, and also handles "scheduler is alive but
  flaked on a single LOOKUP" — which liveness reap would not catch even if we
  tracked schedulers.
- **MQ exception propagation + `assert → raise ValueError`** in `store` /
  `retrieve`. The MQ layer at `mq.py:419-482` currently swallows handler
  exceptions; the adapter's `MessagingFuture` has no `set_exception` API. Once
  errors round-trip cleanly, convert the asserts to typed runtime checks per
  coding standards.
- **`REPORT_BLOCK_ALLOCATION` should use `self.instance_id` instead of
  `os.getpid()`** (adapter line 573). Currently scheduler-side; only used as
  EventBus metadata, so functionally OK, but inconsistent with the
  random-`instance_id` rationale here. Trivial follow-up.
- **Observability for reaping** — `MP_INSTANCE_REAPED` event,
  `lmcache_mp.live_workers` gauge, `lmcache_mp.instance_reaped_total` counter,
  `report_status` live-worker count.

## Alternatives considered

- **Scheduler liveness tracking + prefetch-job reap.** Considered in detail.
  Rejected: L1 lock TTL (default 300s) already bounds the chunk-pinning window;
  the remaining dict-entry leak is better solved by a TTL sweep that doesn't
  depend on per-instance attribution. Adding scheduler tracking would have
  required `REGISTER_SCHEDULER`, reverse-index attribution on `_prefetch_jobs`,
  two-phase commit + rollback in `lookup`, and same-PID drain logic —
  substantial complexity for a recovery-latency improvement (5min → 40s) that
  the workload doesn't need.
- **ZMQ socket-level disconnect detection.** Doesn't catch silent process hangs;
  doesn't give per-instance cleanup hooks.
- **WARMUP/ACTIVE state machine.** Eager-start collapses any meaningful warmup
  phase.
- **Auto-resurrection on PING-False.** Silent-recovery anti-pattern.
