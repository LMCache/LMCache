# Worker Liveness Tracking and Reaping (Multiprocess Mode)

Covers the MP server (`lmcache/v1/multiprocess/`) and the worker heartbeat in
the vLLM multiprocess adapter.

## 1. Problem

When an engine worker dies without sending `UNREGISTER_KV_CACHE` (SIGKILL,
OOM-kill, node loss), its per-instance state leaks on the MP server forever: the
LMCache-driven `ContextEntry` (a `GPUCacheContext` holding CUDA IPC handles), the
`EngineDrivenContextEntry` + `TransferStrategy` pair, and any blend-mode
per-instance state (e.g. CB rope caches). Nothing observes worker death; on a
shared server, leaked contexts accumulate until device memory is exhausted.

Worse, `instance_id` is `os.getpid()`: containerized pods reuse small PIDs, so a
new worker can register with a dead worker's id, and the idempotent register
silently binds it to the stale context — wrong IPC handles, corrupted transfers.

## 2. Design Overview

The server tracks two independent observations on each registered transfer
context: PING proves heartbeat participation, while the first real transfer
through that context proves its request-serving path has been exercised. The
heartbeat starts after KV cache registration and refreshes `last_seen`, but the
normal reap timeout applies to a context only after both signals have arrived.
Until then, the larger registration grace protects model warmup and CUDA Graph
capture without letting a context whose worker dies during startup leak forever.
The worker heartbeat uses a registration-aware request: the server both proves
availability and reports any expected Context that no longer exists. Missing
registrations trigger the existing idempotent re-registration callback even
when the server itself never became unhealthy.

```text
Worker REGISTER -> start heartbeat
       |
       +-- ping_registered(id, primary type) --> Server checks primary Context
       |                                      |
       |<------------- present? ---------------+
       |
       +-- missing -> idempotent re-registration
       +-- STORE/RETRIEVE -> latch transfer activity

Server Reaper: use normal timeout only after PING + transfer; otherwise grace.
```

## 3. Protocol Change

The existing `PING([int | None]) -> bool` operation is unchanged. It reports
server availability and refreshes matching Contexts; `None` marks an untracked
prober such as the scheduler adapter. It remains BLOCKING on the NORMAL pool so
a slow SYNC registration cannot stall heartbeat dispatch on the MQ main loop.

`ping_registered(instance_id, registration_type)` atomically checks and
refreshes the Worker's actual primary KV Context. `True` means present, `False`
means missing on a reachable server, and a timeout means unreachable. Optional
Contexts are refreshed but do not control recovery. Existing PING callers stay
wire-compatible. The registration type is the RPC operation name
`register_kv_cache` or `register_kv_cache_engine_driven_context`; ZMQ encodes
the new operation by name, while existing numeric operation IDs stay fixed.

## 4. Instance ID Generation

The worker adapter replaces `os.getpid()` with
`uuid.uuid4().int & ((1 << 63) - 1)`, INFO-logged at construction so operators can
correlate reap warnings with a specific pod. `uuid4` reads OS entropy (identically
seeded processes cannot collide); the 63-bit mask keeps the value int64-safe.
Every id-carrying request reads the same field, so no other payloads change.

## 5. Server Side

### 5.1 Liveness state and two-signal activation

`ContextEntry` (LMCache-driven and QStore) and `EngineDrivenContextEntry` hold
`last_seen`, `has_liveness_signal` (latched by PING), and
`has_transfer_activity` (latched by a real store/retrieve or prepare/commit).
The normal reap timeout applies only when both booleans are true; every other
state uses registration grace. PING, registration, and transfer all refresh
`last_seen`.

Activation is per registered Context: PING fans out, while a transfer activates
only its serving Context. A GPU retrieve therefore cannot activate QStore before
its first `STORE_Q`. Blend rope state belongs to the GPU Context, so reaping a
QStore with the same `instance_id` does not drop the GPU-owned mirror.

| PING seen | Transfer seen | Reap window |
|---|---|---|
| No | No | Registration grace |
| Yes | No | Registration grace (startup or request-idle) |
| No | Yes | Registration grace (legacy client without heartbeat) |
| Yes | Yes | Normal reap timeout |

Metadata-only Blend operations do not mark transfer activity. The actual
`CB_RETRIEVE_PRE_COMPUTED` scatter does.

### 5.2 Locking

The reaper runs on its own thread, so the per-instance dicts are now mutated off
the MQ handler threads. Each transfer module gains one `threading.Lock` so the
reaper's scan-and-pop cannot race a concurrent register/unregister/transfer (which
would otherwise corrupt the dict or hand out a half-removed entry). In
`EngineDrivenTransferModule` the context and strategy dicts mutate as a pair under
that lock, so a reap racing a re-register can never strand a fresh context without its
strategy. It is a leaf lock — never held across context construction, storage
calls, or any other component — so no thread ever holds two locks. External
readers use the locked accessors `get_and_touch_context_entry` (get-and-refresh) and
`context_entries_snapshot` instead of touching the dict directly.

### 5.3 Reaper

`ManagementModule` (the PING owner) owns the reaper thread, scanning every
`reap_timeout / 4`. Each target atomically selects and pops stale ids under its
module lock, then performs unregister-equivalent cleanup outside the lock. Only
a GPU KV reap invalidates GPU-owned Blend mirror state. On close, the reaper is
stopped before modules clear their state.

### 5.4 Public protocol and config

```python
class InstanceLivenessTarget(Protocol):
    # All methods default to a no-op; an implementer overrides only its role.
    def touch_instance(self, instance_id: int) -> bool: ...
    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]: ...
    def tracked_instance_count(self) -> int: ...
    def drop_instance_state(self, instance_id: int) -> None: ...
```

One protocol covers both reaper-driven roles. The transfer modules override the
liveness methods (`touch`/`reap`/`count`); `BlendModule` overrides only
`drop_instance_state` to drop its mirrored CB state. `ManagementModule` receives
all targets in a single injected list and the GPU KV target as the owner whose
reaps invalidate that mirror. Reaps from QStore or other contexts do not fan out
mirror cleanup. (Earlier a separate one-method
`InstanceReapListener` held `drop_instance_state`; it was folded in since only
`BlendModule` ever implemented it.)

Config: `worker_reap_timeout_seconds` (default `120.0`; `0` disables, otherwise
`>= 30.0`) and `worker_registration_grace_seconds` (default `3600.0`; `>=` the
reap timeout), with matching CLI flags. Grace applies until both activation
signals exist. Keep the timeout `>= 3 x` the client's
`lmcache.mp.heartbeat_interval` so a few missed pings never reap a live worker;
the worker adapter warns at startup when `3 x interval` exceeds the 30 s floor.

## 6. Adapter Side

### 6.1 Registration-time start

The heartbeat starts after successful KV registration, before the first
store/retrieve. It starts healthy, refreshes `last_seen`, and verifies the actual
LMCache-driven or engine-driven primary Context type. PING alone does not claim
request readiness: the first real transfer latches the second signal. A missing
primary Context clears health and invokes the existing idempotent recovery
callback; failure stays unhealthy and retries next heartbeat. Request paths keep
their defensive idempotent start call. A retrieve dropped while unhealthy is
still reported via `get_finished` so async loads cannot hang.

### 6.2 Recovery after a reap

```
T0        heartbeat is starved while the server remains available
T0+120s   server: entry stale -> reap pops it, frees GPUCacheContext/IPC,
          layout-desc refcount; blend rope state dropped via listener <- leak fixed
T1        worker resumes; ping_registered reports the primary Context absent
T1        health_event is cleared; recover callback re-registers the same
          instance_id before health_event is set; traffic resumes
```

The same callback still handles a full server outage. If all Contexts survived,
the idempotent server register path only refreshes `last_seen`. Locally, the
worker initializes a candidate Context before publishing it, then closes the
previous Context under the same lifecycle lock used by requests. Async close
drains background Store work before releasing SHM. Failed cleanup retains one
pending Context, keeps the worker unhealthy, and is retried before new allocation.

### 6.3 Shutdown

`shutdown()` marks the adapter closing and clears health, stops the heartbeat
without holding the lifecycle lock, then acquires that lock before UNREGISTER
and local cleanup. A limited heartbeat join is not a synchronization guarantee:
the closing check prevents late publication, and the lifecycle lock serializes
in-flight registration with UNREGISTER. Heartbeat checks stop again after recovery.

## 7. Failure Modes

| Scenario | Behavior |
|---|---|
| Worker crash (SIGKILL, no UNREGISTER) after serving | Both signals are latched; pings stop and the entry is reaped within ~`timeout + timeout/4`, using normal unregister cleanup. |
| Worker dies during startup | Before both signals exist, prolonged silence is reaped on registration grace. The startup leak stays bounded. |
| Worker remains alive but receives no traffic | PING refreshes registration grace, so request-idle time alone never reaps it. |
| Worker continues warmup after registration | PING alone retains startup grace; silence beyond it is treated as startup death. |
| Legacy or multi-context worker | Each Context needs both signals; one transfer cannot activate another Context. |
| Heartbeat thread starved, worker transferring | Store/retrieve/prepare/commit refresh `last_seen`; never reaped. |
| Server stays healthy but primary KV is reaped | The next heartbeat detects it and re-registers; one in-flight transfer may fall back. |
| Only an optional Context is reaped | Proactive recovery covers the primary KV registration. |
| Partition shorter than the reap window | No reap. On heal, the recovery path is idempotent; an existing Context is only refreshed. |
| Worker crash + restart | The new process gets a fresh uuid-derived id and a fresh entry; the dead id is reaped independently. No PID-reuse aliasing. |
| New worker with an old server | `ping_registered` is unsupported, so the worker becomes unhealthy rather than silently transferring without a Context. Upgrade both sides together. |
