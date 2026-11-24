# Prefetch retained-bytes budget

Status: implemented (`PrefetchController`, `BaseL1Manager`, `LookupModule`,
CacheBlend lookup). Validated on a scaled-down workload B; see
[End-to-end result](#end-to-end-result). Fixes the high-concurrency reuse collapse seen with CacheBlend +
L2 (VAST), where the only working mitigation today is a client-side admission
cap of 12 requests. The cause is in the generic MP prefetch path, so the fix
lives in `PrefetchController` and `BaseL1Manager`, not in the CacheBlend plugin.

## Problem

1. **Every lookup starts a prefetch.** `LookupModule.lookup` and the CacheBlend
   legs (`_submit_prefix_leg`, `_sparse_prefetch_submit`) call
   `StorageManager.submit_prefetch_task` as soon as the lookup arrives.
2. **vLLM looks up the whole waiting queue at once.** While a lookup is pending
   the connector returns `(None, True)`, and vLLM's scheduler `skip_request`s
   the request and moves on to the next one *before* trying to allocate GPU
   blocks. A burst of N requests sends N lookups in the first step or two,
   whatever the GPU can actually hold.
3. **Admission bounds only in-flight bytes.** `_can_admit_new_prefetch` checks
   `staging + lookup_bytes < inflight_watermark * total` and free bytes. Once a
   prefetch finishes, its objects sit in L1 *read-locked* (not evictable) until
   the request retrieves them or the read TTL expires, and they leave
   `staging_bytes`. These finished-but-unused bytes ("retained" bytes below) are
   not bounded by anything.
4. **Follow-up legs queue behind new requests.** CacheBlend runs a prefix leg,
   then a sparse leg for the same request. The sparse leg joins the *tail* of
   the FIFO pending queue, behind every other request's prefix leg. Under a
   burst, a request near the front waits for its sparse leg until the whole
   queue has gone through the prefix leg. This is likely part of the summed
   prefetch duration blowing up (47,210 s with no cap vs 365 s with cap 12) and
   still needs to be confirmed.

Under a burst, L1 fills with data pinned for requests that won't run for
minutes. New prefetches fail to reserve space (`l1_oom`), so prefetch chunks
fail. Unlocked warm data gets evicted, so L1 hits disappear. Requests that do
run find their chunks missing, and CacheBlend recomputes them.

Rule of thumb: trouble starts when the number of admitted requests times the
prefetch bytes per request is larger than L1. For gpt-oss-120b, assuming full
KV on all 36 layers (36 × 8 KV heads × 64 × 2 × bf16), that is about 74 KB per
token, or about 8.8 GB per 120k-token CacheBlend prompt, so about 34 of them
fill a 300 GB L1. Plain prefix MP stops at the first miss (about 2.2 GB on the
RAG workload). That explains why lmc-l2 looked fine at cap 50. The prediction,
not yet measured, is that lmc-l2 degrades too with no cap.

## Goals

- Bound the L1 bytes that prefetch may pin for requests that have not run yet.
  This becomes lookahead that adapts to L1 capacity, with no knowledge of the
  engine's batch.
- No deadlock: a request that holds budget can always finish its lookup.
- No change for L1-only deployments or for traffic that stays under the budget.
- Works for one MP server shared by several engines, which a cap in each
  connector cannot do.

## Design

### 1. Retained-bytes ledger in `BaseL1Manager`

Track the objects that a prefetch loaded from L2 and that are still
read-locked:

```python
self._prefetch_retained: dict[ObjectKey, int] = {}   # key -> charged bytes
self._prefetch_retained_bytes: int = 0
```

- **Charge:** when `PrefetchController` admits an L2-loaded object into L1
  (the `finish_write_and_read_lock` step after a load), charge its size once
  per key. A key shared by several requests through dedup is charged once.
  Objects that L1 already held, locked during the lookup pass, are **not**
  charged. They cost no new memory, and charging them would throttle L1-only
  hits.
- **Release:** drop the charge when the entry's read lock is no longer held:
  - in `finish_read` when the count reaches 0;
  - in `delete` / eviction / `clear`;
  - by a lazy sweep in `get_prefetch_retained_bytes()` for entries whose read
    TTL expired. `TTLLock` expires lazily, so a counter cannot see expiry on
    its own. The sweep only walks charged keys.
- **Public API (new, on `L1ManagerInterface`):**
  - `charge_prefetch_retained(keys)`: the controller calls it right after
    `finish_write_and_reserve_read` for the keys it loaded. It is a separate
    call, so the signature that the serde wrapper shares stays unchanged.
  - `get_prefetch_retained_bytes() -> int`
  - The shadow L1 reports 0 and charges nothing, because views hold no local
    bytes.

The read lock already protects charged objects from eviction. The ledger adds
the missing bound on how much of them there can be.

### 2. The in-flight watermark counts retained bytes

There is still one watermark, `inflight_watermark`
(`--l2-prefetch-inflight-watermark`, default 0.5). It now covers everything
prefetch holds in L1, not only loads in flight:

```
before: staging + lookup_bytes < inflight_watermark * total_bytes
after:  staging + lookup_bytes + retained + request_bytes
            <= inflight_watermark * total_bytes
```

- "In flight" now means held for a request that has not run yet: buffers
  being staged, lookups not yet planned, and loaded data still read-locked.
  A separate retained watermark would only duplicate the knob, since this
  condition is strictly stronger than the old one at the same value.
- Under the same default, the old condition is a subset of the new one, so
  existing deployments only become stricter, and only by the data they
  retain. L1-only setups are unaffected, because nothing loaded from L1 is
  charged.
- Starvation guard: change the current "admit if nothing in flight" to "admit
  if nothing in flight **and** `retained == 0`", so one oversized request still
  runs.
- A request that waits in the pending queue answers `None` to
  `query_prefetch_status`. The connector keeps it waiting (`(None, True)`), and
  vLLM skips it, which is what happens today for any in-progress lookup. The
  connector's per-RPC deadline counts each status RPC separately, not the whole
  wait, so a queued lookup does not time out
  (`mp_lookup_state.py`, `on_scheduler_check`).

At 300 GB L1 with the default 0.5 watermark (150 GB), about 17 CacheBlend
requests (8.8 GB each) are staged or retained at once: about 8 running and 9
queued, more than the cap of 12 that works today. The lookahead scales with L1 size
and prompt size with no tuning.

### 3. Continuation tasks have strict priority

`PrefetchTaskSpec` gains `admission: PrefetchAdmission`, an enum with values
`NEW` (the default) and `CONTINUATION`.

- `CONTINUATION` tasks go to a separate queue with strict priority: no `NEW`
  task starts while one waits. They still pass the watermark and free-space
  checks.
- They do **not** bypass the checks. Bypassing would reopen the overflow: a
  prefix leg that hits little leaves almost nothing retained, so more prefix
  legs are admitted, and every large sparse leg that follows would skip the
  budget.
- Priority is enough for progress. A `NEW` leg is charged at the worst case
  of the whole prompt, which covers the prefix hit plus the sparse remainder.
  So once in-flight lookups settle, a waiting continuation fits in what its
  lookup was charged.
- Set by:
  - the CacheBlend sparse leg (`_sparse_prefetch_submit`) when its prefix leg
    was submitted for the same lookup;
  - `LookupModule.lookup` when the session already has a completed lookup
    result (the re-lookup after the APC hit shrank, which keeps the old range
    locked until the next free).

This is required for deadlock freedom, not just for speed. Without it, new
requests could take the budget that A's sparse leg needs, so A never finishes
its lookup, never runs, and never releases what its prefix leg holds.

A **stall rule** remains as a safety valve: with nothing in flight, a head
refused for 30 s (`PREFETCH_STALL_ADMIT_SECONDS`) is admitted anyway. It
covers engines that wait on lookups in order instead of skipping pending ones
(vLLM skips them), and the edge cases where the worst-case charge does not
cover a continuation, such as sparse legs that read groups the prefix leg
does not.

### 4. Cancelling prefetches for finished or aborted requests

`StorageManager.cancel_prefetch_task(handle)` →
`PrefetchController.cancel_prefetch_request(id)`:

- **Still pending:** remove it from the queue. No L2 lookup is issued.
- **In flight:** mark it cancelled. `_finish_request` then releases every lock
  it took instead of publishing a result.
- **Callers:** `LookupModule.end_session`, and the CacheBlend session-destroy
  listener, which drops a lookup that is still polling its legs.
- `free_lookup_locks` does **not** cancel. After an APC-shrink re-lookup, a
  free for the abandoned range can arrive while the request's current lookup
  is still running.
- To release a finished-but-unconsumed result, the controller keeps the
  finished request beside each unconsumed `LOCK` result. Consuming the result
  drops that request.

Without this, a request aborted mid-prefetch pins budget until the read TTL
(3600 s in the ticket's configuration), and an aborted request that is still
queued runs its prefetch later for nothing.

### 5. Observability

- Gauge `lmcache_mp.l1_prefetch_retained_bytes` (per L1), and
  `prefetch_retained_bytes` in the L1 `report_status`.
- In `PrefetchController.report_status`: `continuation_queue_size` and
  `stall_admissions`, next to the existing `pending_queue_size`.
- Follow-up: a queue-wait histogram and a cancel counter.

## Why deadlock cannot happen

Budget is held only by admitted requests, through their retained or in-flight
bytes. For each admitted request:

1. Its remaining legs are `CONTINUATION` and have strict priority. They fit
   in what the first leg was charged once in-flight lookups settle (§3), so
   its lookup finishes in bounded time. The stall rule covers what the
   worst-case charge misses.
2. A waiting vLLM request whose lookup has finished holds no GPU blocks. GPU
   blocks are freed by running requests, whose progress does not depend on
   prefetch admission. So the request is eventually scheduled, retrieves, and
   releases its read locks.
3. If the request is aborted instead, §4 releases its locks right away.

The read TTL remains as a safety net for clients that never send
`END_SESSION`.

## Alternatives considered

- **Cap in the connector on outstanding lookups (count K).** Simple, but every
  vLLM instance attached to the same MP server gets its own K, so it does not
  bound shared L1. K also has to be tuned per model and prompt size. It could
  still be added later as a complement.
- **Engine hint for scheduling position (lookahead by rank).** Needs a vLLM
  change. Byte admission gets most of the benefit when vLLM schedules in
  first-come order.
- **Only extending the read TTL or protecting staged data.** Staged data is
  already protected while locked. The problem is the amount, not the
  protection.
- **Chunked prefill for CacheBlend (plugin PR #191).** Keeps more requests
  running, but does not bound prefetch for the queued ones. It is independent
  of this fix and should be measured separately.

## Known limitations

- First-come admission order can disagree with vLLM priority scheduling.
  Low-priority requests that were admitted first can hold the budget while
  higher-priority ones wait. Still correct, but not optimal.
- `request_bytes` is the worst case (every cell loaded from L2). A request
  that is mostly in L1 is under-admitted. A later refinement can estimate it
  from L1 residency (`contains`) before admission.
- Several L1 managers keep today's conservative per-L1 charge.

## Tests

Unit tests (`tests/v1/distributed/`, `tests/v1/multiprocess/`):

- ledger (`test_l1_interface.py`): charged once per object; dropped when the
  last reader finishes, on delete and on read-TTL expiry; unlocked or missing
  keys are never charged.
- admission (`test_prefetch_controller.py`): unconsumed loads hold back new
  requests until a reader releases them; L1 hits are not charged; the
  watermark test consumes each result to let the next request start.
- continuations: one that fits starts ahead of a waiting `NEW` request; a
  waiting one blocks `NEW` requests that would fit.
- stall rule: with nothing in flight, a blocked head is admitted after the
  timeout.
- cancel: a queued request never loads; an in-flight or unconsumed one
  releases every lock; a cancel after the result is consumed is ignored.
- callers: a repeat lookup is a continuation; `end_session` cancels an
  unconsumed prefetch; the CacheBlend destroy listener cancels a lookup that
  is still polling its sparse leg.

## End-to-end result

Measured with an earlier revision of this change that had a separate
retained watermark of 0.6 next to the 0.5 in-flight watermark, which is the
same condition at 0.6. Single runs on yotta-h200-26: 2× H200, gpt-oss-120b TP2, vLLM 0.33.0,
CacheBlend plugin v0.5.8rc3, no chunked prefill, chunk size 1024. vLLM's KV
cache was 1.25M tokens, so about 9.6 prompts of 131k tokens fit at once. L1
was 180 GB and the `fs_native` L2 sat in `/dev/shm`. 150 docs of 30,720
tokens were warmed once. Each burst sent N prompts of 4 random docs at once
(about 123k tokens each, 100 output tokens). Reuse is LMCache's tokens
loaded plus prefix hits, over the requested tokens. The `gpu_check` verdict
for the window was clean.

| Arm | N | Time to answer all | Speedup vs vLLM | Reused | Failed L2 prefetch chunks |
|---|---|---|---|---|---|
| vLLM alone | 50 / 100 | 192.2 s / 384.1 s | 1.0x | 0% | - |
| dev, no cap | 50 | 60.4 s | 3.2x | 98.9% | 120 |
| dev, no cap | 100 | 197.3 s | 2.0x | 75.3% | 10,222 |
| dev, client cap 12 | 100 | 104.4 s | 3.7x | 99.2% | 89 |
| this fix, no cap | 50 | 49.1 s | 3.9x | 99.9% | 0 |
| this fix, no cap | 100 | 88.0 s | 4.4x | 99.9% | 0 |

Every arm had 0 L1 read failures and 0 failed KV loads, so this setup did not
reproduce those two symptoms from the ticket.
