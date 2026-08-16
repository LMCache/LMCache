# `lazy_offload_policy/eviction_aware.py`: Eviction-Aware Store Queue

Implements gates 1 and 3 of the store decision defined in
[lazy_offload_decision_model.md](../lazy_offload_decision_model.md); the
buffering / protection mechanism it plugs into is described in
[lazy_offload.md](../lazy_offload.md). This document is the module's contract.

## Scope and non-scope

The module is **pure policy**: no vLLM imports at runtime, no I/O, no lock (it
runs on the scheduler thread). It decides *which* buffered store operations to
release *when*; `LazyOffloadManager` executes the integration side effects
(snapshots, pinning, submission) and returns explicit actions to the connector.
Gate 2 (reuse prediction) is out of scope — phase 1 stores every admitted op
whose blocks come under eviction pressure.

## Objects

- **`BlockPoolReader`** (protocol) — read-only pool view:
  `free_queue_ranks(max_depth)` (block id → LRU eviction rank, rank 0 = next
  victim; absent = not free, *or* deeper than `max_depth`, which the policy
  treats the same way) and `block_hash(block_id)`. Production impl
  `GPUBlockPoolView` wraps the `BlockPool` bound via the vLLM
  `bind_gpu_block_pool` hook; both must never mutate pool state. The depth
  bound is not an optimisation detail the policy may ignore: this call is on
  the scheduler's critical path once per step, and an unbounded read is
  O(free blocks) — tens of thousands on a pool sized to fill the GPU.
  `collect_due` asks for `danger_depth + max_drain_per_step × largest pending
  op`, capped by the total pending blocks. This is a safe upper bound on the
  deepest rank a pin cascade can reach without walking the pending queue to
  size every operation; it skips the call entirely at danger depth 0.
- **`PendingStoreOp`** — one deferred store: opaque `store_metadata` (the
  ready `LMCacheMPRequestMetadata`), the covered blocks' hash snapshot taken
  at admission, `prefix_start_tokens` / `prefix_end_tokens` (the op's token
  range; the start detects deduplication holes in the pending list), and
  `cache_salt` (part of the op's content identity).
- **`EvictionAwareStoreQueue`** — the policy object, one per connector.

## Per-step protocol (policy-caller obligations)

`LazyOffloadManager` is the production caller that fulfills this contract; the
connector only forwards lifecycle events to the manager.

1. Route each `GetStoreMetadata` result to `admit(op)` instead of the step
   metadata. Handle the outcome:
   - `ADMITTED` → nothing now.
   - `REJECTED_UNHASHED_BLOCK` → **skip and warn** (a hash-less block's later
     eviction is undetectable: evicted-and-reallocated also reads `None`).
     The tracker has already advanced past the skipped range by the time the
     op reaches `admit`, so the request's later chunks are unreachable; the
     queue blacklists the request and rejects them as prefix-broken. With
     plain prefix caching (enforced at connector init) chunk-aligned ranges
     never cover unhashed blocks, but hybrid-attention models (sliding
     window, mamba) can place hash-less null blocks in block tables.

     Measured on `google/gemma-3-270m-it` (18 layers, 5 sliding-window
     layers of 512 tokens for every full-attention layer, so vLLM builds
     six kernel groups). The case that reaches it is not the long request
     itself — its blocks are in the window as each chunk is buffered — but a
     request whose prefix comes back from vLLM's *own* prefix cache:
     `SlidingWindowManager.find_longest_cache_hit` prepends
     `block_pool.null_block` for every out-of-window position, and those
     positions hold no KV for the sliding-window layers. The eager path has
     no admission step and stores them: on a 2166-token prompt replayed
     against an empty LMCache, 7 of the prefix's 8 chunks came back with
     different bytes under the same content-addressed key (the one that
     matched is the chunk still inside the attention window). This is what
     the rejection avoids, and why it is a skip rather than a best-effort
     store. Both counters are exercised by layer-1 scenario S18.
   - `REJECTED_PREFIX_BROKEN` → **skip** (an earlier chunk was dropped; this
     chunk would be unreachable on retrieval).
   - `DEDUPLICATED` → nothing now (identical content — same salt, range, and
     block-hash chain — is already buffered under another request and will be
     stored or dropped with that op; this op must not defer its own request's
     teardown). Deduplication is what bounds the queue: without it every
     request over a hot shared prefix (blocks never in the free queue, so
     never due) would buffer its own copy indefinitely; with it the queue is
     bounded by the unique cached content on the GPU. A hit is validated
     against the pool: if the covering op's block snapshot is no longer
     intact (its blocks were recycled while it waited for its eviction
     drop — e.g. behind an in-flight batch, or by this very step's
     allocation), or an earlier pending op of the covering op's request has
     lost a block (the next drain then prefix-closes over the cover too),
     the new op is admitted instead and takes over the content key; a
     doomed op never absorbs a live copy. Past that check it is
     optimistic: if the covering op is dropped later, chunks the
     deduplicated request stores past that point are unreachable until a
     future request re-buffers the prefix — wasted storage, never
     corruption. A deduplicated chunk also
     leaves a *hole* in its request's pending list; emission never spans a
     hole (each batch is coalesced into one contiguous store op), so the ops
     on each side go out in separate batches.

     **Range equality is required, and a capped step budget breaks it.** An
     op covers the range one step made known, so the same tokens produce
     different ops depending on how the prefill was chunked. Measured with
     the same 1965-token prompt sent twice: with `max_num_batched_tokens`
     capped at 512 the first request admits four per-step ops
     (512/1024/1536/1792) while the repeat prefix-cache-hits the whole
     prompt in one step and admits a single op over the whole range —
     `deduplicated` stays 0 and the content is buffered twice, under two
     different chunkings. Uncapped (one step per prefill), the two match and
     the repeat deduplicates. So the queue is bounded by the distinct
     (range, content) pairs resident, not by the request count: still a
     bound that does not grow with load, but the constant is the number of
     distinct chunkings of a hot prefix, not 1.

     A consequence for the doomed-cover check above: on vLLM it is defensive
     rather than load-bearing. `add()` runs before `collect_due()` in a step,
     so an op that becomes doomed in a step is dropped in that same step
     unless its request holds an in-flight batch — and the follower whose op
     could hit a doomed cover has to share its exact range, which (per the
     paragraph above) means an uncapped step budget, where a request has a
     single op and so never holds a batch in flight with siblings pending.
     The two conditions pull against each other; the branch is covered at
     layer 0 (`test_lazy_offload_eviction_aware.py`) and was not reachable on
     hardware.
2. Once per step, call `observe_step(gross_blocks_allocated,
   est_next_step_blocks, allocated_block_ids)` and then `collect_due()`.
   The connector obtains the ids from the scheduler output. The queue keeps a
   block-to-request reverse index and revalidates only requests touched by
   those allocations or represented in the bounded free-queue snapshot; a
   caller that cannot supply ids passes `None`, which requests a compatibility
   full-scan validation pass.
3. For every op in `DrainResult.to_store` (already ordered): pin (`touch`)
   its blocks, **coalesce each request's released ops into one store op**
   (the worker adapter tracks a single in-flight store future per request),
   and put it into this step's connector metadata. `dropped_*` lists need no
   action beyond accounting.
4. On the store-completion receipt: unpin with `free_blocks(prepend=True)`
   (a stored block has a copy below the GPU, so among free blocks it should
   die first) and call `notify_stored(id)` — the queue holds back a
   request's remaining ops while a batch is in flight; a True return means
   the request is finished and fully drained, so its session may end.
5. On `request_finished`: call `mark_request_finished(id)`; True means
   stores are pending or in flight — defer `end_session` until the id
   appears in `DrainResult.released_requests` (remaining ops all dropped) or
   `notify_stored` returns True (stored).
6. When the request's buffered state goes stale — today only the preemption
   tracker reset (the recreated tracker re-produces metadata from token
   zero, overlapping anything buffered) — call `drop_request(id)`. It
   discards pending ops only: an in-flight batch stays tracked until its
   receipt, so a re-admitted op cannot be emitted while the worker still
   holds an outstanding store for the request. The controller advances the
   store epoch, making the surviving submitted batch stale for failure
   interpretation (see step 7). An abort is **not** a drop: it routes
   through `request_finished` → `mark_request_finished`, and the aborted
   request's buffered ops stay storable until drained or evicted.
7. When a receipt reports the store **failed** (worker-side failure signal):
   call `mark_store_failed(id)` before `notify_stored(id)`. It drops the
   request's held-back ops and rejects its later chunks (without the failed
   prefix they would be stored unreachable), while leaving the finished and
   in-flight markers alone so the accompanying receipt still tears the
   request down through `notify_stored` as usual. Before calling the policy,
   the controller compares the submitted batch epoch with the request's
   current epoch. It ignores an old-epoch failure because operations admitted
   after reset or reuse do not depend on the failed prefix; the receipt still
   clears and unpins the old batch.
8. When a **new** request's id is first seen (tracker creation): call
   `reclaim_finished_request(id)`. In lazy mode a finished request leaves
   vLLM's request table immediately (`request_finished` returns False), so a
   client-supplied id can return while its previous owner's teardown is
   still deferred; without the reclaim the two requests' pending lists
   conflate (the predecessor's eviction drop prefix-closes over the
   successor's intact ops, and the deferred release fires while the
   successor is live). The reclaim discards the predecessor's buffered ops
   and its finished marker; a True return means the caller must
   `end_session(id)` now, before the successor's first operation. With an
   in-flight predecessor batch it returns False instead: successor arrival
   advances the epoch and the id-keyed session, which now covers both requests,
   ends once through the successor's own lifecycle — the predecessor's
   receipt only clears the in-flight hold. The marker must not ride the
   receipt: the successor is live when the reclaim fires, so any teardown
   the marker later authorizes (the predecessor's receipt, the successor's
   own receipt, or an eviction drop landing the id in
   `released_requests`) would end a running request's session. Note which
   deployments can produce the duplicate at all: vLLM's input processor
   appends 8 random characters to every externally supplied id
   (`assign_request_id`), so an HTTP client cannot force one unless the
   engine runs with `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1`; callers that
   drive the engine core directly with their own ids always can.

**Prerequisite for 1 and the dedup path**: the connector must record vLLM's
prefix-cache hit in the tracker even when the LMCache lookup misses. In lazy
mode a follower over a hot APC-shared prefix always misses the lookup (the
predecessor's ops are buffered, not stored); without the vllm-hit share,
`GetStoreMetadata` stages under one chunk and the follower never reaches
`admit` — deduplication is dead code for followers, and a dropped
predecessor op is never re-buffered while APC keeps hitting. The recording
is mode-independent by design: in eager mode, an APC-hit request whose
lookup misses (predecessor's store in flight, or data evicted from LMCache)
now issues a store covering its full prefix at once instead of accumulating it
over decode steps. That backfills the under-store the old behavior left
when LMCache had really evicted the data, at the cost of duplicate stores
in the in-flight window (eager has no client-side dedup; content-addressed
keys make them idempotent server-side). It also makes the
`cached_token_stats` reported through `kv_transfer_params` show the true
vLLM hit instead of 0 on a lookup miss.

## Decision rule

- **Danger depth** = `ceil(max(EMA(gross allocation/step), next-step
  feedforward) × horizon_steps)`; below half a block over the horizon it is 0
  (a decayed EMA must not hold the depth at 1 forever). An idle engine never
  drains — free-queue *position* alone is never a trigger (that would be the
  inverted-gate-1 anti-pattern, decision model §6).
- An op is **due** when any covered block's rank < danger depth. Blocks not
  in the free queue (in use / resurrected) are not at risk.
- **Horizon calibration.** The default is 2.5 scheduler steps. A fine sweep
  over 2.0–8.0 on two opposing Qwen3-8B/H200 workloads selected it as the
  measured compromise: three 120-request hot/cold runs at 2.5 completed in
  27.0–27.1 s with 0.952–0.957 cache coverage and three lower-tier eviction
  cycles, while 2.0 took 31.6–32.2 s and 4.0 took 36.2–36.3 s. On the
  no-hot-set GSM8K workload, 2.5 retained 0.945–0.961 coverage versus 0.961
  once the horizon reached 3.0–4.0. The value is a calibrated default, not a
  universal optimum: increase it when eviction loss is more important than
  filtering, and decrease it when lower-tier write or eviction pressure is
  the limiting cost.
- **Pin-cascade shift**: emitting a segment pins its blocks out of the free
  queue, moving every block behind them toward the head before the next
  step's allocation runs. The shift is the number of **unique emitted
  blocks that were in the free-queue snapshot**: an in-use block does not
  leave the queue, and a block shared by multiple emitted ops leaves it only
  on the first touch. Within one `collect_due` call, each later candidate is
  therefore checked against `danger depth + free blocks removed so far in
  this call`; without this, a candidate teleported into the danger window by
  an earlier emission loses its tail to the next allocation before the next
  drain can see it (observed as `dropped_evicted` under back-to-back drains).
  The first emission still requires a plain danger-depth hit, so this never
  opens the gate on an idle system; dropped (unpinned) segments do not extend
  the shift.
- **Prefix closure** (amendment A1): a due op releases the request's ops from
  the front through the last due one; a data-loss drop (hash mismatch) drops
  from the first lost op through the tail and blacklists the request's later
  chunks; the intact stored prefix stays pending and stays valuable.
- **Gate 3**: when a request comes due with known prefix <
  `min_prefix_tokens`, all its ops are dropped (the due front is dying, which
  breaks the chain for the rest). The threshold is the offline break-even
  prefix length; 0 disables.
- A due segment is cut at the first deduplication hole before emission
  (the batch must coalesce into one contiguous token range); the request
  keeps its due-rank urgency, and the post-hole ops follow in a later
  batch once the front run's receipt arrives.
- Cross-request drain order = min due rank ascending; `max_drain_per_step`
  bounds the per-step D2H burst. The cap may split a request's due segment,
  but only ever emits a front slice of it, so within-request prefix order is
  preserved and the remainder stays pending.
- **Sizing the cap.** It bounds emissions per step while a prefilling request
  *admits* one op per step, and a request with a batch in flight is skipped
  entirely until its receipt arrives (one more step). So the cap has to sit
  above the concurrent prefill admission rate, or the queue cannot work off
  a backlog and buffered ops are lost to eviction instead of stored.
  Measured on a 448-block pool with a 512-token step budget, one 4-op
  request buffered ahead of five prefilling fillers: at the default 64 the
  workload emitted 21 of 24 admitted ops, dropped 1 and left none pending;
  at 1 it emitted 11 of 26, dropped 6 and left 9 pending at shutdown, and
  the buffered request stored its first two ops while losing the other two
  (prefix closure held — the replay retrieved exactly the surviving 1024 of
  1792 tokens). A cap near 1 is a steady-state loss setting, not a
  burst-shaping one.

  There is no static validation for this: the break-even depends on how
  many requests prefill concurrently, which the policy learns only at
  runtime. The sensor is therefore a runtime one. `DrainResult.ops_held_back`
  reports what a drain found due and did not emit (a lower bound: candidates
  the loop never reached are not counted, their due-ness being unevaluated),
  `throttled_drains` counts the drains that held anything back, and the
  pending store logs one WARNING per process the first time a drain both
  held ops back and lost ops to eviction — the pair that separates a cap
  merely delaying a burst from one below the workload's admission rate.
  Neither symptom alone warns: ops lost without the cap binding is ordinary
  pressure, and a cap that binds without loss is the knob doing its job.
- **Idle consequences**: receipts travel in worker metadata, which only
  flows on steps that schedule tokens. If the engine goes idle with a
  batch in flight, its pins and its request's session stay held until the
  next non-empty step delivers the receipt; finished requests whose ops
  never come due likewise hold their sessions open. Both resolve on the
  next activity — nothing leaks permanently, by design ("idle never
  drains" also means "idle never settles").

## Scheduler-path complexity

A dedicated pending-operation owner maintains the primary per-request lists
together with their content covers, admission order, block-to-request reverse
index, per-request block reference counts, and bounded operation-size multiset.
Admission, replacement, and departure update primary storage and every derived
index through one atomic API. A production drain therefore walks only the
bounded free-queue window and the requests represented in that window or
touched by this step's allocations. Its cost is proportional to the pressure
window and drain cap, not total pending queue depth. The pure-policy tests
retain an `allocated_block_ids=None` compatibility path that performs a full
validation pass.

Request lifecycle is stored separately from pending operations in one
per-request record (`prefix_broken`, `finished`, and `in_flight`). Stale submitted batches
are identified by the controller's store epochs rather than policy state. This
keeps multi-flag transitions such as preemption, id reuse, failed stores, and
completion receipts atomic in one owner rather than synchronizing parallel
sets. Empty lifecycle records are pruned, so completed request ids do not
accumulate.

## Observability

`stats()` returns cumulative counters. `dropped_evicted` is the gate-1 sensor
(drop rate: data lost before we drained — lower the horizon is too tight);
`emitted / admitted` is store precision's denominator; `rejected_short_prefix`
audits gate 3. Tests: `tests/v1/test_lazy_offload_eviction_aware.py` (pure, no vLLM).

The counters surface in the scheduler process log, not in vLLM's
`get_kv_connector_stats` plumbing (that hook is polled worker-side, where the
policy does not live). Three hooks, all on the pending-store facade:

- each drain that dropped ops logs one aggregate INFO line **per cause**
  (`dropped N store op(s): blocks evicted before drain (req (prefix P), ...)`
  and `dropped N store op(s): request prefix below the break-even length
  (...)`, each naming at most 8 ops and counting the rest), so both kinds of
  cache-quality loss are attributable to a request without running at DEBUG,
  while a burst that evicts a large queue cannot flood the scheduler hot
  path; per-op detail logs at DEBUG. The later chunks a broken request keeps
  producing are rejected at admission and log at DEBUG only: their cause was
  already reported, and one broken request produces many of them;
- every drain re-logs the whole ledger as one greppable `key=value` line
  (`Lazy offload counters: admitted=... emitted=... pending=N`) when the
  counters changed, throttled to one line per 5s. `pending` is the queue
  depth at the same instant, which closes the line as an equation over
  exactly six outcome counters:

  ```
  admitted == pending + emitted + dropped_evicted + rejected_short_prefix
              + dropped_on_request_drop + dropped_failed_store
              + dropped_id_reuse
  ```

  so a reader can separate an operation still waiting for pressure from one
  that left the queue without incrementing any outcome counter. The set is
  neither "every `dropped_*` counter" nor "every drop and reject":
  `rejected_short_prefix` belongs in it although it is not named `dropped_*`
  (gate 3 discards ops that were admitted), while `rejected_unhashed`,
  `rejected_prefix_broken` and `deduplicated` must stay out of it — those ops
  are turned away at admission and never counted in `admitted`. Summing by
  name instead of by this list makes the equation fail the moment gate 3
  fires. `throttled_drains` stays out for a different reason: it counts
  drains, not operations, so it belongs alongside the step count rather
  than in an equation over ops;
- connector `shutdown()` (invoked by vLLM's scheduler shutdown) calls
  `log_final_stats()`, which emits the exact final ledger
  (`Lazy offload final counters: ...`). Best-effort: `vllm serve` under
  SIGINT force-kills the engine core (abort mode) and can beat scheduler
  shutdown to it -- that is why the periodic line exists. A log reader
  should take the last line matching `Lazy offload (final )?counters:`.
