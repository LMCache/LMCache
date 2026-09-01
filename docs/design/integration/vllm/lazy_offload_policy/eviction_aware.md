# `lazy_offload_policy/eviction_aware.py`: Eviction-Aware Store Queue

The buffering / protection mechanism this policy plugs into is described in
[lazy_offload.md](../lazy_offload.md). This document is the module's contract.

## Scope and non-scope

The module is **pure policy**: no vLLM imports at runtime, no I/O, no lock (it
runs on the scheduler thread). It decides *which* buffered store operations to
release *when*; `LazyOffloadManager` executes the integration side effects
(snapshots, pinning, submission) and returns explicit actions to the
connector. Reuse prediction is out of scope: the policy stores every admitted
operation whose blocks come under eviction pressure.

## Objects

- **`GPUBlockPoolView`** -- read-only view of the `BlockPool` bound via the
  vLLM `bind_gpu_block_pool` hook, and the policy's only pool access:
  `free_queue_block_ids()` (a *lazy* iterator over the free queue from the
  eviction head; a block's position is its LRU rank, rank 0 = next victim),
  `is_free(block_id)` (O(1) queue membership), and `block_hash(block_id)`.
  None of them may mutate pool state.

  Laziness is contractual, not an optimisation detail: the walk runs on the
  scheduler's critical path once per step and the full queue is O(free
  blocks). The drain consumes the iterator through `_FreeQueueWindow`,
  which opens at `danger_depth` and widens only by the blocks an emission has
  already pinned out of the queue (see pin cascade), so a step reads exactly
  the ranks its decisions compare -- and nothing at danger depth 0. The depth
  deliberately does not scale with `max_drain_per_step`: that budget bounds
  the D2H burst, not the rank read. `is_free` exists so pin accounting asks
  the pool, not the window: a pin deeper than the window still shifts the
  queue.
- **`PendingStoreOp`** -- one deferred store: opaque `store_metadata` (the
  ready `LMCacheMPRequestMetadata`), the covered blocks' hash snapshot taken
  at admission, the op's token range, and its admission drain and timestamp.
- **`EvictionAwareStoreQueue`** -- the policy object, one per connector.

## Per-step protocol (policy-caller obligations)

`LazyOffloadManager` is the production caller; the connector only forwards
lifecycle events to it.

1. Route each `GetStoreMetadata` result to `add(meta, block_hashes, epoch)`
   instead of the step metadata. The queue takes custody or drops the
   operation itself; the caller has nothing to submit either way. Two
   operations are dropped at this point and logged:
   - A **hash-less covered block**. Its later eviction is undetectable
     (evicted-and-reallocated also reads `None`), so storing it risks filing
     wrong bytes under a content-addressed key: hybrid-attention models
     (sliding window, mamba) put hash-less null blocks into block tables when
     vLLM's own prefix cache serves a prefix whose out-of-window positions
     hold no KV. The tracker has already advanced past the skipped range, so
     the queue blacklists the request and drops its later chunks too.
   - A **broken prefix**: an earlier chunk of the request was dropped, so
     this one would be unreachable on retrieval.
2. Once per step call `drain(signals)` with that step's `DrainSignals`
   (gross blocks allocated, next-step estimate, allocated block ids, and the
   finished and in-flight request-id sets). Pending ops of finished requests
   wait out their eviction clock -- that is the point of lazy offload.
   Validation is incremental: the queue keeps a block-to-request reverse
   index and revalidates only requests touched by this step's allocations or
   represented in the free-queue window. Requests blocked by a submitted
   batch are skipped until their receipt.
3. For every op in `LazyOffloadDrain.items` (already ordered): pin (`touch`)
   its blocks, coalesce each request's released ops into one store op,
   register the submitted batch and its epoch, and put it into this step's
   connector metadata. `dropped_evicted` needs no action beyond accounting;
   `emptied_request_ids` is only a buffer transition -- teardown additionally
   requires the registry to say the request is finished with no submitted
   batch.
4. On a store-completion receipt, complete the batch and unpin. End the
   session only when the registry says the request is finished and the policy
   has nothing pending.
5. On `request_finished`, record `FINISHED` in the registry; end immediately
   only under the same predicate, otherwise a later drain or receipt applies
   it.
6. On preemption tracker reset, advance the epoch and call
   `drop_request(id)`: buffered ops and prefix-validity state are discarded;
   an already submitted batch stays registered and blocks new emission until
   its receipt. An abort is **not** a drop: its buffered ops remain
   storable until drained or evicted.
7. On a failed receipt, compare the batch epoch with the current request
   epoch. A current-epoch failure calls `mark_store_failed(id)`, which drops
   held-back ops and marks the prefix broken; an old-epoch failure never
   enters the policy. Both paths complete the receipt and unpin.
8. On request-id reuse, detect the `FINISHED` predecessor in the registry,
   advance the epoch, and call `discard_for_reuse(id)`. With a submitted
   batch outstanding, the id-keyed session spans both epochs and ends once
   through the successor's lifecycle. (vLLM's HTTP layer randomizes external
   ids, but direct engine callers and
   `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` reach this path.)

**Prerequisite for 1**: the connector must record vLLM's prefix-cache hit in
the tracker even when the LMCache lookup misses. A lazy follower over a hot
APC-shared prefix always misses the lookup (the predecessor's ops are
buffered, not stored); without the vllm-hit share it never reaches `admit`
and stores nothing at all. The recording is mode-independent by design: in
eager mode an APC-hit request whose lookup misses now stores its full prefix
at once instead of accumulating it over decode steps -- backfilling the
under-store when LMCache really evicted the data, at the cost of duplicate
in-flight stores (idempotent under content-addressed keys) -- and
`cached_token_stats` reports the true vLLM hit instead of 0.

## Decision rule

- **Danger depth** = `ceil(max(EMA(gross allocation/step), next-step
  feedforward) x horizon_steps)`; below half a block over the horizon it is
  0 (a decayed EMA must not hold the depth at 1 forever). Free-queue
  *position* alone is never a trigger -- an idle engine never drains.
- An op is **due** when any covered block's rank < danger depth. Blocks not
  in the free queue (in use / resurrected) are not at risk.
- **Pin cascade.** Emitting a segment pins its blocks out of the free queue,
  moving every block behind them toward the head before the next step's
  allocation runs. Within one drain, each later candidate is
  therefore tested against `danger_depth` plus the unique emitted blocks
  that were in the queue so far (an in-use block does not leave the queue; a
  shared block counts once), and the window widens by the same amount to
  reveal the next candidates. The first emission still needs a plain
  danger-depth hit, so the extension never opens the gate on an idle system;
  the alternation terminates because each round either emits (the cap is
  finite) or finds nothing due. Dropped (unpinned) segments do not extend
  the shift.
- **Deferral deadline (`max_deferral_seconds`)**: the danger window is a
  *spatial* signal -- how close a block sits to the free-queue head -- which
  answers when the block dies on the GPU, not when the conversation comes
  back. Those two clocks are independent, and when the reuse interval is the
  shorter of the two the window emits after the entry was already needed:
  the store lands, but too late to be found. A request whose oldest pending
  op has waited longer than this bound is due regardless of rank, is decided
  before the window-driven candidates, and releases its whole surviving
  front (still subject to `max_drain_per_step`, so an expired backlog is
  spread over steps rather than dumped in one). `0.0` (the default) disables
  it. Set it below the reuse interval the workload has to beat -- previous
  turn's end to the next turn's *scheduling*, which includes the
  waiting-queue delay. Sensor: `emitted_overdue` against `emitted` says how
  much of the traffic the deadline rather than the window released.
- **Horizon calibration.** The default of 2.5 scheduler steps was selected
  by a fine sweep over 2.0-8.0 on two opposing Qwen3-8B/H200 workloads as
  the measured compromise between eviction loss (pushes the horizon up) and
  store filtering (pushes it down). It is a calibrated default, not a
  universal optimum: increase it when eviction loss matters more than
  filtering, decrease it when lower-tier write or eviction pressure is the
  limiting cost.
- **Prefix closure**: a due op releases the request's ops from the front
  through the last due one; a data-loss drop (hash mismatch) drops from the
  first lost op through the tail and blacklists the request's later chunks;
  the intact stored prefix stays pending and stays valuable.
- Cross-request drain order = min due rank ascending; `max_drain_per_step`
  bounds the per-step D2H burst in operations. The cap may split a request's
  due segment but only ever emits a front slice of it, so within-request
  prefix order is preserved. **Sizing**: a prefilling request admits one op
  per step, and a request with a batch in flight is skipped until its
  receipt, so the cap must sit above the concurrent prefill admission rate
  or the queue can never work off a backlog and buffered ops are lost to
  eviction instead of stored -- a cap near 1 is a steady-state loss setting,
  not a burst-shaping one. There is no static validation (the break-even
  depends on runtime prefill concurrency). The sensor is `throttled_drains`,
  counting drains that held a due op back -- a lower bound, since candidates
  the loop never reached are uncounted. Read it against `dropped_evicted`:
  both nonzero cumulatively separates a cap below the admission rate from one
  merely delaying a burst. They land on different steps (the cap builds the
  backlog, a later step's allocation kills it), so they are not expected in
  the same drain.
- **Idle consequences**: receipts travel in worker metadata, which only
  flows on token-producing steps. Pins, in-flight sessions, and finished
  requests whose ops never come due all settle on the next activity --
  nothing leaks permanently, by design. An engine that is not stepping runs
  no drains at all.

## Scheduler-path complexity

A dedicated pending-operation owner maintains the per-request lists together
with their admission order, block-to-request reverse index, and per-request
block reference counts; admission, replacement, and departure update primary
storage and every derived index through one atomic API. A production drain
therefore walks only the bounded free-queue window and the requests
represented in that window or touched by this step's allocations: cost
proportional to the pressure window and drain cap, not total pending queue
depth. The pure-policy tests retain an `allocated_block_ids=None`
compatibility path that performs a full validation pass.

Request lifecycle is not policy state. The controller registry owns request
phase, epoch, and submitted batches; the policy receives blocked request ids
as a drain input and retains only prefix validity, a consequence of its own
store decisions. `release_request`, `drop_request`, and `discard_for_reuse`
clear that non-pending state at controller-defined epoch boundaries, so
completed request ids do not accumulate.

## Observability

`stats()` returns cumulative counters. `dropped_evicted` is the quality
sensor (data lost before we drained -- the horizon is too tight);
`dropped_evicted_tokens` weighs the same losses by token range, separating a
tail of short suffixes from a few destroyed full-length chains.
`emitted_deferral_drains / emitted` is the mean deferral in drains -- the
direct measure of what the policy buys; `dropped_deferral_drains /
dropped_evicted` separates losses that waited too long (addressable by a
wider danger window) from ones that died young (addressable by nothing).
`emitted / admitted` is store precision's denominator; `emitted_overdue` and
`throttled_drains` count events, not operations. Four counters
(`drain_steps`, `free_queue_blocks_read`, `requests_validated`,
`blocks_validated`) measure the decision loop's own cost: divided by
`drain_steps` they give the mean free-queue depth walked and block-hash
comparisons made per step -- the quantities that turn a policy which saves
prefill time into one that spends more decode time than it saves.

The counters surface in the scheduler process log, not in vLLM's
`get_kv_connector_stats` plumbing (that hook is polled worker-side, where the
policy does not live). Three hooks, all on the pending-store facade:

- each drain that dropped ops logs one aggregate INFO line, naming at most 8
  ops and counting the rest; per-op detail logs at DEBUG. The later chunks a
  broken request keeps producing are rejected at admission and log at DEBUG
  only -- their cause was already reported.
- every drain whose counters changed re-logs the whole ledger as one
  greppable `key=value` line, throttled to one per 5 s. `pending` is the
  custody depth at the same instant, which closes the line as an equation
  over exactly five outcome counters:

  ```
  admitted == pending + emitted + dropped_evicted
              + dropped_on_request_drop + dropped_failed_store
              + dropped_id_reuse
  ```

  `rejected_unhashed` and `rejected_prefix_broken` stay out (those ops are
  turned away before `admitted`), and token- or event-valued counters
  (`dropped_evicted_tokens`, `*_deferral_drains`, `emitted_overdue`,
  `throttled_drains`) are weights beside the equation, not terms of it.
- connector `shutdown()` calls `log_final_stats()`, which emits the exact
  final ledger (`Lazy offload final counters: ...`). Best-effort: `vllm
  serve` under SIGINT can beat scheduler shutdown to it -- that is why the
  periodic line exists. A log reader takes the last line matching
  `Lazy offload (final )?counters:`.

Tests: `tests/v1/test_lazy_offload_eviction_aware.py` (pure, no vLLM).
