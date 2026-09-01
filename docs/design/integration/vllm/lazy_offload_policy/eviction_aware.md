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

  Laziness is contractual, not an optimisation: the walk runs on the
  scheduler's critical path once per step and the full queue is O(free
  blocks). The drain consumes the iterator through `_FreeQueueWindow`, which
  opens at `danger_depth` and widens only by the blocks an emission pinned out
  of the queue (see pin cascade), so a step reads exactly the ranks its
  decisions compare -- nothing at danger depth 0. `is_free` exists so pin
  accounting asks the pool, not the window: a pin deeper than the window still
  shifts the queue.
- **`PendingStoreOp`** -- one deferred store: opaque `store_metadata` (the
  ready `LMCacheMPRequestMetadata`), the covered blocks' hash snapshot taken
  at admission, the op's token range, and its admission drain and timestamp.
- **`EvictionAwareStoreQueue`** -- the policy object, one per connector.

## Per-step protocol (policy-caller obligations)

`LazyOffloadManager` is the production caller; the connector only forwards
lifecycle events to it.

1. Route each `GetStoreMetadata` result to `add(meta, block_hashes, epoch)`
   instead of the step metadata. The queue takes custody or drops the
   operation itself. Two are dropped here and logged:
   - A **hash-less covered block**. Its later eviction is undetectable
     (evicted-and-reallocated also reads `None`), so storing it risks filing
     wrong bytes under a content-addressed key. Hybrid-attention models
     (sliding window, mamba) put hash-less null blocks into block tables when
     vLLM's own prefix cache serves a prefix whose out-of-window positions
     hold no KV. The tracker has already advanced past the skipped range, so
     the queue blacklists the request and drops its later chunks too.
   - A **broken prefix**: an earlier chunk of the request was dropped, so
     this one would be unreachable on retrieval.
2. Once per step call `drain(signals)` with that step's `DrainSignals`.
   Pending ops of finished requests wait out their eviction clock -- that is
   the point of lazy offload. Validation is incremental: the queue keeps a
   block-to-request reverse index and revalidates only requests touched by
   this step's allocations or represented in the free-queue window. Requests
   blocked by a submitted batch are skipped until their receipt.
3. For every item in `LazyOffloadDrain.items` (already ordered): pin
   (`touch`) its blocks, coalesce each request's released ops into one store
   op, register the submitted batch and its epoch, and put it into this
   step's connector metadata. `emptied_request_ids` is only a buffer
   transition -- teardown additionally requires the registry to say the
   request is finished with no submitted batch.
4. On a store-completion receipt, complete the batch and unpin. End the
   session only when the registry says the request is finished and the policy
   has nothing pending.
5. On `request_finished`, record `FINISHED` in the registry; end immediately
   only under the same predicate, otherwise a later drain or receipt applies
   it.
6. On preemption tracker reset, advance the epoch and call `drop_request(id)`:
   buffered ops and prefix-validity state are discarded; an already submitted
   batch stays registered and blocks new emission until its receipt. An abort
   is **not** a drop: its buffered ops remain storable until drained.
7. On a failed receipt, compare the batch epoch with the current request
   epoch. A current-epoch failure calls `mark_store_failed(id)`, which drops
   held-back ops and marks the prefix broken; an old-epoch failure never
   enters the policy. Both paths complete the receipt and unpin.
8. On request-id reuse, detect the `FINISHED` predecessor in the registry,
   advance the epoch, and call `discard_for_reuse(id)`. With a submitted batch
   outstanding, the id-keyed session spans both epochs and ends once through
   the successor's lifecycle. (vLLM's HTTP layer randomizes external ids, but
   direct engine callers and `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` reach
   this path.)

**Prerequisite for 1**: the connector must record vLLM's prefix-cache hit in
the tracker even when the LMCache lookup misses. A lazy follower over a hot
APC-shared prefix always misses the lookup (the predecessor's ops are
buffered, not stored); without the vllm-hit share it never reaches `add` and
stores nothing at all. The recording is mode-independent by design: in eager
mode an APC-hit request whose lookup misses now stores its full prefix at once
instead of accumulating it over decode steps, and `cached_token_stats` reports
the true vLLM hit instead of 0.

## Decision rule

- **Danger depth** = `ceil(max(EMA(gross allocation/step), next-step
  feedforward) x horizon_steps)`; below half a block over the horizon it is 0
  (a decayed EMA must not hold the depth at 1 forever). Free-queue *position*
  alone is never a trigger -- an idle engine never drains.
- An op is **due** when any covered block's rank < danger depth. Blocks not in
  the free queue (in use / resurrected) are not at risk.
- **Pin cascade.** Emitting a segment pins its blocks out of the free queue,
  moving every block behind them toward the head before the next step's
  allocation runs. Within one drain, each later candidate is therefore tested
  against `danger_depth` plus the unique emitted blocks that were in the queue
  so far (an in-use block does not leave the queue; a shared block counts
  once), and the window widens by the same amount to reveal the next
  candidates. The first emission still needs a plain danger-depth hit, so the
  extension never opens the gate on an idle system; the alternation terminates
  because each round either emits (the cap is finite) or finds nothing due.
  Dropped (unpinned) segments do not extend the shift.
- **Deferral deadline (`max_deferral_seconds`)**: the danger window is a
  *spatial* signal -- how close a block sits to the free-queue head -- which
  answers when the block dies on the GPU, not when the conversation comes
  back. When the reuse interval is the shorter clock the window emits after
  the entry was already needed: the store lands, too late to be found. A
  request whose oldest pending op has waited longer than this bound is due
  regardless of rank, is decided before the window-driven candidates, and
  releases its whole surviving front (still capped by `max_drain_per_step`).
  `0.0`, the default, disables it.
- **Horizon calibration.** The default 2.5 steps came from a sweep over
  2.0-8.0 on two opposing Qwen3-8B/H200 workloads, balancing eviction loss
  (pushes it up) against store filtering (pushes it down). Calibrated, not
  universal.
- **Prefix closure**: a due op releases the request's ops from the front
  through the last due one; a data-loss drop (hash mismatch) drops from the
  first lost op through the tail and blacklists the request's later chunks;
  the intact stored prefix stays pending and stays valuable.
- Cross-request drain order = min due rank ascending; `max_drain_per_step`
  bounds the per-step D2H burst in operations. The cap may split a request's
  due segment but only ever emits a front slice of it, so within-request
  prefix order is preserved. Sized below the concurrent prefill admission rate
  it becomes a steady-state loss setting rather than burst shaping;
  `configuration.rst` states the rule and `throttled_drains` is the sensor,
  read against `dropped_evicted`.
- **Idle consequences**: receipts travel in worker metadata, which only flows
  on token-producing steps. Pins, in-flight sessions, and finished requests
  whose ops never come due all settle on the next activity -- nothing leaks
  permanently. An engine that is not stepping runs no drains at all.

## Scheduler-path complexity

`_PendingOperations` owns the per-request lists and the block-to-request
reverse index together; admission, replacement, and departure update both
through one atomic API. A drain therefore walks only the bounded free-queue
window and the requests represented in it or touched by this step's
allocations: cost proportional to the pressure window and the drain cap, not
to total pending depth.

Request lifecycle is not policy state. The controller registry owns request
phase, epoch, and submitted batches; the policy receives blocked request ids
as a drain input and retains only prefix validity, a consequence of its own
store decisions. `release_request`, `drop_request`, and `discard_for_reuse`
clear that non-pending state at controller-defined epoch boundaries, so
completed request ids do not accumulate.

## Observability

Counters are cumulative and surface in the scheduler process log, not in
vLLM's `get_kv_connector_stats` plumbing (that hook is polled worker-side,
where the policy does not live). `dropped_evicted` is the quality sensor --
data lost before we drained, meaning the horizon is too tight -- weighed by
token range in `dropped_evicted_tokens`. `emitted_deferral_drains / emitted`
is the mean deferral in drains, the direct measure of what the policy buys,
and `emitted_overdue` how much of that the deadline rather than the window
released. `drain_steps` is the denominator for both.

Two log hooks:

- A drain that dropped ops logs one aggregate INFO line naming up to 8 of the
  requests, and a drain whose counters changed re-logs the whole ledger as one
  greppable `key=value` line, throttled to one per 5 s. `pending` is the
  custody depth at the same instant, which closes the line as an equation over
  exactly five outcome counters:

  ```
  admitted == pending + emitted + dropped_evicted
              + dropped_on_request_drop + dropped_failed_store
              + dropped_id_reuse
  ```

  `rejected_unhashed` and `rejected_prefix_broken` stay out (those ops are
  turned away before `admitted`); token- and event-valued counters
  (`dropped_evicted_tokens`, `*_deferral_drains`, `emitted_overdue`,
  `throttled_drains`) are weights beside the equation, not terms of it.
- Connector `shutdown()` calls `log_final_stats()`, which emits the exact
  final ledger. Best-effort: `vllm serve` under SIGINT can beat scheduler
  shutdown to it -- that is why the periodic line exists. A log reader takes
  the last line matching `Lazy offload (final )?counters:`.
