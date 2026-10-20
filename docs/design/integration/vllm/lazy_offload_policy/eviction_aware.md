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

- The **`BlockPool`** bound via the vLLM `bind_gpu_block_pool` hook is only
  ever read: the free queue's links from the eviction head (a block's
  position is its LRU rank, rank 0 = next victim) and block hashes. The walk
  runs on the scheduler's critical path once per step, so it follows the
  queue's links and stops at the danger depth instead of materialising the
  whole queue -- nothing at danger depth 0.
- **`PendingStoreOp`** -- one deferred store: opaque `store_metadata` (the
  ready `LMCacheMPRequestMetadata`), the covered blocks' hash snapshot taken
  at admission, and its admission timestamp. Its request id and token range
  are not copied out: the buffer is keyed by request id and holds each
  request's ops in token order.
- **`EvictionAwareStoreQueue`** -- the policy object, one per connector.

## Per-step protocol (policy-caller obligations)

`LazyOffloadManager` is the production caller; the connector only forwards
lifecycle events to it.

1. Route each `GetStoreMetadata` result to `add(meta, block_hashes)`
   instead of the step metadata. The queue takes custody or drops the
   operation itself. Two are dropped here and logged:
   - A **hash-less covered block**. Its later eviction is undetectable
     (evicted-and-reallocated also reads `None`), so storing it risks filing
     wrong bytes under a content-addressed key. The tracker has already
     advanced past the skipped range, so the queue blacklists the request and
     drops its later chunks too. The null block is not offered: the manager
     leaves it out of `block_hashes`. Hybrid models put it at positions that
     hold no data -- every interior mamba block in `--mamba-cache-mode align`,
     out-of-window sliding-window blocks -- and it is never recycled, so it
     has nothing to validate. Counting it rejected every request of a
     mamba-hybrid model (Qwen3.5-122B-A10B: 2 of ~5100 operations admitted).
   - A **broken prefix**: an earlier chunk of the request was dropped, so
     this one would be unreachable on retrieval.
2. Once per step call `drain(signals)` with that step's `DrainSignals`.
   Pending ops of finished requests wait out their eviction clock -- that is
   the point of lazy offload. Every drain revalidates every pending request's
   hash snapshots, including those of requests blocked by a submitted batch.
   A blocked request is never emitted -- the worker accepts one store future
   per request -- but the ops it would release now (its due front segment)
   are returned in
   `LazyOffloadDrain.holds`: the manager pins those blocks so they cannot be
   recycled while the batch is in flight, and once the receipt arrives the
   held ops are due and their batch takes the pins over. Its other ops stay
   buffered and unpinned: the window is a bound, so they cannot be recycled
   before a later drain sees them. Without holds, a slow store let the
   blocks behind it go: on Qwen3.5-122B-A10B at N=150 with
   `prefix-cache-retention-interval` 4240, 49% of admitted ops were dropped,
   91% of them while their request was blocked. Holding only the front
   segment then still lost 27%, all to allocations the estimated window
   missed, so holds covered every op of a blocked request until the window
   became a bound and admissions and resumed loads got their own
   protection. Holding everything held 4843 of 5790 ops (84%), and each
   was stored on unblock, at risk or not, which made the hold, not
   eviction, decide when most ops were stored.
   The danger window does not cover allocations by requests that are not
   yet running: an admission with a long prefix hit allocates its whole
   prefix in one step, and vLLM evicts while allocating, before the step's
   drain. vLLM asks the connector for the external hit right before that
   allocation, so `get_num_new_matched_tokens` reports the admission to
   `LazyOffloadManager.on_admission`, which bounds its tokens: hit + one
   chunk + draft lookahead, or the hit alone for an asynchronous load,
   whose rest `on_loads_finished` bounds when the load completes. Per KV
   group the blocks are those the tokens span, plus a partial block at the
   start, one for a partial hit's copy-on-write block or a Mamba
   checkpoint, and `L` Mamba speculative state blocks (vLLM reserves one
   per draft token on a request's first allocation in align mode). An
   align-mode Mamba group is the exception: it nulls every block the
   tokens span but the state block, and `MambaManager` caps an allocation
   at `1 + extra + L` blocks, so that is its count whatever the hit's
   length. The `L` only applies to a first allocation: a resumed load's
   admission already allocated for its hit, so `on_loads_finished` counts
   `1 + extra` per Mamba group. The local prefix hit adds its own block count: vLLM touches
   those blocks before allocating, which takes the free ones out of the
   queue and brings every block behind them closer to the head. An
   align-mode Mamba hit is null blocks but its last, so it adds at most
   one, and an external hit one more: `allocate_external_computed_blocks`
   gives it a state block before the capped allocation. Leaving that block
   out lost 6 of 5285 ops in a c150 run. Counting Mamba groups by the tokens spanned made a 100k-token hit
   on Qwen3.5-122B-A10B (3 Mamba groups of 4) count about 116 blocks for
   at most about 47, and every admission past the free blocks outside
   buffered ops is squeezed.
   `protect(num_blocks)` walks the free queue until it has counted that
   many blocks outside buffered ops. The ops met move to the
   free-queue **tail** (`touch` then `free_blocks`, hash intact) and stay
   buffered: at the tail they are as far from eviction as any block, so
   the drain stores them once they near the head again. They are not
   pinned: a pin shrinks the free queue the admission allocates from, and
   when the rest no longer fits vLLM refuses the admission and every
   waiting request behind it for that step. Only when the walk reaches the
   tail before counting `num_blocks` -- the allocation would take the tail
   too -- is every buffered op held (pinned) instead, counted as
   `squeezed_admissions`: deferring the admission is cheaper than losing
   the op and, with it, the request's prefix chain. Every op, not only
   the ones met: a block freed between the walk and the allocation joins
   a tail the allocation already reaches. A resumed load allocates after
   the running requests, and those free blocks first -- align-mode Mamba
   frees a running request's older state blocks, hash intact, on each
   step. Pinning a block its request still owns costs no free block.
   A request in `preempted_request_ids` releases its whole buffer: vLLM
   freed its blocks, and once rescheduled its tracker restarts from token
   zero, which discards whatever is still buffered (step 6).
3. For every item in `LazyOffloadDrain.items` (already ordered): pin
   (`touch`) its blocks, coalesce each request's released ops into one store
   op, register the submitted batch, and put it into this
   step's connector metadata. `emptied_request_ids` is only a buffer
   transition -- teardown additionally requires the registry to say the
   request is finished with no submitted batch.
4. On a store-completion receipt, complete the batch and unpin. A block
   that sat in the free queue when it was pinned goes back to the queue
   **head**, in order: `free_blocks` would append it as most recently used,
   while eviction order had it next, so every lazy store would otherwise
   refresh the GPU prefix cache's LRU. A block a request still owns is left
   to vLLM. End the session only when the registry says the request is
   finished and the policy has nothing pending.
5. On `request_finished`, record `FINISHED` in the registry; end immediately
   only under the same predicate, otherwise a later drain or receipt applies
   it.
6. On preemption tracker reset, call `drop_request(id)`: buffered ops and
   prefix-validity state are discarded, and the manager unpins held ops (as
   it does when `mark_store_failed` or `discard_for_reuse` drops them); an
   already submitted batch stays
   registered -- now marked orphaned -- and blocks new emission until its
   receipt. An abort is **not** a drop: its buffered ops remain storable
   until drained.
7. On a failed receipt, check whether the submitted batch is orphaned. A
   non-orphaned failure calls `mark_store_failed(id)`, which drops held-back
   ops and marks the prefix broken; an orphaned batch's failure never enters
   the policy, because it cannot break the prefix of the request generation
   now using the id. Both paths complete the receipt and unpin.
8. On request-id reuse, detect the `FINISHED` predecessor in the registry and
   call `discard_for_reuse(id)`. With a submitted batch outstanding, that
   batch is marked orphaned and the id-keyed session spans both generations,
   ending once through the successor's lifecycle. (vLLM's HTTP layer randomizes external ids, but
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

- **Danger depth** = `DrainSignals.next_step_blocks`, a bound, not an
  estimate, on the blocks the running requests can allocate before the next
  drain. With `c` a request's tokens computed after this step and `L` the
  draft lookahead, the next step schedules it at most
  `min(remaining, B)` tokens while prefilling or `1 + L` while decoding,
  and all requests together at most `B` (`max_num_scheduled_tokens`,
  defaulting to `max_num_batched_tokens`); its slots run `L` past what it
  schedules. Per KV group, the first new block of a request costs the
  tokens that fill the blocks its table holds, less `L` (at least 1); each
  further block costs `bs` tokens. The table length, tracked per group from
  the scheduler's block ids, already covers the block the last step's
  lookahead reached, which `ceil(c / bs)` would count again. Two vLLM rules for align-mode Mamba hybrids
  tighten that. A prefill chunk short of the prompt's end ends on the hash
  grid (`_mamba_block_aligned_split`, when a block fits in a chunk), so a
  prefilling crossing costs up to the next grid point, or the rest of the
  prompt if that is less. A Mamba group takes one state block per step,
  plus a checkpoint block if the spec has them and a copy-on-write block
  after a partial hit (`1 + partial hit + checkpoint`), whatever the chunk
  covers, and ignores `L`. The connector derives both from the KV-cache
  config (`get_prefill_alignment`, `get_mamba_step_extra_blocks`). Without
  them, every prefilling request at a block boundary counted 2 blocks for
  1 token per group, and hold7's bound came out 3.83x the measured
  allocation. When every crossing yields as many blocks (the common case:
  one each, or `1 + extra` in a Mamba group), the bound is the most
  crossings whose costs, cheapest first, fit `B` whole: a crossing is paid
  in full or not at all. Otherwise it is the fractional knapsack over those
  blocks with budget `B`. Either is at least what any schedule allocates.
  On a replay of the hold7 workload (simulated, Qwen3.5-122B-A10B, N=104),
  the table length and whole crossings bring the bound from 1.47x to 1.01x
  the allocation, with no step over it. It is the token budget that keeps it tight: summing each
  request's own reach instead counts a full chunk for every prefilling
  request, several times what one step can schedule. Spec-decode
  rejections only lower `c`. Free-queue *position* alone is never a
  trigger -- an idle engine never drains.
- The bound excludes the two allocations the scheduler makes for requests
  that are not yet running, which the manager protects right before they
  happen (step 2): admissions, via `get_num_new_matched_tokens`, and
  asynchronous loads resuming, via `update_connector_output`'s
  `finished_recving` -- vLLM schedules a resumed load without asking for a
  hit again. vLLM allocates for the running requests before either, so the
  resume notice protects `next_step_blocks` plus the resumed requests'
  chunks. vLLM schedules the resumed request only when its chunk fits,
  which with a full cache can be steps later; meanwhile the head moves
  past the blocks the notice moved to the tail. Until a step schedules
  it, each drain adds its chunk to the danger depth (`resuming_blocks`,
  kept out of the bound and its `bound_exceeded` check). Protecting only
  at the notice lost 18 ops in one event on Qwen3.5-122B-A10B, N=150:
  one op's three Mamba state blocks were recycled by a new allocation
  while no admission exceeded its estimate and no step its bound. An admission with an asynchronous load allocates for its hit
  alone. The EMA-and-burst-floor estimate this replaced lost 31 of 5790
  ops on Qwen3.5-122B-A10B, N=150, most of them to steps that allocated
  more than it predicted.
- An op is **due** when the window walk below meets any covered block. Blocks
  not in the free queue (in use / resurrected) are not at risk.
- Emitting a segment pins its blocks out of the free queue, moving every
  block behind them toward the head. The drain accounts for this
  shift by walking until it has counted `danger depth` blocks outside
  buffered ops: every buffered block met is due, so it leaves the queue
  (or moves to the tail) and the blocks behind it are what the next step
  can reach. Due ops past `max_drain_per_step` move to the free-queue
  tail, out of the next step's reach, and stay buffered like protected
  ops -- unless the walk reached the tail before counting the danger
  depth (`squeezed_drains`): the next step may then take the whole queue,
  and every block freed before it allocates, so every buffered op comes
  due and those past the budget are held (pinned) instead, the same trade
  as a squeezed admission. In a c150 benchmark run the ops still lost
  matched this path: only Mamba groups, requests still running, and
  every loss during a burst of squeezed walks.
- **Deferral deadline (`max_deferral_seconds`)**: the danger window is a
  *spatial* signal -- how close a block sits to the free-queue head -- which
  answers when the block dies on the GPU, not when the conversation comes
  back. When the reuse interval is the shorter clock the window emits after
  the entry was already needed: the store lands, too late to be found. A
  request whose oldest pending op has waited longer than this bound is due
  even when no block of it sits in the window, and then releases its whole
  surviving front (still capped by `max_drain_per_step`). It is served after
  every request that does hold a block in the window, so an expired backlog
  cannot spend the budget a block about to be recycled needs. A request past
  both is ranked and sized by the window: it releases the due front segment,
  which always holds the op that made it overdue, and the ops behind that keep
  their own admission clocks and come due on their own deadlines. `0.0`, the
  default, disables it.
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
  prefix order is preserved. Sized below the concurrent prefill admission
  rate it becomes a steady-state loss setting rather than burst shaping;
  `configuration.rst` states the rule.
- **Idle consequences**: receipts travel in worker metadata, which flows on
  every step, including steps that schedule no tokens. Those steps drain as
  well, which a squeezed admission needs: its pins can leave no waiting
  request room to run, and with nothing running only storing the pinned ops
  frees the blocks. hold8 (Qwen3.5-122B-A10B, N=150) deadlocked there after
  19 minutes when zero-token steps returned no actions. Pins, in-flight
  sessions, and finished requests whose ops never come due all settle on the
  next activity -- nothing leaks permanently. An engine that is not stepping
  runs no drains at all.

## Scheduler-path complexity

A drain walks the free queue to the danger depth and re-checks every pending
request's hash snapshots: cost proportional to the pressure window plus the
buffered blocks, both small -- pending depth is bounded by concurrent
requests, and each check is a dict lookup and a comparison.

Request lifecycle is not policy state. The manager's registry owns request
phase and submitted batches; the policy receives blocked request ids as a
drain input and retains only prefix validity, a consequence of its own store
decisions. `release_request`, `drop_request`, and `discard_for_reuse` clear
that non-pending state at manager-defined lifecycle boundaries, so
completed request ids do not accumulate.

## Observability

Counters are cumulative and surface in the scheduler process log, not in
vLLM's `get_kv_connector_stats` plumbing (that hook is polled worker-side,
where the policy does not live). `dropped_evicted` is the quality sensor --
data lost before we drained, meaning an allocation escaped both the
running-request bound and the admission protection. `bound_exceeded`
checks the first of those directly: the manager counts the blocks each
step handed to requests already running (new ids, less null blocks and
blocks vLLM only moved within the table) and the policy compares them with
the previous drain's bound, logging each excess. Nonzero means the bound
is wrong for this model or vLLM version. `bound_blocks` over
`bound_allocated_blocks` measures the other direction: the bound's slack,
how many blocks it reserves per block actually allocated, and so how much
earlier than necessary it drains. And
`emitted_overdue` is how much of `emitted` the deadline rather than the
window released.

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
  turned away before `admitted`), and `emitted_overdue` is a weight beside
  the equation, not a term of it. `emitted` means handed to the manager, not
  submitted: the manager re-reads the hashes before pinning and can still
  drop an op there. That drop is a WARNING line and no counter, and for this
  policy it is unreachable, since `drain()` checked the same hashes
  synchronously a moment earlier.
- Connector `shutdown()` calls `log_final_stats()`, which emits the exact
  final ledger. Best-effort: `vllm serve` under SIGINT can beat scheduler
  shutdown to it -- that is why the periodic line exists. A log reader takes
  the last line matching `Lazy offload (final )?counters:`.
