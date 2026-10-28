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
  at admission. Its request id and token range
  are not copied out: the buffer is keyed by request id and holds each
  request's ops in token order.
- **`EvictionAwareStoreQueue`** -- the policy object, one per connector.

## Lazy offload covers the blocks vLLM retains

Lazy offload stores a block at the latest moment before vLLM reuses it.
vLLM frees hash-less blocks to the head of the free queue, reused first,
and hashed blocks to the tail in LRU order (`BlockPool.free_blocks`,
vLLM `v1/core/block_pool.py:777-806`); an allocation takes blocks from the
head (`get_new_blocks`, `:684`), and a touch removes a block from the queue
(`touch`, `:766-767`).

- A **hashed block has slack**: it is reused only after the blocks ahead of
  it, so the policy tracks it as it moves toward the head and stores it
  before reuse.
- A **hash-less block has none**: the next allocation takes it. Pinning it
  to store it later would make vLLM take the next free block instead,
  evicting a hashed, still reusable block early: a reusable block traded
  for one vLLM does not intend to reuse. So lazy offload covers only hashed
  blocks, and the manager replaces a hash-less block by the null block when
  it buffers an op (`add_store_candidate`). It never enters a hash snapshot
  and never breaks a request's chain, which is defined over hashed blocks.
- The **hashed blocks are exactly those vLLM retains** past the step: it
  decides and hashes them in `allocate_slots` (`kv_cache_manager.py:607-616`)
  before the connector builds its metadata. Under sparse retention
  (`--prefix-cache-retention-interval 0`) the hash-less non-null blocks in
  a store op are the align-mode recurrent states it does not keep
  (`single_type_kv_cache_manager.py:1572-1630`); `remove_skipped_blocks`
  frees each two schedules after the step that wrote it (`:1653-1681`,
  called from `allocate_slots`, `kv_cache_manager.py:555`), in the same
  call that then allocates. Full attention blocks in an op are always
  hashed: an op covers whole computed blocks, and lazy offload requires
  prefix caching. Sliding-window blocks out of the window are already null.
- **Hash-less blocks ahead of a victim still count** for the window. They
  occupy free-queue positions and absorb allocations first, so the window
  walk counts every free block outside buffered ops toward the danger depth
  `D`, hash-less ones included.

The invariant (no missed store): L0 (vLLM's prefix cache) plus L1 covers
every block vLLM retains, that is, every hashed block. A hashed block
leaves L0 only through an allocation, and the policy stores it before an
allocation within the danger depth `D` can reach it (see below for what
`D` does not cover).

### Known limits

This comes from the separate Mamba null-page commit bug (a recurrent chunk
whose last block is null commits null page 0 as its state):

- When a prompt's kept state lands mid-chunk (prompt length modulo the
  chunk in `1..block_size-1` with chunk = 2 blocks), the chunk ending at
  the hit point carries that state in its first block and null in its
  last; storing it hits the bug.

## Per-step protocol (policy-caller obligations)

`LazyOffloadManager` is the production caller; the connector only forwards
lifecycle events to it.

1. Route each `GetStoreMetadata` result to `add(meta, block_hashes)`
   instead of the step metadata. The queue takes custody or drops the
   operation itself. `block_hashes` holds only hashed blocks: the manager
   has replaced each hash-less block by the null block (see "Lazy offload
   covers the blocks vLLM retains"), and the null block, which hybrid
   models put at positions that hold no data, is never recycled. One case
   is dropped here and logged: a **broken prefix**, where an earlier chunk
   of the request was dropped, so this one would be unreachable on
   retrieval.
2. Once per step call `drain(signals)` with that step's `DrainSignals`.
   Pending ops of finished requests wait out their eviction clock -- that is
   the point of lazy offload. Every drain revalidates every pending request's
   hash snapshots, and drops the suffix from the first op whose blocks were
   recycled; requests blocked by a submitted batch are then left pending
   until their receipt.
3. For every item in `LazyOffloadDrain.items` (already ordered): pin
   (`touch`) its blocks, coalesce each request's released ops into one store
   op, register the submitted batch, and put it into this
   step's connector metadata. `emptied_request_ids` is only a buffer
   transition -- teardown additionally requires the registry to say the
   request is finished with no submitted batch.
4. On a store-completion receipt, complete the batch and unpin. End the
   session only when the registry says the request is finished and the policy
   has nothing pending.
5. On `request_finished`, record `FINISHED` in the registry; end immediately
   only under the same predicate, otherwise a later drain or receipt applies
   it.
6. On preemption tracker reset, call `drop_request(id)`: buffered ops and
   prefix-validity state are discarded; an already submitted batch stays
   registered -- now marked orphaned -- and blocks new emission until its
   receipt. An abort is **not** a drop: its buffered ops remain storable
   until drained.
7. On a failed receipt, mark the in-flight batch failed and wait for the
   rest of its receipts: another worker may still be reading its blocks.
   Once the batch settles, unpin it; a failed batch that is not orphaned
   goes back to the policy with `requeue_failed(item)`, ahead of the
   request's later ops, and unpinning frees its blocks to the free-queue
   tail. The data is still on the GPU, so the request's prefix chain stays
   whole. The batch failed because a worker failed or the server was
   unhealthy; the retry waits for the window like any buffered op. Keys
   already resident count as stored, so retrying a partly stored batch is
   safe. An orphaned batch is never retried: the request generation now
   using the id restarted from token zero or is a different request. After
   three failed batches in a row the manager gives up instead:
   `mark_store_failed` drops the request's ops (`dropped_failed_store`) and
   breaks its prefix chain.
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

- **Running-request bound** = `DrainSignals.next_step_blocks`, a bound,
  not an estimate, on the blocks the running requests can allocate before
  the next drain. With `c` a request's tokens computed after this step and
  `L` the draft lookahead, the next step schedules it at most
  `min(remaining, B)` tokens while prefilling or `1 + L` while decoding,
  and all requests together at most `B` (`max_num_scheduled_tokens`,
  defaulting to `max_num_batched_tokens`); its slots run `L` past what it
  schedules. Per KV group, the first new block of a request costs the
  tokens that fill the blocks its table holds, less `L` (at least 1); each
  further block costs `bs` tokens. The table length, tracked per group from
  the scheduler's block ids, already covers the block the last step's
  lookahead reached, which `ceil(c / bs)` would count again. Two vLLM
  rules for align-mode Mamba hybrids tighten that. A prefill chunk short
  of the prompt's end ends on the hash grid (`_mamba_block_aligned_split`,
  when a block fits in a chunk), so a prefilling crossing costs up to the
  next grid point, or the rest of the prompt if that is less. A Mamba
  group takes one state block per step, plus a checkpoint block if the
  spec has them and a copy-on-write block after a partial hit
  (`1 + partial hit + checkpoint`), whatever the chunk covers, and ignores
  `L`. The connector derives both from the KV-cache config
  (`get_prefill_alignment`, `get_mamba_step_extra_blocks`). Without them,
  every prefilling request at a block boundary counted 2 blocks for 1
  token per group, and on Qwen3.5-122B-A10B at N=104 the bound came out
  3.83x the measured allocation. The bound is the fractional knapsack over
  every crossing with budget `B`: crossings sorted by tokens per block,
  taken whole while they fit, and the one the budget ends in rounded up to
  its full blocks. It ignores that a further crossing needs its request's
  first one, which can only overcount, so it is at least what any schedule
  allocates; the rounding costs at most one crossing per group, on a step
  `B` binds. On a replay of that run (simulated), the table length and the
  token budget bring the bound from 1.47x to 1.01x the allocation, with no
  step over it (measured with crossings paid whole, before the rounded-up
  one). It is the token budget that keeps it tight: summing each request's
  own reach instead counts a full chunk for every prefilling request,
  several times what one step can schedule. Spec-decode rejections only
  lower `c`. Free-queue *position* alone is never a trigger -- an idle
  engine never drains.
- The **danger depth** `D` adds `admission_blocks` to that bound. vLLM
  allocates an admitted request's blocks before the step's drain, so the
  drain keeps enough of the head clear for one admission: a step's first
  admission recycles no buffered block. `admission_blocks` is derived from
  `max_model_len`, not tuned: an admission's local hit plus the slots it
  allocates hold at most `max_model_len + 1 + L` tokens, and splitting them
  between the hit and the allocation costs a group at most one block (two
  for an align-mode Mamba group's hit state blocks). On Qwen3.5-122B-A10B
  (`max_model_len` 262144, `L` 3) it is 86 of 2551 blocks. A second
  admission in the same step, and an asynchronous load resuming without a
  new lookup, can still reach past `D`. The EMA-and-burst-floor estimate
  this replaced (`horizon_steps` x the smoothed allocation rate) lost 31 of
  5790 ops on Qwen3.5-122B-A10B, N=150, most of them to steps that
  allocated more than it predicted.
- An op is **due** when the window walk below meets any covered block. Blocks
  not in the free queue (in use / resurrected) are not at risk.
- Emitting a segment pins its blocks out of the free queue, moving every
  block behind them toward the head. The drain accounts for this shift by
  walking until it has counted `D` blocks outside buffered ops: every
  buffered block met is due, so it leaves the queue and the blocks behind it
  are what the next step can reach. A due op that stays in the queue (its
  request has a batch in flight, or the drain budget is spent) only makes
  the walk reach further.
- **No deadline**: eviction pressure alone decides when an op is stored;
  how long it has waited does not. A time bound would store ops that no
  allocation is about to reach.
- **Prefix closure**: a due op releases the request's ops from the front
  through the last due one; a data-loss drop (hash mismatch) drops from the
  first lost op through the tail and blacklists the request's later chunks;
  the intact stored prefix stays pending and stays valuable.
- Cross-request drain order = min due rank ascending; `MAX_DRAIN_PER_STEP`
  (64) bounds the per-step D2H burst in operations. It is a constant, not a
  setting. The cap may split a request's due segment but only ever emits a
  front slice of it, so within-request prefix order is preserved; due ops
  past it stay buffered for a later drain.
- **Idle consequences**: receipts travel in worker metadata, which only flows
  on token-producing steps. Pins, in-flight sessions, and finished requests
  whose ops never come due all settle on the next activity -- nothing leaks
  permanently. An engine that is not stepping runs no drains at all.

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
where the policy does not live). `bound_exceeded` counts steps whose
running requests allocated more blocks than the previous drain's
`next_step_blocks`, which the bound says cannot happen: nonzero means it is
wrong for this model or vLLM version. `bound_blocks` over
`bound_allocated_blocks` is the bound's overcount, how many blocks it
reserves per one allocated. `dropped_evicted` is the quality sensor --
data lost before we drained, meaning an allocation reached past the
danger depth.

Two log hooks:

- A drain that dropped ops logs one aggregate INFO line naming up to 8 of the
  requests, and a drain whose counters changed re-logs the whole ledger as one
  greppable `key=value` line, throttled to one per 5 s. `pending` is the
  custody depth at the same instant, which closes the line as an equation:

  ```
  admitted + requeued == pending + emitted + dropped_evicted
                         + dropped_on_request_drop + dropped_id_reuse
                         + dropped_failed_store
  ```

  `requeued` counts emitted ops whose store failed and came back to be
  retried; each retry counts in `emitted` again.

  `rejected_prefix_broken` stays out (those ops are
  turned away before `admitted`), and the `bound_*` counters are weights
  beside the equation, not terms of it. `emitted` means handed to the manager, not
  submitted: the manager re-reads the hashes before pinning and can still
  drop an op there. That drop is a WARNING line and no counter, and for this
  policy it is unreachable, since `drain()` checked the same hashes
  synchronously a moment earlier.
- Connector `shutdown()` calls `log_final_stats()`, which emits the exact
  final ledger. Best-effort: `vllm serve` under SIGINT can beat scheduler
  shutdown to it -- that is why the periodic line exists. A log reader takes
  the last line matching `Lazy offload (final )?counters:`.
