# APC-covered lookup skip

## Summary

When the serving engine's prefix cache (vLLM APC) already covers the first
`N` tokens of a request, the MP-server lookup no longer read-locks or
L2-prefetches the LMCache objects for that covered prefix. It instead
presence-checks and LRU-touches them, so they stay warm in the eviction
order without any lock or data-movement cost. Only the uncovered tail
`[N, end)` goes through the normal lock + prefetch path.

The feature is gated by the connector extra-config flag:

```yaml
lmcache.mp.skip_covered_lookup: true   # default: false
```

With the flag off (or against an older server), the wire carries
`covered_chunks = 0` and behavior is bit-for-bit identical to today.

## Implementation status (draft branches)

This doc describes the target design. The `core/skip-apc-pin` and
`core/skip-apc-nonpin` draft branches implement the shared foundation and one
shrink-handling method each, with two deliberate interim simplifications that
avoid a gRPC protobuf change (no `grpc_tools` in the authoring env to
regenerate stubs):

1. **`covered_chunks` is carried in `request_configs`** (key
   `COVERED_CHUNKS_CONFIG_KEY`) rather than as a first-class
   `IPCCacheServerKey` / `IpcCacheServerKey`-proto field. `request_configs` is
   already transmitted (msgpack blob), is not part of cache identity, and is
   only populated when the feature is on -- so the feature-off path is
   byte-identical. A production version should promote it to a proto field.
2. **The `covered_present` store-hole signal is not transmitted yet.** The
   server computes it (for the LRU touch) but the LOOKUP reply stays `None`
   (adding an int reply needs a `LookupResponse` proto field + regen), so
   `LookupResult.stored_tokens` currently equals `hit_tokens`. Until the
   follow-up lands, a covered chunk evicted while covered is re-stored by the
   normal store path once a later request observes it as a miss, rather than
   via the presence probe.

Both simplifications are pure carriage/among-tiers concerns; the skip + touch +
lock-clamp + shrink-handling logic is unaffected. The server realizes the skip
by prefetching only the uncovered sub-range `chunk_hashes[covered_chunks:]` and
offsetting hit counts by `covered_chunks` at status time -- equivalent to the
fold-surgery described below but leaving `_submit_prefix_fold` untouched.

## Motivation

Today the scheduler-side lookup always covers the full prompt from token 0
(`vllm_multi_process_adapter.maybe_submit_lookup_request` builds the key
with `start=0`), even though `get_num_new_matched_tokens` already knows the
vLLM APC hit (`num_computed_tokens`). For the covered prefix this costs,
per request:

- `num_kv_readers` read-lock acquires and releases per object
  (`StorageManager.submit_prefetch_task` → `L1Manager.reserve_read`,
  later `free_lookup_locks` → `finish_read_prefetched`);
- an L2→L1 fetch of covered chunks that are L2-resident, immediately
  deleted as temp objects after the lock release;
- one extra `FREE_LOOKUP_LOCKS` RPC from
  `LMCacheMPConnector.update_state_after_alloc` to unlock
  `[0, num_vllm_hit_tokens)`.

All of it is pure waste: those tokens are never retrieved from LMCache.
The only thing the covered prefix actually needs is (a) an LRU touch so a
busy prefix is not evicted from L1, and (b) presence information so the
store path knows what is already persisted.

## The two hit numbers

The change decouples two meanings that today are conflated in the single
lookup result `ret`:

| Number | Semantics | Used for | Wire |
| --- | --- | --- | --- |
| `hit_chunks` | Prefix fold where covered chunks count as *satisfied by APC* regardless of LMCache presence. Always `>= covered_chunks`. | `need_to_load`, `num_lmcache_hit_tokens`, lock ranges | unchanged: `QUERY_PREFETCH_STATUS -> int` |
| `covered_present_chunks` | Leading-ones of actual L1 presence over `[0, covered_chunks)`, AND-ed across all object groups × kv ranks (no sliding-window credit — conservative). | store skip (`num_stored_tokens`), true-hit stats | new: returned as the `LOOKUP` ack payload (previously `None`, never read) |

The connector reconstructs the true LMCache hit:

```text
stored_chunks = covered_present_chunks              if covered_present_chunks < covered_chunks
              = hit_chunks                          otherwise
```

If any covered chunk is missing from LMCache (`covered_present < covered`),
the contiguous stored prefix ends at the hole, and the store path re-stores
from there — APC-hit blocks are in `tracker.allocated_block_ids`, so the
offload path can save them. This is what prevents *store holes*: without
the presence probe, marking the covered range as stored would leave chunks
permanently missing, capping every future lookup at the hole once the GPU
blocks age out.

## Wire protocol

- `IPCCacheServerKey` gains `covered_chunks: int = field(default=0,
  compare=False)`. Not part of cache identity. msgspec map-encoding gives
  forward compatibility both ways: old payloads decode as 0 on new
  servers; old servers ignore the unknown field.
- The `LOOKUP` reply (previously an ack carrying `None`) now carries
  `covered_present_chunks: int`. Existing clients never read the ack value
  (`_LookupAck` only polls for completion), so old clients are unaffected.
  The SGLang and TensorRT MP adapters likewise call
  `req_client.lookup(...).result(...)` only for completion and discard the
  value; they always send `covered_chunks = 0` and need no changes.
- `QUERY_PREFETCH_STATUS` / `WAIT_PREFETCH_STATUS` are unchanged.
  `QUERY_PREFETCH_LOOKUP_HITS` keeps its early-hits contract but reports
  satisfied semantics (covered chunks count as hit), consistent with the
  final status.
- gRPC transport: `IpcCacheServerKey` in `common.proto` gains
  `covered_chunks`; the empty `LookupResponse` in `lookup_service.proto`
  gains `covered_present_chunks` (proto3 defaults preserve compatibility
  with old peers). ZMQ transport is generic msgspec and needs no changes.

**Version-skew rule:** a new connector with the flag enabled against an old
server would assume no locks exist below `covered_chunks` while the old
server locked the full range — leaking those locks until TTL expiry.
Operators must upgrade servers before enabling the flag. A new connector
receiving a `None` ack treats `covered_present = 0` (over-store, never
corrupt) and logs a version-skew warning once.

## Server-side design

### `LookupModule.lookup` (`lmcache/v1/multiprocess/modules/lookup.py`)

1. Resolve covered object keys (`ipc_key_to_object_keys` over chunks
   `[0, covered_chunks)`, all object groups × kv ranks).
2. Probe presence with the new non-locking `L1Manager.peek_keys` (only
   readable resident objects count — staging objects are invisible, the
   same rule `reserve_read` applies).
3. `touch_l1_keys` on the present covered keys (drives the eviction-policy
   listeners via `on_l1_keys_accessed`).
4. Compute `covered_present_chunks` (leading-ones AND across groups/ranks)
   and return it as the handler result.
5. `session.begin_lookup` additionally records `covered_chunks` for later
   lock-release clamping.
6. Submit the prefetch task with the full key range plus the new
   `PrefetchTaskSpec.covered_prefix_chunks` field; chunk/row indices stay
   absolute so `fold_unfold_grouped` at status time needs no offset math.

The peek + touch is synchronous dict probing — cheap enough for the
BLOCKING handler. `end_session`'s full-range touch is unchanged and
idempotent with the lookup-time touch.

### `StorageManager._submit_prefix_fold` (`lmcache/v1/distributed/storage_manager.py`)

With `covered_prefix_chunks = c0` and `stride = num_object_groups *
world_size`:

- `reserve_read` runs only on `keys[c0 * stride:]`.
- Two bitmaps: `reserved_presence` (locks actually taken) and
  `satisfied = covered_ones | reserved_presence`.
- `fold_unfold_ranked(satisfied, ...)` yields `hit_chunks` (`>= c0` by
  construction) and the retain mask.
- `released = reserved_presence & ~retain` — covered bits can never be
  released because no lock was taken on them.
- The L2 boundary `hit_chunks * stride >= c0 * stride`, so the L2 fetch
  request never includes covered keys.
- `PrefetchHandle` gains `covered_chunks` so hit logging and the
  `MP_LOOKUP_PREFETCH_END` accounting can distinguish satisfied-not-present
  chunks from real L1 hits.

`TrimPolicy.SPARSE` and `PrefetchMode.WARM` ignore the hint (the blend and
warm-prefetch paths build their own specs and keep it 0).

**Full-coverage degenerate case:** when `covered_chunks >= len(chunk_hashes)`
(APC covers every lookup-able chunk), no special case is needed — the split
degenerates naturally: `reserve_read` runs on an empty key list, `satisfied`
is all ones, `hit_chunks = num_chunks`, the L2 boundary equals the end so
nothing is submitted to L2, and the prefetch job is still registered so
status polling works unchanged. Implementations must not add an early-exit
branch for this; a test pins the behavior.

### Lock-release clamping (correctness-critical)

L1 read locks are anonymous refcounts shared across requests; releasing a
lock that was never taken silently consumes a concurrent reader's lock.
With the covered range unlocked, every release path must clamp:

- `resolve_prefetched_obj_keys` gains `covered_chunks: int = 0` and clamps
  every range start: `lo = max(lo, covered_chunks)` — including the
  `hit_chunks < 0` (unconsumed prefetch) branch.
- Callers pass the session's recorded value:
  - `FREE_LOOKUP_LOCKS` handler (`LookupModule.free_lookup_locks`);
  - the failed-RETRIEVE release in
    `lmcache_driven_transfer.py` (the session's
    `prepare_failed_retrieve_release` lock-state tuple carries
    `covered_chunks`).
- `_free_inconsistent_lookup_locks` (multi-server hit disagreement) frees
  `[min_chunks, hit)`; since every server reports `hit >= c0`, the range is
  always inside the locked region — no change needed.

## Connector-side design (`lmcache/integration/vllm/lmcache_mp_connector.py`)

1. **Submit** — `get_num_new_matched_tokens` passes
   `covered_tokens = num_vllm_hit_tokens` into
   `maybe_submit_lookup_request`, which embeds
   `covered_chunks = covered_tokens // tokens_per_chunk` in the key.

   Alignment: vLLM reports `num_computed_tokens` at its
   `_cache_hit_alignment_tokens` granularity (hash-block or
   scheduler-block size — the finest common block boundary), which for
   hybrid models is finer than any single group's block and finer than
   the LMCache chunk. So the value is floored twice: the connector's
   existing `_hit_alignment_tokens` (`lcm` of per-group `tokens_per_block`)
   floor giving `num_vllm_hit_tokens`, then the `// tokens_per_chunk`
   floor giving `covered_chunks`. `c0 = covered_chunks * tokens_per_chunk`
   is therefore chunk-aligned and always `<=` the true APC hit — the
   split, clamp, and touch all operate on clean chunk boundaries, and
   tokens in the sub-chunk gap `[c0, num_vllm_hit_tokens)` are simply
   looked up normally (conservative: fewer chunks skipped, never more).
   This also makes the "covered range subset of vLLM GPU" invariant
   trivially hold, since `c0 <= actual APC hit`.

   The
   tracker records `lookup_covered_tokens` only when this call actually
   submits. The eager-prefetch path (`on_new_request`) runs before the APC
   hit exists and submits with 0; the later call is deduplicated by
   `_pending_lookups`, so the optimization is simply inactive for
   eager-submitted requests.
2. **Result** — `check_lookup_result` captures the per-server ack values
   and returns `LookupResult(hit_tokens, stored_tokens)` (Python-level API
   change only). Both fields take the min across servers; per-server
   `stored = f(covered_present, hit)` per the table above.

   Ack bookkeeping: the resolved ack values land in a new
   `_per_server_covered: dict[request_id, dict[url, int]]`, populated as
   `_LookupAck` futures resolve and consumed at final aggregation.
   `cleanup_lookup_result` must pop it (leak prevention, same lifecycle as
   `_per_server_hits`). The existing ack-timeout path (server marked
   unhealthy, lookup returns 0) is unchanged; a missing ack value degrades
   to `covered_present = 0` for that server.
3. **Decision math** — `need_to_load` uses `hit_tokens` exactly as today;
   `tracker.increase_num_stored_tokens(stored_tokens)` replaces the current
   `increase_num_stored_tokens(ret)`; `num_lmcache_hit_tokens = hit_tokens`
   (drives `needs_retrieve` and free ranges); `cached_token_stats`
   reports `stored_tokens` as the true LMCache hit.
4. **APC block pinning during lookup** — the covered boundary is frozen
   at submit time, but vLLM does not protect a WAITING request's matched
   blocks: they sit at `ref_cnt == 0` in the free queue until
   `allocate_slots` touches them, and the scheduler re-matches from
   scratch on every poll. The connector closes this window itself using
   the already-bound GPU block pool (`bind_gpu_block_pool`, invoked by
   vLLM with `kv_cache_manager.block_pool`):

   - `get_num_new_matched_tokens` receives only the integer
     `num_computed_tokens`, not the matched `KVCacheBlocks`, so the block
     ids must be re-derived rather than read from the call. At the actual
     lookup submission (once per submission, same synchronous scheduler
     step that computed the match — no allocation or eviction intervenes,
     so it is race-free), re-derive the covered blocks from
     `request.block_hashes[: c0 // block_size]` via
     `BlockPool.get_cached_block(...)` — the identical lookup vLLM's own
     finder uses, so it returns the exact blocks vLLM matched — then
     `pool.touch(blocks)` to pin. `get_cached_block` is a pure read; only
     `touch` changes the refcount. This is the pin pattern the
     lazy-offload manager already uses. Record the pinned block ids on the
     tracker. For SWA/mamba groups `get_cached_block` returns nothing for
     the null-front positions, so the loop naturally pins only the blocks
     that exist (the full-attention prefix plus any resident window
     blocks). The double touch (ours at submit, vLLM's at `allocate_slots`)
     balances: vLLM's `touch` only removes from the free queue at
     `ref_cnt == 0`, so our prior pin does not disturb it, and our release
     brings the count back to the ref vLLM holds.
   - Release exactly once with `pool.free_blocks(blocks)` via an
     idempotent helper called from both `update_state_after_alloc`
     (request admitted — vLLM's own `allocate_slots` has re-touched the
     blocks by then, so the release never drops an in-use block to zero;
     this covers the hit, miss, and bypass outcomes) and
     `request_finished` (request aborted while the lookup was in
     flight). An unreleased pin is a permanent GPU-block leak, so the
     release paths are the correctness-critical part and carry
     leak-assertion tests.
   - Pinning holds the covered blocks out of the free pool for the
     lookup-polling window (bounded by lookup latency / `mq_timeout`).
     The blocks would be allocated to this request anyway; under burst
     pressure this slightly favors waiting requests with APC hits over
     new allocations, matching the effect of scheduling them a few steps
     earlier.
   - **Hybrid-model pin scope**: the APC hit's servability is per-group
     (full attention: the whole prefix; SWA: the trailing window blocks;
     mamba: the boundary state block), so a prefix-hash derivation pins
     only the full-attention set. Reproducing each group's servability
     set would duplicate vLLM's per-group `find_longest_cache_hit`
     logic — rejected as version-coupled. Phase 1 therefore pins
     exactly for uniform full-attention models (shrink race eliminated)
     and pins only the full-attention groups' prefix blocks for hybrid
     models (shrink race reduced; the bypass guard is the correctness
     mechanism for SWA/mamba block eviction, and its firing is logged).
     The clean long-term fix is upstream: vLLM exposing or pinning the
     matched blocks for connector-pending requests.
   - Release is never re-derived: it frees the recorded
     `tracker.pinned_apc_block_ids` list, so whatever shape the pin took,
     the unlock is symmetric by construction (same principle as the
     server-side lock clamp: derive at acquisition, record, release
     only what was recorded).
   - **Coexistence with lazy-offload pins**: lazy offload maintains its
     own store-pin bookkeeping (`_requests.register_batch`). The APC pin
     set (`pinned_apc_block_ids`) is kept strictly separate; each purpose
     frees exactly its own set — never a global "release all pins for this
     request". Because `touch`/`free_blocks` are pure increment/decrement,
     the two sets compose correctly even if a block appears in both
     (ref +2, two independent frees); correctness does not depend on them
     being disjoint. In practice they are temporally disjoint anyway (APC
     pin releases at `update_state_after_alloc`, before generation; the
     lazy store pin is taken at drain, after generation).

5. **APC-shrink fallback guard** — with pinning active the covered
   blocks cannot be evicted, so the aligned hit can no longer drop below
   `lookup_covered_tokens`. The guard remains as a defensive layer for
   the cases where pinning is unavailable (block pool never bound on an
   older vLLM, or `get_cached_block` missing a block at submit time): if
   the current aligned hit is below the frozen boundary at decision time,
   the connector reuses the existing failed-load bypass pattern — free
   the locks (the server clamps the range to `[c0, ret)`), zero the
   tracker's hit/stored counters, enter `BYPASS_LMCACHE`, and return
   `(0, False)`.
6. **Free-RPC elision** — `update_state_after_alloc` skips the
   `free_lookup_locks` RPC when `free_end <= lookup_covered_tokens`
   (the common case: nothing is locked below `c0`). The RPC still fires
   when the APC hit grew past the submit-time boundary
   (`free_end > c0`), freeing exactly `[c0, free_end)` after server-side
   clamping.
7. **Re-lookup after preemption / failed load** — `lookup_covered_tokens`
   is bound to one lookup submission. Whenever lookup state is reset and a
   fresh lookup may be submitted (`cleanup_lookup_result` on the
   failed-async-load bypass, preemption resubmission), the tracker's
   `lookup_covered_tokens` is reset with it and re-recorded on the next
   actual submit. Server-side, `begin_lookup` overwrites the session's
   `covered_chunks` together with the lookup generation, so a stale value
   can never clamp a newer lookup's locks.

## Preemption

A rescheduled preempted request never engages this feature: the MP
connector bails out of `get_num_new_matched_tokens` for
`status == PREEMPTED` before any lookup is submitted (existing TODO in
`lmcache_mp_connector.py`), so vLLM recomputes locally regardless of
whether its APC hit survived preemption. That is pre-existing behavior,
unchanged here.

What this design guarantees for that scenario is that LMCache remains able
to serve the rescheduled request once preempted-request loading is
implemented: the covered chunks were LRU-touched at lookup time (warm, not
eviction bait) and any presence hole in the covered range was re-stored
during the first pass, so a future fresh lookup — submitted with a fresh
covered boundary, `0` if the APC is gone — hits the full contiguous
prefix. No change to this design is needed when that TODO lands.

## Sliding-window / hybrid correctness (the straddle case)

For a sliding-window (or recurrent, `w == 1`) group, the serving window
`[ret - w, ret)` commonly straddles the covered boundary when
`ret - w < c0 < ret`: part of the window is APC-covered, part is uncovered
and served from LMCache. Marking covered chunks *satisfied* in the fold is
correct here, and the reason is a subset invariant rather than LMCache
residency:

```
[ret - w, c0)  ⊂  [c0 - w, c0)
```

The right side is exactly the window that vLLM's all-groups-agree APC hit
at `c0` guarantees resident in GPU. So the covered part of every group's
window (for every candidate length `L in [c0, ret]`) is backed by vLLM's
GPU blocks — the fold may treat it as present without LMCache holding it.

Three existing mechanisms complete the picture, so no window-aware server
change is needed beyond the lock clamp:

- **Retrieve** already reads only the uncovered part `[c0, ret)` for a
  SWA group (`start = floor(vllm_hit)`, per-group window skip,
  `skip_first_n_tokens`); the covered window portion comes from vLLM GPU.
  No covered object is read from LMCache.
- **Locks**: today the straddle's covered part `[ret - w, c0)` is
  locked-then-freed; under this design it is simply never locked. The
  per-group release clamp (`lo = max(lo, c0)`, inside the per-group
  `lo/hi` computation, not a global floor) keeps release from touching
  those never-taken locks.
- **Partial LMCache window** (a gap in `[c0, ret)`) just shortens `ret`:
  the fold requires the whole window present, so no holey window is ever
  served.
- **Partial vLLM window** (`c0 < w`, the SWA finder's start-anchored
  `not match_found` branch): the reported prefix is fully contiguous from
  token 0, so `[0, c0)` has no null gaps and the whole covered range is
  physically resident. When `c0 >= w` the covered front `[0, c0-w)` is
  null, but LMCache never consults it — every window `[ret-w, ret)` for
  `ret >= c0` meets the covered range only inside `[c0-w, c0)`, which is
  the present window.

**Where `c0` comes from:** for hybrid models vLLM's
`HybridKVCacheCoordinator.find_longest_cache_hit` runs an iterative
fixed-point across groups — each group's finder accepts or reduces the
candidate length until convergence — and reports a single reconciled
`num_computed_tokens` equal to the longest prefix every group can serve
(full attention: all blocks present; SWA: its trailing window present,
earlier blocks null; mamba: boundary state present). That reconciled
value is the all-groups-agree boundary this design relies on, computed by
construction rather than assumed.

**Load-bearing precondition:** the argument requires `c0` to be that
reconciled all-groups-agree boundary. vLLM's divergent-hybrid path
(`supports_divergent_local_hybrid_hits`) can otherwise hand the connector
a full-attention-only hit that runs deeper than a lagging SWA/mamba group,
whose window at `c0` is *not* in GPU — marking it satisfied would
over-report. The MP connector does not override that property (defaults
`False`), so vLLM feeds it the reconciled `get_computed_blocks()`
boundary. The implementation must keep this invariant explicit: assert /
gate that the connector does not opt into divergent hybrid hits while
`skip_covered_lookup` is enabled, so a future connector change cannot
silently break the straddle correctness.

## Background: the APC-shrink window and why pinning closes it

vLLM does not protect a WAITING request's prefix-cache match:
`get_computed_blocks` is a pure read (`find_longest_cache_hit`, no
ref-count change), and `block_pool.touch` — the call that removes blocks
from the free queue — only runs inside `allocate_slots`, which the
scheduler reaches only after the connector returns a token count. While
our async lookup returns `None`, the scheduler pops the request, prepends
it to `step_skipped_waiting`, and `continue`s — discarding the matched
blocks, which remain `ref_cnt == 0` eviction candidates in the free
queue. Any concurrent allocation (`get_new_blocks` →
`_maybe_evict_cached_block`) can reclaim them, and the next poll's
re-match comes back smaller.

Today's full-range lookup absorbs a shrink seamlessly (everything from
chunk 0 is locked in LMCache, the connector simply loads more). With the
covered range deliberately unlocked, a shrink below `c0` would leave
tokens backed by nothing. There are two ways to keep it safe — pinning the
GPU blocks so a shrink cannot happen, or leaving them unpinned and
re-looking-up the gap on shrink (the approach vLLM's own native offloading
connector takes). They are compared next.

## Shrink handling: pin vs. non-pin (re-lookup)

Both skip lookup of the covered prefix and touch it to avoid eviction. They
differ only in how they keep it backed if the APC hit shrinks mid-lookup.

- **Pin** — freeze `c0`, pin the covered GPU blocks for the lookup window
  so a shrink can't happen.
- **Non-pin (re-lookup)** — don't freeze or pin; touch the prefix in L1+L2,
  and if the APC shrinks `c0 -> c0'`, look up just the gap `[c0', c0)` and
  fetch it L2→L1. This is what vLLM's native offloading connector does.

| | Pin | Non-pin (re-lookup) |
| --- | --- | --- |
| **Idea** | Hold the covered GPU blocks so they can't be evicted | Let them go; re-fetch the gap from LMCache if needed |
| **Pros** | Shrink impossible; simple accounting; no extra RPC | No GPU-pool pressure; matches vLLM; handles grow and shrink |
| **Cons** | Holds blocks out of the pool (up to `mq_timeout`); couples to vLLM block-pool internals | Extra gap lookup + L2→L1 fetch on shrink (rare); gap lookup must not re-lock `[c0, ret)` |
| **Main risk** | Pool starvation under load (risk A) — needs a pin budget | Soft guarantee: a gap chunk could evict before re-fetch → small local recompute |
| **Efficiency** | Best happy path; cost is pinned-block opportunity cost | Same happy path; pays only the gap delta on shrink |
| **Complexity** | Simpler server, heavier connector (pin lifecycle + budget) | Heavier server (incremental gap lookup), lighter on GPU coupling |
| **Robustness** | Eliminates the race but can starve the pool | No availability risk; degrades gracefully but leaves a tiny miss window |

**Guidance.** L1-only / small-L2: pin — short window, hard guarantee,
simpler. Large-L2 / high-concurrency: non-pin — no starvation, cheap rare
gap fetch. Likely end state: non-pin as default, pin as an opt-in (or
budget-gated with re-lookup as the fallback). Either simplifies if vLLM
upstream ever pins connector-pending matches.

## Observability

`MP_LOOKUP_PREFETCH_END` gains `covered_tokens` and `stored_prefix_tokens`
metadata. `hit_tokens` keeps the satisfied semantics (documented); the
L1/L2 split subtracts satisfied-not-present chunks via
`PrefetchHandle.covered_chunks` so dashboards do not over-report L1 hits.

## Shared invariants (model-independent)

These hold for dense and hybrid alike and form the foundation the
hybrid-only work (fold straddle, partial pin) builds on. Each has a
single enforcement point and a test written against the failure mode, not
the happy path.

| Invariant | Single enforcement point | Failure mode if violated | Test |
| --- | --- | --- | --- |
| No release resolves keys below the session's covered boundary | `resolve_prefetched_obj_keys`: `lo = max(lo, covered)` per group; covered recorded at `begin_lookup` with the lookup generation; all five release callers pass the session value | Over-release drops a concurrent prefix-sharing request's read lock → its object evictable mid-retrieve | Two requests share a prefix; A's release leaves B's refcounts intact; pool lock counts return to baseline |
| Each APC pin is released exactly once, after `allocate_slots` re-touches (or on abort) | one idempotent `_release_apc_pins(tracker)` helper (frees `tracker.pinned_apc_block_ids`, then clears it) called from every terminal path | Unreleased pin permanently removes a GPU block from the pool; double release drops a block the request still owns | Five terminal paths (hit / miss / bypass / abort / timeout) each restore pool refcounts to baseline |
| Pin taken exactly once per submission | taken only in the `get_num_new_matched_tokens` call that actually submits, guarded by `pinned empty and covered > 0 and pool bound` | Repeated pins across scheduler polls leak refcounts | Poll `get_num_new_matched_tokens` N times while lookup pending; exactly one pin taken |
| `num_stored_tokens` is the contiguous LMCache-present prefix, never the satisfied hit | two-number result; `stored = covered_present if covered_present < c0 else hit`; `increase_num_stored_tokens(stored)` | A covered chunk missing from LMCache is marked stored → permanent hole, future lookups capped there | Hole at covered chunk k → `num_stored_tokens` stops at k → offload re-stores from k |
| Ack bookkeeping never outlives the request | `_per_server_covered` popped in `cleanup_lookup_result` | Slow scheduler-process memory leak | Finished request leaves no `_per_server_covered` entry |
| Old server never corrupts a new connector | `None` ack → `covered_present = 0` (over-store, never over-report) + warn once; flag default-off; servers upgrade first | Version skew → assumed-unlocked covered range actually locked → lock leak until TTL | `None`-ack path yields `stored = 0` and full behavior; warn emitted once |

The pin block ids index the single shared `BlockPool`
(`kv_cache_manager.block_pool`); pin is `pool.touch`, release is
`pool.free_blocks`, mirroring the lazy-offload manager.

## Testing

- `L1Manager.peek_keys`: takes no locks; staging objects invisible.
- Storage manager: covered keys never locked; L2 request excludes covered
  keys; a presence hole in the covered range does not disturb `released`;
  `hit_chunks >= covered_chunks` invariant; full-coverage degenerate case
  (`covered_chunks >= num_chunks`: no locks, no L2, `hit = num_chunks`,
  status polling still works).
- `LookupModule`: ack carries `covered_present_chunks`; touch listener
  fires for present covered keys only; `free_lookup_locks` and the
  failed-RETRIEVE path release exactly `[c0, hit)` (lock-count
  assertions).
- Adapter/connector: per-server min combine; `None`-ack (old server)
  fallback to `stored = 0`; shrink fallback guard enters bypass and frees
  locks; free-RPC elision; store-hole repair (`num_stored_tokens` stops at
  the hole and the offload path re-stores from it).
- APC block pinning: pin taken exactly once per lookup submission;
  released exactly once on every terminal path (scheduled with hit,
  scheduled with miss, bypass, abort while polling, lookup timeout) —
  block-pool ref counts restored to baseline after each scenario (leak
  assertions); pinned blocks survive a simulated free-queue reclaim so
  the re-match never shrinks below `c0`; unbound-pool deployments fall
  back to the bypass guard.
- E2E with the flag on: a second request with an APC hit performs zero
  lock churn on covered keys and updates their LRU order.

## Cache-tier retention

Today every APC-covered request re-locks and re-prefetches `[0, c0)`,
promoting L2→L1 and keeping an always-covered prefix hot in L1. Covered
skip removes that promotion, so an always-covered prefix could sink to L2
and eventually evict — a double miss when APC finally drops it.

Retention is preserved without re-promotion, by two mechanisms covering
the two L2-eviction regimes:

- **Backend-managed L2** (Redis, S3, mooncake, bigtable — eviction is the
  backend's TTL / own policy): the store-path self-heal covers it. A
  covered chunk that evicts is found absent by the next covered request's
  L1 peek, `num_stored_tokens` stops at it, and the store path re-writes
  it from the GPU APC blocks — refreshing the backend TTL. This makes the
  two-number decoupling (see the store-hole invariant) load-bearing for
  retention, not only for hole avoidance.
- **LMCache-managed L2** (adapters with `supports_global_eviction` or a
  quota policy — LMCache owns the LRU): a covered-range L2 touch refreshes
  recency directly. Firing `L2_KEYS_ACCESSED` flows to
  `L2EvictionPolicy.on_l2_keys_accessed → on_keys_touched` with no byte
  movement (mirroring `touch_l1_keys` one tier down). This requires
  knowing L2 residency without a fetch, so it bundles with the deferred
  L2 existence probe below.

## Deferred work

- **L2 existence probe + touch for the covered range** — today an
  L2-resident, L1-absent covered chunk reads as "not stored" and is
  redundantly re-stored (safe, suboptimal), and its LMCache-managed L2 LRU
  recency is not refreshed. A follow-up runs the covered keys through the
  L2 lookup phase (existence only, no fetch, no lock) and fires
  `L2_KEYS_ACCESSED` for the hits — one probe that fixes both the
  redundant re-store and the LMCache-managed retention path.
- Covered-hint support for the SPARSE trim policy (blend).
