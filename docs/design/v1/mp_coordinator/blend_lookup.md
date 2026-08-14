# Fleet-wide CacheBlend lookup

A coordinator-level capability that lets a CacheBlend lookup on one
mp-server find and reuse chunk KV cached anywhere in the fleet, fetched
from a shared, content-addressed L2 or directly from a peer's L1 over the
P2P transfer channel (RDMA). It is **opt-in** (a coordinator URL plus
cache-event reporting) and **additive** (the local matcher still runs).

Code: `lmcache/v1/mp_coordinator/blend_index.py`,
`lmcache/v1/mp_coordinator/blend_client.py`,
`lmcache/v1/multiprocess/modules/blend_v3.py`.
Index internals: [blend_index.md](blend_index.md).

## Why

CacheBlend lookup on its own is **local to one mp-server**: the matcher
(`BlendTokenRangeMatcherV3`, `blend_v3.py`) indexes only chunks that
server stored, so a request routed to a different replica recomputes KV a
peer already holds. As replicas scale, cache sharding works against
reuse. The coordinator federates content fleet-wide so any server can
discover and reuse what any other cached.

## Where the fleet index comes from

**Nothing is published for blend specifically.** The coordinator's
blend index is a derived view of the key directory's token bindings,
which are built from the cache-event stream every mp server already
emits (`mp.tokens` + the L1/L2 store/delete events — see
[cache_events.md](cache_events.md)). A blend store rides
`LMCacheDrivenTransfer.store`, so its chunks and their token content
reach the directory with no blend-specific plumbing at all.

```
STORE (blend server, worker-0)
  transfer module emits mp.tokens (chunk hashes + tokens + offsets)
  storage layer emits l1/l2 store events
        ── POST /events ──▶ key directory: token bindings
                                        └─▶ blend index: fingerprints

LOOKUP (blend server, cb_unified_lookup)
  local matcher (always) ─┐
  request tokens          │
    ── POST /directory/blend-lookup ──▶ blend index: roll + filter +
                                        resolve + verify token-exact
    ◀── matches: [(chunk_hash, old_st, cur_st)] ──┘
  → union of both sources
  → non-prefix set (drop prefix-covered, leftmost-greedy overlap dedup)
  → one sparse prefetch (shared L2 and/or peer L1 via P2P)
  → retrieve + re-RoPE
```

Consequences of being event-fed rather than blend-published:

- **Eviction is exact.** A chunk's binding — and its index entry — dies
  with its last placement, so a match cannot point at evicted bytes.
- **Matches are token-verified**, not hash-trusted, so a fingerprint
  collision costs a skipped candidate rather than a wasted prefetch.
- **Placements are known**, so the directory can already tell a peer-L1
  copy from a shared-L2 one (not yet used for ranking).
- **It needs event reporting on.** See Gating below.

## Identity and scope

The index is keyed by **content only** — no model, salt, or rank. A match
names a `chunk_hash`, and the querying server expands it into per-rank
`ObjectKey`s using **its own** `cache_salt`, `model_name`, and
`world_size` (`ipc_key_to_object_keys`), exactly as the local path does.
So a cross-salt or cross-model match lands in the requester's own
namespace and confirmed-misses at the sparse prefetch unless a matching
copy exists — tenant isolation holds with **one index for the fleet**.
Filtering by the *storer's* identity would be wrong: content is shared
first-writer-wins, so the first storer's tenant would get pinned and
others could never match their own copies of identical content.

The cost is that a cross-tenant match with no same-salt copy is a wasted
prefetch — the already-tolerated stale-entry failure mode. The index does
reveal cross-salt content *existence* to the trusted mp-servers, and the
coordinator sees raw tokens (content), not just opaque hashes:
acceptable for trusted fleet infra, revisit if not.

## Lookup flow in `cb_unified_lookup`

Local and fleet matching are **additive**. The local matcher always runs;
a coordinator only adds candidates.

1. First call: run the local matcher **and** submit the coordinator query
   (request tokens). A per-lookup wall-clock deadline is armed
   (`match_budget_s`, from `LMCACHE_COORDINATOR_BLEND_TIMEOUT`).
2. After the prefix resolves, poll the coordinator **before** the sparse
   prefetch: defer while pending, give up at the deadline (then that
   lookup is local-only). The deadline — not just the per-request HTTP
   timeout — bounds the total wait, since queue time counts.
3. Union both sources, keep matches outside the prefix coverage, and
   apply **leftmost-greedy overlap dedup**: neither source is
   overlap-free with respect to the other, and two matches over the same
   request range can't both scatter. Chunks both sources report collapse
   here.
4. Submit **one** sparse prefetch over that set, classify, retrieve.

**Why additive rather than fleet-only.** The fleet index is not a
guaranteed superset of the local table: it is fed by best-effort cache
events, so a dropped flush or a token binding evicted from the emitter's
bounded cache leaves a chunk *this server holds* unmatchable fleet-wide.
Running both makes recall the union, which is never worse than either
alone, and costs only the local matcher's CPU (a vectorized probe) on the
path that was already paying for a round-trip.

### Fetch sources: shared L2 and peer L1

The sparse prefetch fans out to **every** registered L2 adapter — P2P
peer adapters included — so a matched chunk loads from whichever tier
holds it: shared L2, or a peer's L1 via RDMA. The peer's
`p2p_lookup_and_lock` locks sparse key sets (gaps allowed), not only the
contiguous prefix, and the serving side never recurses into its own L2
(`skip_l2` holds for sparse lookups too). Because chunks reach the
directory on their store events regardless of L2 offload, chunks resident
only in a storer's L1 are matchable fleet-wide.

Fleet matches become `CBMatchResult` (each `hash` is the chunk content
hash — the coordinator's `chunk_hash`), so they ride the **identical**
sparse prefetch + classify + retrieve + re-RoPE path as local matches and
surface in `CBUnifiedLookupResult.non_prefix_segments`. There is no
separate `global_segments` field and no protocol change.

## Gating

Fleet matching requires **both**:

- `--coordinator-url` (or `LMCACHE_COORDINATOR_URL`), and
- `--coordinator-event-reporting` (or
  `LMCACHE_COORDINATOR_EVENT_REPORTING`), which defaults to **off**.

With a URL but no event reporting the coordinator has no cache state to
match against, so every query would return empty. `server.py` treats
that as not-configured and logs a warning at startup rather than paying a
round-trip per lookup for nothing. Since matching is additive, blend
still works locally in that case — the fleet leg is simply absent.

## Failure modes

| event | effect | handling |
| --- | --- | --- |
| coordinator down | no fleet leg | HTTP times out → empty → local matches only |
| event reporting off | no fleet leg | detected at startup, warned, client not created |
| cache-event batch dropped | a chunk unindexed fleet-wide | local matcher still finds it; re-indexed on its next store |
| emitter token-binding evicted | chunk unindexed fleet-wide | same |
| fingerprint collision | candidate skipped | token verification rejects it |
| chunk evicted from peer L1, no L2 copy | wasted prefetch | miss → recompute |
| cross-salt / cross-model match, no local copy | wasted prefetch | requester-identity ObjectKey misses → recompute |

## Future evolution

Reuse is whole-chunk and only where content is chunk-phase aligned
(store-side non-overlapping chunks). Two refinements raise reuse, gated
by the partial-KV-transfer work:

- **Block-level** — match at the inference block size `G`
  (`block_content_hashes` + minimizer-sparse anchor index +
  seed-and-extend), so reuse is `G`-grained and partial chunks are
  reusable. Still `G`-phase-sensitive.
- **Token-level** — rolling k-mer **minimizer seeds** + token-level
  extend, giving arbitrary-offset reuse and ragged partial-page tails.
  Requires dense per-token hash arrays at the coordinator.

### Role of the minimizer (for the refinements)

The minimizer picks ~1 anchor per window `W` by **content** (local-min
hash). It **decouples** three things: match completeness (a run `≥ W`
shares an anchor), index cost (`÷W`), and offset (content-defined
selection is offset-free). It sparsifies the *seed index* only; the dense
per-position arrays needed to *extend* stay dense. With it, you keep a
fine match granularity *and* a small index; without it you must coarsen
granularity to shrink the index.

### Complexity (n=request tokens, m=stored tokens, C=chunk, G=block, W=window)

| | current | block-level | token-level |
| --- | --- | --- | --- |
| store | O(m) hash + O(m/C) insert | O(m) + O(m/(G·W)) | O(m) + O(m/W) |
| lookup | O(n) roll + O(n/stride) probe + O(hit·C) verify | O(n) + O(n/(G·W)) + O(hit/G) | O(n) + O(n/W) + O(hit) |
| coordinator memory | O(m) tokens + O(m/C) index | O(m) + O(m/G) | O(m) |
| match unit | C (256) | G | 1 token |
| offset | store chunk-phase bound | G-phase bound | arbitrary |

Token verification is what puts `O(m)` in the memory row today: the
tokens must stay resident to verify against. That buys exactness and,
because the tokens are already in the directory for introspection, adds
no new wire cost.

## Scope

Additive: no change to local prefix/blend lookup, retrieve/re-RoPE/
scatter, or the coordinator backbone. Composes via the documented
extension seam — the `/directory` router plus the opt-in blend client —
with no edits to membership or the health loop.
