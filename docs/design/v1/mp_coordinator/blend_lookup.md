# Global CacheBlend Lookup (Coordinator Fingerprint Directory)

A coordinator-level capability that lets a CacheBlend lookup on one mp-server
find and reuse chunk KV cached anywhere in the fleet, fetched from a shared,
content-addressed L2. Blend servers **publish fingerprints on STORE** and
**query on LOOKUP**; the coordinator holds the fleet-wide directory and runs the
match. It is **opt-in** (enabled only when a coordinator URL is configured) and
**additive** (the existing local matcher fast path is unchanged).

Code: `lmcache/v1/mp_coordinator/blend_directory.py`,
`lmcache/v1/mp_coordinator/http_apis/blend_directory_api.py`,
`lmcache/v1/multiprocess/modules/blend_coordinator.py`.

## Why

CacheBlend lookup today is **local to one mp-server**: the matcher
(`BlendTokenRangeMatcherV3`, `lmcache/v1/multiprocess/modules/blend_v3.py`)
indexes only chunks that server stored, so a request routed to a different
replica recomputes KV a peer already holds. As replicas scale, cache sharding
works against reuse. The coordinator directory federates fingerprints so any
server can discover and reuse content cached fleet-wide.

## This PR: the original matching algorithm, globalized

The coordinator reuses the **exact matching algorithm** the local matcher
already uses — non-overlapping chunk fingerprints + a strided rolling-hash probe
— lifted to a fleet table. Nothing about the match semantics changes; only its
*scope* (one server → the fleet) and its *transport* (in-process → HTTP).

**The coordinator does all hashing.** Servers send **raw tokens** plus the
storage mapping only they know; the coordinator chunks, hashes, and matches. This
keeps the whole matching algorithm in one place, so future evolution (see below)
is a coordinator-only change — no server hashing change, no fleet redeploy, no
cross-server hash-base consistency to maintain.

```
STORE (blend server, worker-0)
  per stored range: tokens[start:end] + object_keys (per chunk) + old_st_base
        ── POST /blend/fingerprints ──▶ coordinator.register():
                                          chunk tokens, hash each → (poly→object_key,old_st)

LOOKUP (blend server, cb_unified_lookup)
  coordinator present? fleet match only (local matcher skipped); else local only
  request tokens
        ── POST /blend/match ──▶ coordinator.match():
                                   roll hash over tokens, probe every probe_stride
        ◀── matches: [(object_key, old_st, cur_st)] ──
  → non-prefix set (drop prefix-covered, leftmost-greedy overlap dedup)
  → one sparse prefetch from shared L2 → retrieve + re-RoPE
```

The coordinator owns `chunk_size` (`blend_chunk_size`, fleet config, must equal
the servers' LMCache chunk size) and the polynomial base; servers carry neither.
**Trust note:** the coordinator now sees raw tokens (content), not just opaque
hashes — acceptable when it is trusted fleet infra; revisit if not.

### What is matched, and the granularity knob

- **Match window = `chunk_size`** (the LMCache chunk, 256, a fleet config). A
  stored chunk's content is one polynomial hash (`chunk_hash_windows_numba`);
  a match is "this request window equals a stored chunk."
- **Probe stride** = how many positions the coordinator skips between probes
  (`rolling_hash_windows_numba` is computed over every position; the table is
  probed every `probe_stride`). It is a **coordinator-side config**
  (`blend_probe_stride`, default `1`), **not** sent by servers and **not** tied
  to any vLLM block size.

Why not the vLLM block size: with **partial-fill** KV transfer a matched chunk is
reusable at **any** `cur_st` (the server scatters it token-wise, filling ragged
pages), so there is no block-alignment constraint to honor. The default
`probe_stride=1` therefore probes every offset for full recall; raise it only to
trade recall for coordinator CPU. (vLLM block sizes are per-machine and
dynamically computed — another reason not to bake one into the match.) `stride`
controls probe density; `chunk_size` is the reuse unit.

### Identity and scope

The table is keyed by `(model_scope, poly_hash)`:

- **`poly_hash`** — the content-only 64-bit polynomial chunk hash
  (`chunk_hash_windows_numba` with the fleet-constant base `POLY_BASE`), computed
  **by the coordinator** from the published tokens.
- **`model_scope`** = `f"{model_name}@{cache_salt}"`. CacheBlend reuse is
  same-model only (K is model-specific), and `cache_salt` is explicit cache
  isolation — both are part of `ObjectKey` (`distributed/api.py`). Content from a
  different scope lands under a different key and is never matched.

**TP rank is not in the key.** KV is stored per rank, but the directory matches
content (rank-agnostic); the querying server expands a matched `object_key` into
its `world_size` per-rank `ObjectKey`s at retrieve (`ipc_key_to_object_keys`),
exactly as the local path does. So one fingerprint serves all ranks.

The 64-bit poly-hash carries the same collision behavior as the local matcher
(which also matches on the 64-bit poly); acceptable within a model scope, and a
collision only causes a wasted prefetch (caught downstream), never wrong KV.

## Coordinator directory (`blend_directory.py`)

`GlobalBlendMatcher` (thread-safe, single lock, mirroring `InstanceRegistry`):

```
_index   : dict[(model_scope, poly_hash:int) -> ChunkLoc(object_key, old_st)]
_by_key  : dict[object_key -> list[(model_scope, poly_hash)]]   # for eviction
```

- `register(ranges)` — for each `StoreRange(model_scope, tokens, object_keys,
  old_st_base)`, chunk the tokens, hash each chunk, and insert
  `(model_scope, poly_hash) → (object_key, old_st)`; idempotent (first-writer
  wins per key).
- `remove(object_keys)` — drop all entries for an evicted chunk (reverse map).
- `match(model_scope, tokens)` — roll a chunk-window hash over the request
  tokens, probe every `probe_stride` positions against the scope's table; dedup
  by `object_key`; return `[(object_key, old_st, cur_st)]`. Mirrors
  `BlendTokenRangeMatcherV3.match_sub_sequence`, minus the local direct-address
  table (a real dict gives exact 64-bit match, no separate collision step).

It is **ephemeral**: rebuilt from publishes after a restart; a stale instance
leaving the fleet does not drop fingerprints (shared-L2 object keys stay valid).

## Blend-server comms (`blend_coordinator.py`)

Blend handlers run in sync thread pools with no asyncio loop, so the client owns
a synchronous `httpx.Client` plus a daemon, mirroring the module's existing
`_fingerprint_queue` worker:

- **STORE**: `enqueue_register` → best-effort, fire-and-forget `POST
  /blend/fingerprints` (register). Eviction, when wired, is the distinct
  `enqueue_evict` → `DELETE /blend/fingerprints`.
- **LOOKUP**: `submit_match` (once per request) + `poll_match` — the same
  non-blocking submit/poll pattern the prefix and sparse legs already use;
  `cb_unified_lookup` polls until the match resolves and merges it. The single
  wall-clock bound is the client HTTP `request_timeout` (default 50 ms): a slow
  or down coordinator returns/​times-out into an empty result, so the lookup
  proceeds local-only without stalling. No separate poll-count budget.

Opt-in via `LMCACHE_COORDINATOR_URL`; absent → the module receives `None` and
every publish/query path is skipped (behavior unchanged).

## Lookup flow in `cb_unified_lookup`

Local and coordinator matching are **mutually exclusive**, chosen by coordinator
presence: with a coordinator configured the fleet directory (a superset of the
local table) is the *only* match source and the local matcher is skipped
entirely; with none, matching is purely local as before. There is no merge.

With a coordinator:

1. First call: skip the local fingerprint match; submit only the coordinator
   match (request tokens). A per-lookup wall-clock deadline is armed
   (`match_budget_s`, from `LMCACHE_COORDINATOR_BLEND_TIMEOUT`).
2. After the prefix resolves, poll the coordinator **before** the sparse
   prefetch: defer while pending, give up at the deadline (then no fleet matches
   that lookup). The deadline (not just the per-request HTTP timeout) bounds
   total wait, since the coordinator daemon services match queries serially.
3. Keep matches outside the prefix coverage and apply **leftmost-greedy overlap
   dedup** — the coordinator dedups only by `object_key`, so its matches can
   still overlap, and two matches over the same request range can't both scatter.
4. Submit **one** sparse prefetch over that set, classify, retrieve.

Without a coordinator, step 1 instead runs the local matcher and steps 3–4 use
its matches (already non-overlapping, so the dedup is a no-op).

Fleet matches are `CBMatchResult` (each `hash` is the chunk content hash — the
coordinator's `object_key` hex — which `ipc_key_to_object_keys` expands to
per-rank shared-L2 keys), so they ride the **identical** sparse prefetch +
classify + retrieve + re-RoPE path as local matches and surface in
`CBUnifiedLookupResult.non_prefix_segments`. There is no separate
`global_segments` field and no protocol change.

## Eviction & staleness

Lazy. On local eviction the server may publish a `remove`, but the directory
tolerates stale entries: a match to an evicted object key simply misses at sparse
prefetch → the server recomputes. Never wrong KV. (Eager per-chunk eviction
publish is optional and not required for correctness.)

## Failure modes

| event | effect | handling |
| --- | --- | --- |
| coordinator down | no global leg | HTTP times out → empty result → local-only |
| poly-hash collision | wasted prefetch | confirmed-miss → recompute |
| stale entry (evicted) | wasted prefetch | miss → recompute; lazy remove |
| publish dropped | a chunk unindexed globally | recomputed on a peer until re-published |

## Future evolution (not in this PR)

The original algorithm reuses content only at **chunk granularity** and only when
the content is chunk-phase aligned (store-side non-overlapping chunks). Two
refinements raise reuse, gated by the partial-KV-transfer work:

- **Block-level** — match at the inference block size `G` (`block_content_hashes`
  + minimizer-sparse anchor index + seed-and-extend), so reuse is `G`-grained and
  partial chunks are reusable. Still `G`-phase-sensitive.
- **Token-level** — rolling k-mer **minimizer seeds** + **token-level extend**,
  giving arbitrary-offset reuse and ragged partial-page tails. Requires dense
  per-token hash arrays at the coordinator.

### Role of the minimizer (for the refinements)

The minimizer picks ~1 anchor per window `W` by **content** (local-min hash). It
**decouples** three things: match completeness (a run `≥ W` shares an anchor),
index cost (`÷W`), and offset (content-defined selection is offset-free). It
sparsifies the *seed index* only; the dense per-position arrays needed to
*extend* stay dense. With it, you keep a fine match granularity *and* a small
index; without it you must coarsen granularity to shrink the index.

### Complexity (n=request tokens, m=stored tokens, C=chunk, G=block, W=window)

| | original (this PR) | block-level | token-level |
| --- | --- | --- | --- |
| store | O(m) hash + O(m/C) insert | O(m) + O(m/(G·W)) | O(m) + O(m/W) |
| lookup | O(n) roll + O(n/stride) probe | O(n) + O(n/(G·W)) + O(hit/G) | O(n) + O(n/W) + O(hit) |
| coordinator hash mem | O(m/C) | O(m/G) | O(m) |
| match unit | C (256) | G | 1 token |
| offset | store chunk-phase bound | G-phase bound | arbitrary |

All lookups are O(n + hit) asymptotically; the differences are constants,
coordinator memory (`m/C : m/G : m` ≈ `1 : 16 : 256` at C=256,G=16), and
capability (granularity + offset). This PR takes the cheapest, simplest point;
the refinements trade coordinator memory for finer, offset-robust reuse.

## Scope

Additive: no change to local prefix/blend lookup, retrieve/re-RoPE/scatter, or
the coordinator backbone. Composes via the documented extension seam — a new
`http_apis` router reading `app.state`, plus the opt-in blend client — with no
edits to membership or the health loop.
