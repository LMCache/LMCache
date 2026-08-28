# Blend index (fragment lookup)

Module: `lmcache/v1/mp_coordinator/blend_index.py`
HTTP surface: `POST /directory/blend-lookup`
(`http_apis/directory_api.py`)

The fragment counterpart to the key directory's prefix lookup, and the
follow-up [key_directory.md](key_directory.md) names as "fragment tokens
→ keys". It answers: **given a request's tokens, which cached chunks does
it contain, and where?** — the query behind fleet-wide CacheBlend reuse,
where a request assembled from documents cached anywhere in the fleet
must discover them at arbitrary offsets.

It is a **derived view of the directory's token bindings**, not a second
source of truth: bindings are created and dropped by the cache-event
stream, and the index follows. Nothing publishes to it directly.

It is **off unless the coordinator is started with
`--enable-blend-lookup`**, which calls `KeyDirectory.enable_blend_lookup`
at startup. Until then no chunk content is hashed and `blend_match`
returns nothing — a fleet that does not run CacheBlend pays nothing for
this. Chunks stored before the call are not retroactively indexed.
An entry whose `token_offset` is `UNKNOWN_TOKEN_OFFSET` fills its
binding's content but is not indexed: without the stored position a match
could not say where to re-RoPE from, and a wrong position yields wrong KV
rather than a miss.

## Why it replaces the standalone blend directory

The previous design (`blend_directory.py`, `POST /blend/fingerprints`)
maintained its own fleet table from a **separate publish path** that
blend servers drove on every store. That table held a 64-bit fingerprint
per chunk and nothing else, which forced three compromises the key
directory removes:

| | standalone fingerprint table | blend index |
| --- | --- | --- |
| fed by | its own publish RPC per blend store | the one cache-event stream |
| match | 64-bit hash, trusted | hash to discover, **tokens to verify** |
| collision | wasted prefetch | candidate skipped |
| eviction | lazy tombstone, stale entries tolerated | exact, with the binding |
| placements | unknown; prefetch is blind | known (peer L1 vs shared L2) |
| identity | one chunk hash, namespace-blind | per-namespace claims on each chunk |

The cost is coordinator memory: verification needs the tokens resident,
`O(m)` rather than `O(m/C)`. See
[key_directory.md](key_directory.md) — Token index for the
representation that keeps that affordable.

## Identity and scope

Entries are keyed by a **content fingerprint only** — the 64-bit
polynomial hash of one chunk's tokens (`POLY_BASE`, fleet-constant).
Content is the key; **identity rides on the occupants**.

- **No prefix in the key.** `chunk_hash` is prefix-chained, so the same
  text stored after different prefixes is two chunks; both attach to
  **one** content entry (each with its own `token_offset`), so evicting
  one leaves the other discoverable.
- **Each occupant carries its namespaces.** A `chunk_hash` names content
  and prefix only — everything deciding *whose* KV it is lives on
  `ObjectKey` beside it. So every occupant records the
  `BlendNamespace`s that stored it, and a match is offered only to a
  requester in one of them.

### The namespace

`BlendNamespace` (`api.py`) is `(model_name, cache_salt, world_size)` —
exactly the three fields `ipc_key_to_object_keys` reads to turn a
`chunk_hash` into `ObjectKey`s. On the store side it is derived from the
key itself, `world_size` from the top byte `ComputeKVRank` packed into
`kv_rank`; on the query side it arrives on the request. `object_group_id`
is deliberately out: groups partition a server's own layout rather than
the fleet, and blend servers must not enable `--separate-object-groups`.

**Scoping loses no legitimate reuse.** Each dimension is one the
requester's own key expansion would miss on anyway: another model's KV is
not interchangeable, another tenant's salt is isolated by design, and
another parallel setup shards heads differently. What scoping removes is
two real defects of a namespace-blind table:

| defect | namespace-blind | with claims |
| --- | --- | --- |
| a match no requester key can reach | returned; confirmed-misses at prefetch, burning a blend slot | never offered |
| the requester's own chunk hidden behind another namespace's | lost hit — one occupant per content is returned, and it may be someone else's | found; the walk skips occupants this namespace does not hold |

The second is the one that costs reuse rather than work: the same
document cached under a different prefix is a *different* `chunk_hash`,
so it is a second occupant, and returning only the first silently drops
it.

The earlier design rejected filtering on the grounds that "the first
storer's tenant would get pinned and others could never match their own
copies." That is a consequence of treating a content entry as
single-tenant. A content entry is inherently multi-tenant: the fix is to
filter by *membership* — every namespace that stored the chunk — not by
first-writer identity.

**Residual.** Ranks and object groups within one namespace are not
tracked individually, so a chunk stored by only some ranks of a world
size still matches and confirms at expansion, as before.

## Structure and the recall property

```
_contents : dict[fingerprint -> _Entry { token_ids, occupants{} }]   # authoritative
_slots    : uint8[2^k]  occupancy filter, 1 where any fingerprint lands
```

`occupants` maps `chunk_hash -> { token_offset, namespaces }`, in
first-indexed order — usually exactly one chunk, claimed by one
namespace. A chunk stored under identical prefixes by two tenants is
**one** occupant with two claims, so the tokens are held once. Cost is
one small tuple per `(chunk, namespace)` pair, reported as `num_claims`
in `stats()`.

A match rolls a `chunk_size` window hash over the query
(`rolling_hash_windows_numba`), gathers `_slots` at every
`probe_stride`-th position in one vectorized op, and resolves the few
survivors through `_contents`. Then it **verifies the query window
against `entry.token_ids` token-for-token** before accepting.

The filter deliberately stores **no identity** — just "something lands
here". That is what makes recall complete: two fingerprints sharing a
slot both pass the filter and the dict resolves each correctly. A
direct-address table mapping slot → entry (what the local matcher and the
old blend directory use) instead lets the later writer evict the earlier
one from its slot, making the loser unmatchable. At that design's own
growth threshold the load factor sits in 12.5–25%, so **6–12% of indexed
chunks are silently unmatchable** — measured at 7 of 121 reference
matches lost on a 60-entry table. Since recall *is* reuse, the filter
design trades a rare wasted dict lookup for keeping all of it.

The load factor is therefore only a false-positive rate: `_TABLE_GROWTH`
keeps it under ~6%, costing ~16 bytes per indexed chunk.

Verification is what makes this safe. A fingerprint collision between
different contents means the second content is never added, so it stays
undiscoverable — never wrongly matched.

## Lifecycle

Driven entirely by binding lifecycle in `KeyDirectory`:

| directory event | index |
| --- | --- |
| `STORE` with `token_ids` | `add(tokens, chunk_hash, token_offset, ns)` |
| `STORE` first filling a binding's content | `add` for **every** namespace on the binding |
| re-`STORE` with different content | `remove_chunk`, then re-`add` per namespace |
| `DELETE` of a namespace's last key for a chunk | `remove(tokens, chunk_hash, ns)` |
| `DELETE` of a chunk's last placement overall | last `remove` drops the occupant |
| `fence_instance` (restart / deregistration) | `remove` per dropped chunk |

The namespace comes from the key the event carries, so nothing new is
published for it. The steady-state store path claims for one namespace
per entry; only the two content-changing rows walk the binding's keys, so
per-rank stores do not rehash the content once per rank.

A chunk whose `STORE` carried no tokens (the emitter's bound cache had
already evicted it) is simply not indexed — a lookup miss, repaired by
the chunk's next stamped store. A **key** arriving without tokens for a
chunk whose content is already known is different: it claims the chunk
for its namespace immediately, so a second tenant storing content the
fleet already holds is matchable without waiting for a re-store. Content whose length is not the fleet
`chunk_size` is not indexed either: it can never fill a `chunk_size`
match window. Both are logged, never errors.

**Locking.** The index owns its lock and never calls back into the
directory, so the only order is directory → index. `match` takes the
index lock alone, so fragment queries — which *are* on the serving path,
unlike the rest of the directory — do not serialize behind event
application.

## HTTP surface

`POST /directory/blend-lookup` takes `tokens_b64` (base64 little-endian
`uint32`, ~1.4x smaller than a JSON list and decoded in one
`np.frombuffer`) plus the caller's `model_name`, `world_size`, and
`cache_salt`.

Those three are the same fields `/directory/lookup`'s tokens form
carries, for a related but distinct reason: prefix lookup uses them to
*build* the keys it looks up, while a fragment match already names a
stored `chunk_hash` and uses them to stay inside the namespace the caller
can retrieve from. The token encodings deliberately differ for now — a
fragment query is the request's whole sequence, so the compact form earns
its place — and the two endpoints are free to converge later.

It returns matches as `(chunk_hash, old_st, cur_st)`:

- `old_st` — where the chunk sat in the sequence it was stored under
  (the re-RoPE source).
- `cur_st` — where its content was found in the query (the re-RoPE
  target).

Matches are ascending by `cur_st`, at most one per chunk. **They may
overlap in the query**, since dedup is per chunk, not per query range —
two matches over the same tokens cannot both scatter, so callers apply
their own overlap resolution (blend uses leftmost-greedy).

The rolling hash walks the whole query, so the handler runs it in a
thread rather than on the event loop.

## Deliberately out of scope

- **Sub-chunk granularity.** Reuse is whole-chunk and only where content
  is chunk-phase aligned, matching the algorithm blend uses today.
  Block-level and token-level refinements (minimizer anchors,
  seed-and-extend) trade coordinator memory for finer, offset-robust
  reuse; see the future-evolution table in the blend design notes.
- **Placement-aware ranking.** Matches do not yet prefer a peer holding
  the chunk in L1 over a shared-L2 copy, though the directory knows both.
- **Liveness beyond the binding.** A claim says some instance stored the
  chunk in that namespace, not that the bytes are reachable *now* — the
  directory is eventually consistent, so a match is still a hint the
  owner validates. Filtering on placements would tighten this, at the
  cost of taking the directory lock on the serving path (see
  [key_directory.md](key_directory.md)).
