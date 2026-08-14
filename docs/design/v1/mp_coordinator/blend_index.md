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

The cost is coordinator memory: verification needs the tokens resident,
`O(m)` rather than `O(m/C)`. See
[key_directory.md](key_directory.md) — Token index for the
representation that keeps that affordable.

## Identity and scope

Entries are keyed by a **content fingerprint only** — the 64-bit
polynomial hash of one chunk's tokens (`POLY_BASE`, fleet-constant).
Deliberately absent:

- **No model, salt, or rank.** A match names a `chunk_hash`; the querying
  server expands it into `ObjectKey`s with **its own** model, salt, and
  world size, exactly as the local blend path does. A cross-model or
  cross-tenant match therefore lands in the requester's own namespace and
  confirmed-misses at prefetch. Filtering by the *storer's* identity
  would be wrong: content is shared first-writer-wins, so the first
  storer's tenant would get pinned and others could never match their own
  copies.
- **No prefix.** `chunk_hash` is prefix-chained, so the same text stored
  after different prefixes is two chunks; both attach to **one** content
  entry (each with its own `token_offset`), so evicting one leaves the
  other discoverable.

## Structure and the recall property

```
_contents : dict[fingerprint -> _Entry { token_ids, occupants[] }]   # authoritative
_slots    : uint8[2^k]  occupancy filter, 1 where any fingerprint lands
```

`occupants` is `(chunk_hash, token_offset)` per chunk holding that
content — usually exactly one.

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
| `STORE` with `token_ids` | `add(tokens, chunk_hash, token_offset)` |
| re-`STORE` with different content | old fingerprint removed, new added |
| `DELETE` of a chunk's last placement | `remove(tokens, chunk_hash)` |
| `fence_instance` (restart / deregistration) | `remove` per dropped chunk |

A chunk whose `STORE` carried no tokens (the emitter's bound cache had
already evicted it) is simply not indexed — a lookup miss, repaired by
the chunk's next stamped store. Content whose length is not the fleet
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
`np.frombuffer`) and returns matches as
`(chunk_hash, old_st, cur_st)`:

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
- **Cross-model filtering.** Left to the requester's key expansion, as
  above.
