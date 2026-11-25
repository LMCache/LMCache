# Request index

Module: `lmcache/v1/mp_coordinator/views/request_index.py`
HTTP surface: `lmcache/v1/mp_coordinator/http_apis/requests_api.py`
(`GET /requests/{instance_id}/{request_id}`)

Given a request id, the request index recovers the request's token
sequence and tells you where each of its KV chunks lives. It keeps the
tokens after the KV has been evicted.

## Why

The [key directory](key_directory.md) answers "where is this key". It has
no notion of the request that produced a key, and it drops a chunk's token
content when the last placement goes. Debugging a cache miss, replaying a
request, or reloading its KV needs the reverse lookup: request id to
ordered chunks, tokens and placements.

## What reaches the coordinator

The MP server already knows the request when it submits a store. It
publishes `MP_TOKENS` with `session_id = request_id`, and for each chunk
it includes the token ids, the token offset, the hash of the preceding
chunk and the chunk's `ObjectKey`. The cache-event subscriber stamps two
new fields from that binding onto every `STORE` entry of the chunk:

| `CacheEventEntry` field | Meaning | Unknown value |
|---|---|---|
| `request_ids` | Requests on the emitting instance whose store submitted the chunk | `[]` |
| `parent_hash_hex` | Hex chunk hash of the chunk before it in the sequence | `""` (also a sequence's first chunk) |

`request_ids` is a list because two requests can submit the same chunk
concurrently. The subscriber's binding cache is keyed by chunk hash, so a
single field would keep only the later request. The list keeps the 8 most
recent requests, so a chunk that many requests re-submit does not grow
every `STORE`.

Chunk hashes depend on tokens alone, so two tenants (`cache_salt`) or
models storing the same tokens share one binding. The subscriber records
each request with its namespace (model, salt, world size, tags, taken
from the `ObjectKey`) and stamps a request id only on `STORE` entries of
keys in that namespace. The index likewise lists only the request's own
namespace's keys, so one tenant's request never shows another tenant's
placements.

### Requests served from cache

A request whose whole prompt is cached stores nothing, so it emits no
`MP_TOKENS`. Its lookup reports it instead. When the lookup module hands a
finished lookup to the session (the status poll), it publishes
`MP_LOOKUP_HIT_TAIL` with the key of the last full chunk hit and its
offset. The hit counts engine-covered chunks too, which LMCache never
checked; a tail the index does not hold answers 404. Early exits and
lookups the load policy declined report nothing. The extra work (one
`ObjectKey` per lookup) runs only when something subscribes.

The subscriber keeps each request's latest hit and sends them on the next
flush as `lookups` on an `access` batch with no entries:

```json
{"event_type": "access", "tier": "l1", "backend": "", "entries": [],
 "lookups": [{"request_id": "cmpl-7-0",
              "key": {"chunk_hash_hex": "cd34...", "...": "..."},
              "token_offset": 256}]}
```

It is a batch-level field rather than a new event type on purpose. A
coordinator rejects a whole `POST /events` (422) that names an event type
it does not know, which would lose the `STORE`s in the same flush, but it
ignores a field it does not know. With no entries, the key directory and
the usage view count no access; the storage layer already reports hits
as `ACCESS` entries.

Adding these fields bumps `CACHE_EVENT_SCHEMA_VERSION` to 2. All of them
default to empty, and the coordinator ignores fields it does not know,
so MP servers and the coordinator can upgrade in either order. The `events` trace replayer requires an exact schema match, so it
refuses trace files recorded at version 1.

## Recovery

Chunk hashes are prefix-chained, so a chunk names its whole prefix. The
index stores:

```
chunk_hash -> (token_ids, token_offset, parent_hash, keys)  # bounded LRU
(instance_id, request_id) -> (tail chunk_hash, namespace)  # bounded LRU
```

The tail is the request's chunk with the highest `token_offset`, stored
or hit. That is
not necessarily the last one to arrive, because one chunk yields several
`STORE`s (L1, async L2, every rank and object group) in no fixed order.
Recovery walks `parent_hash` back from the tail until it reaches offset 0
or a chunk the index no longer holds:

```
req-2 tail = C ──parent──► B ──parent──► A (offset 0)
tokens     = A.tokens + B.tokens + C.tokens
placements = KeyDirectory.lookup(A.keys + B.keys + C.keys)
```

Prefix chunks stored by an earlier request (cache hits, here A and B) are
recovered too, because the walk follows content links rather than request
ids.

`GET /requests/{instance_id}/{request_id}` returns:

```json
{
  "instance_id": "mp-0",
  "request_id": "cmpl-42-0",
  "start_offset": 0,
  "complete": true,
  "token_ids": [101, 102, "..."],
  "chunks": [
    {
      "chunk_hash_hex": "ab12...",
      "token_offset": 0,
      "num_tokens": 256,
      "keys": [{"key": {"chunk_hash_hex": "ab12...", "...": "..."},
                "placements": [{"instance_id": "mp-0", "tier": "l1", "...": "..."}]}]
    }
  ]
}
```

- `complete` is `true` when the chunks reach back to offset 0. It means
  the chain is unbroken, not that the whole prompt came back: a request
  that hit part of its prompt and stored nothing returns only the hit
  prefix. When an ancestor chunk has aged out of the index, `start_offset`
  is the first recovered token's position and `token_ids` starts there.
- A key with empty `placements` has been evicted. Its tokens are still
  returned. Reload it through the existing prefetch or retrieve path.
- An unknown request answers 404.

The coordinator returns placements only, never KV bytes. It stays a
control plane, and every placement is a hint to be validated at the owner,
as with directory lookups.

`instance_id` is the MP server's id. vLLM request ids are unique only per
engine, so a request is addressed by the pair.

## Bounds and settings

Token ids cost about 1 KiB per 256-token chunk; a tracked request costs
a few hundred bytes. The two tables have separate LRU bounds, set through
`extra_config`:

| Key | Default | Covers |
|---|---|---|
| `request_index_max_chunks` | 65536 | about 64 MiB of token content |
| `request_index_max_requests` | 1048576 | about 17 minutes of requests at 1000/s |

Requests get the larger bound because every request with a hit takes a
slot, while chunks are shared.

```bash
lmcache coordinator --extra-config '{"request_index_max_chunks": 262144, "request_index_max_requests": 4194304}'
```

Retention is by recency of store or hit, not by placement: evicting a
chunk's KV does not remove its tokens. Storing or hitting a chunk also
refreshes its retained ancestors, root first. A popular system prompt is
a cache hit for later requests and is never stored again, so without this
refresh it would be the first chunk to age out, and every request built
on it would come back incomplete.

## Limitations

- **Lookup reports are best effort.** A flush that fails drops its
  lookups along with its batches. They are not counted in
  `dropped_events`, which counts entries.
- **Only the LMCache-driven store path reports.** Engine-driven transfer,
  blend pre-computed documents and the qstore plugin module emit no
  `MP_TOKENS`, so their stores record nothing, and a hit on chunks only
  they stored answers 404.
- **The trailing partial chunk is lost** unless `save_unfull_chunk` stores
  it, because only complete chunks carry token bindings.
- **Not checkpointed.** The index rebuilds from the event stream. A
  coordinator restart without durable replay starts it empty.
- **Emitter binding cache.** A store that completes after its binding
  aged out of the subscriber's cache (`_TOKEN_BINDING_CACHE_SIZE`) carries
  no tokens or request ids, and the index skips it. Conversely, while a
  binding is cached, a later re-store of the chunk (after its KV was
  evicted) carries the earlier request ids again. That only refreshes the
  request's entry, unless the request has already aged out of the index,
  in which case it comes back holding only this chunk's prefix.

## Planned: eviction count per chunk

The retained per-chunk record is where a future eviction counter will
live. Two things need settling first. `DELETE` covers both eviction and an
explicit `POST /cache/delete`. Fencing a restarted instance drops its L1
placements with no `DELETE` at all.
