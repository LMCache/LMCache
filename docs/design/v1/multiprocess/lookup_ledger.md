# Lookup Lock Ledger (Multiprocess Mode)

Covers `lmcache/v1/multiprocess/lookup_ledger.py` and its relationship to
`lmcache/v1/multiprocess/session.py`.

## 1. Problem

A scheduler LOOKUP reserves one L1 read lock per KV worker (or one reader
count per TP worker for MLA) on every key it prefetched. Those locks are
anonymous refcounts, so a later `FREE_LOOKUP_LOCKS`, `END_SESSION`, or a failed
RETRIEVE can only release them by reconstructing *exactly* what the lookup
locked: the hit length, the locked object groups, each group's sliding window,
and the retained L1 owners.

That reconstruction state used to live on `Session`, next to the token sequence
and its chunk hashes. Three consequences made this the wrong home:

- **Lifetime mismatch.** A session exists to remember tokens and hashes and is
  dropped on `END_SESSION` or after its TTL. The lock record has to survive
  until the locks are actually released. Storing it on the session meant
  "session is gone" silently became "no locks to release".
- **Ownership blur.** `Session.extras: dict[str, Any]` was an untyped bag that
  any module could write to (blend stashed its sparse reservation there, the
  engine-driven transfer stashed timing there), so no single owner could answer
  what a session held or who released it.
- **One lock for everything.** The session's mutex guarded both the CPU-bound
  rolling hash and every lock bookkeeping update.

## 2. Design

`LookupLockLedger` is a `request_id -> LookupLockState` map owned by
`MPCacheServerContext` (next to the session manager) with these operations:

| Call | Writer | Purpose |
|---|---|---|
| `ensure(request_id, key)` | `resolve_obj_keys` | Record the worker-less key when object keys are resolved without a lookup (a store), so `END_SESSION` can still touch what was written. Idempotent. |
| `begin_lookup(request_id, key, group_windows)` | `LookupModule.lookup`, blend's prefix leg | Open a new lookup: reset the hit length, locked groups and failed-release claims, and bump the generation. |
| `record_prefetch_result(request_id, hit_chunks, locked_gids, l1_owners)` | `_consume_prefetch_result` | Publish the hit length and lock model once the prefetch resolves. |
| `lock_state(request_id)` | `free_lookup_locks`, `end_session`, failed-retrieve cleanup | Snapshot of the record; callers cannot mutate the ledger through it. |
| `release(request_id)` | `end_session`, the session-destroy listener | Drop the record and return it. |

`prepare_failed_retrieve_release` and `claim_failed_retrieve_release` stayed on
the ledger as atomic pairs: a failed worker may release only its own reader
share, exactly once, and only when the ledger can prove the request owns the
RETRIEVE's range.

## 3. Lifetime

A ledger entry is dropped in exactly one of two ways:

1. `END_SESSION` — `LookupModule.end_session` releases the record *before*
   removing the session, then uses it to touch every key of the request.
2. Session destruction — `MPCacheServerContext` registers a
   `SessionManager.destroy_listeners` callback that releases the record, so an
   abandoned request (no `END_SESSION`, or TTL expiry) does not leave its
   record behind.

Rebinding the chunk size clears the ledger, because hit lengths and group
windows are expressed in chunks.

## 4. What `Session` keeps

`Session` is now only `request_id`, `hasher`, the packed `tokens` and the
`chunk_hashes`/`last_prefix_hash`/`num_chunks_processed` rolling state. Per
request creation time moved to `SessionManager`'s entry, and feature state
moved to its owner: blend's sparse reservation to `BlendModule`, transfer
timing to the engine-driven transfer module.

Invariants worth keeping:

- "No session" no longer implies "no locks". A release must consult the
  ledger, never the session.
- Anything a module stores per request needs its own release path; the
  session-destroy listener is the signal, not the storage.
