# L2 Keys Management HTTP API

## Overview

Two HTTP endpoints, both auto-discovered out of
`lmcache/v1/multiprocess/http_apis/l2_keys_api.py`:

- `POST /l2/keys:evict` — delete a caller-supplied list of keys.
- `GET /l2/keys` — paginate keys currently resident in L2, filtered by
  `cache_salt` and/or `model_name`.

Both endpoints operate on the **primary** L2 adapter — the first
adapter configured in the storage manager's adapter list. There is no
adapter selector on the wire. A deployment that wants these endpoints
to target a specific adapter must configure that adapter first.

These endpoints serve operator + admin workflows: "purge this user's
keys," "show me what's resident in L2," "garbage-collect orphans after
a deployment rename." They are NOT in the hot-path read/write flow.

---

## Surface Area

### Python

```python
# StorageManager
def evict_l2_keys(self, keys: list[ObjectKey]) -> dict[str, object]
def list_l2_keys(
    self,
    cache_salt: str | None = None,
    model_name: str | None = None,
    page_size: int = 500,
    page_token: str | None = None,
) -> L2KeyListPage

# L2AdapterInterface  (NEW abstract method)
def list_l2_keys(
    self,
    cache_salt: str | None = None,
    model_name: str | None = None,
    page_size: int = 500,
    cursor: str | None = None,
) -> L2KeyListPage
    # Default: raises NotImplementedError. S3L2Adapter overrides.
```

### New dataclasses (in `l2_adapters/base.py`)

```python
@dataclass(frozen=True)
class L2KeyEntry:
    key: ObjectKey
    size_bytes: int
    adapter_name: str = ""   # filled in by StorageManager

@dataclass(frozen=True)
class L2KeyListPage:
    entries: tuple[L2KeyEntry, ...]
    next_page_token: str | None   # None ⇒ listing exhausted
```

### HTTP

```
POST /l2/keys:evict
Body:  {"keys": [{"chunk_hash_hex": "...", "model_name": "...",
                  "kv_rank": <int>, "cache_salt": "<opt>"}, ...]}
200:   {"requested": N, "adapter": "<type>", "ok": <bool>,
        "error": "<opt>"}
400:   malformed body / per-key validation error
503:   engine not initialized OR no L2 adapters configured

GET /l2/keys
Query: cache_salt=<str>  use "_default" sentinel for the empty salt
       model_name=<str>
       page_size=<int 1..5000>   (default 500)
       page_token=<opaque str>   (omit on first call)
200:   {"entries": [{...key fields..., "size_bytes": N,
                     "adapter": "<type>"}, ...],
        "next_page_token": "<opaque>" | null}
400:   invalid filter / malformed page_token
501:   primary adapter does not implement listing
503:   engine not initialized OR no L2 adapters configured
```

The response carries the adapter's type name in the `"adapter"` field
(`evict`) or per-entry (`list`), so operators always know which
adapter answered.

---

## Eviction Semantics

### Single target, idempotent

`evict_l2_keys` reads `self._l2_adapters[0]` and calls its
`delete(keys)`. No selection logic, no fan-out. Idempotent:
re-evicting an already-deleted key is harmless — the adapter filters
keys it doesn't have or that are locked by an in-flight operation.

### Failure shape: in-body, not 5xx

Best-effort: an exception from `adapter.delete` is caught, logged via
`logger.exception`, and reported in the response body as
`{"adapter": "...", "ok": False, "error": "..."}`. The HTTP status is
still 200 — the call reached the right adapter and got a determinate
outcome.

Rationale: today's adapters (S3) already catch their own I/O
exceptions and log warnings without raising, so this branch is
defensive — but when it does fire (e.g. a future adapter), surfacing
the per-call detail in JSON is more useful than a generic 500.

### Reuses existing `L2AdapterInterface.delete()`

No new per-adapter eviction method was added. Adapters that override
`delete` (S3) already handle their own in-flight-lock checks and fire
`on_l2_keys_deleted` to listeners. Adapters that don't override
`delete` (the default no-op in `L2AdapterInterface`) silently succeed
with `ok: True`.

### L1 is intentionally NOT touched

Eviction operates on L2 only. Keys evicted from L2 may still return
from L1 until natural L1 eviction expires them. This keeps the API
contract narrow ("evict L2") and avoids accidentally invalidating L1
entries that other in-flight requests are reading.

---

## Listing Semantics

### v1 scope: S3 only

Only `S3L2Adapter` implements `list_l2_keys` in v1. When the primary
adapter is anything else, the endpoint returns 501. Future PRs can
opt additional adapters in by overriding the method; no
`StorageManager` changes are needed.

### S3 listing is served from in-memory accounting

S3's `list_l2_keys` reads from `self._key_sizes` — the
`dict[ObjectKey, int]` the adapter maintains alongside its byte
accounting. This is the same source `_notify_keys_stored` /
`_notify_keys_deleted` keep current, so the listing matches exactly
what the adapter considers resident.

The AWS CRT `s3.S3Client` used by the adapter does NOT expose a
low-cost `ListObjectsV2`-style API, so a bucket-side listing would
require falling back to boto3 — out of scope for v1. The in-memory
approach is also cheaper (no S3 RTT) and consistent with how the
adapter answers `get_usage()`.

**Caveat:** keys written to the bucket by a *different* writer (e.g.
another LMCache instance sharing the same prefix) are invisible to
this listing. This matches the rest of the adapter's semantics — that
writer also doesn't show up in `get_usage()` byte counts.

### Stable ordering

For a fixed filter, keys are sorted by `(cache_salt, model_name,
kv_rank, chunk_hash)`. This ordering is:

- **stable** across calls when the underlying set doesn't change
  (snapshot taken under `self._lock`, sort applied off-lock so a
  paginated walk doesn't block writers);
- **deterministic** so the wire-level cursor (an offset into the
  sorted list) converges to "end-of-list" instead of looping.

If stores/evictions mutate the key set between pages, individual keys
MAY appear, disappear, or shift between pages. The contract is
best-effort consistency, not snapshot isolation. Operator workflows
that need an exact snapshot should quiesce writes first.

### Pagination

Because each call targets one adapter, the wire `page_token` is just
the adapter's own cursor passed through verbatim — no envelope
encoding. For the S3 adapter today, that cursor is a stringified
integer offset into the sorted list.

Callers MUST still treat the token as opaque: a future adapter might
use a different cursor shape (e.g. base64 of `LastEvaluatedKey` for a
DynamoDB-backed L2), and the storage manager makes no commitments
about format stability beyond "pass it back verbatim."

`page_size` is clamped to `[1, 5000]` at the HTTP layer and to
`> 0` at the StorageManager layer. Default 500 — chosen to keep a
single response under typical HTTP body soft-limits even with verbose
keys.

---

## Contract Table

| Requirement | Where enforced |
|---|---|
| Empty cache_salt expressible in URL | `_DEFAULT_SALT_SENTINEL = "_default"` in `l2_keys_api.py` |
| `chunk_hash_hex` is valid hex | `bytes.fromhex` in `_parse_object_key` raises `ValueError` |
| `model_name` / `cache_salt` invariants (no `@`, etc.) | `ObjectKey.__post_init__` |
| Per-request eviction batch cap | `_MAX_EVICT_BATCH = 10_000` in `l2_keys_api.py` |
| `page_size` bounds | `Query(ge=1, le=_MAX_PAGE_SIZE)` |
| Listing returns stable order under fixed filter | sort by `(cache_salt, model_name, kv_rank, chunk_hash)` |
| Adapter listing snapshot taken under `_lock` | `with self._lock:` in `S3L2Adapter.list_l2_keys` |
| No adapters configured → 503 | endpoint catches `ValueError("no L2 adapters …")` |
| Adapter doesn't support listing → 501 | endpoint catches `NotImplementedError` |
| Adapter delete failure → in-body, not 5xx | `evict_l2_keys` catches per-call exceptions |
| L1 not touched on evict | documented in module + `StorageManager.evict_l2_keys` docstrings |

---

## Caller Impact

### Existing callers of `L2AdapterInterface`

`list_l2_keys` was added as a **non-abstract** method with a default
that raises `NotImplementedError`. All existing concrete L2 adapters
inherit the default — no caller code changes needed.

The new dataclasses (`L2KeyEntry`, `L2KeyListPage`) are additive — no
existing import path moves.

### Existing callers of `StorageManager`

Both new methods are additive. No existing method's signature, return
type, or behavior changed. Test code that constructs a partial
StorageManager via `__new__` is the only path that interacts with the
new methods directly — see
`tests/v1/distributed/test_storage_manager_l2_keys.py`.

### Existing callers of S3L2Adapter

`S3L2Adapter.list_l2_keys` is new. The adapter's existing `delete`,
`get_usage`, store/load paths are unchanged. The `_key_sizes` dict is
already protected by `self._lock`; the new method takes a snapshot
under that same lock so existing lock ordering is preserved.

---

## Test Coverage

- `tests/v1/distributed/test_storage_manager_l2_keys.py` — selection
  + delegation: primary adapter wins, no-adapters raises, adapter
  failures are reported (not raised), `NotImplementedError`
  propagates, filters thread through, `page_token` passes through
  verbatim, secondary adapters never touched. Uses
  `StorageManager.__new__` + stub adapters to bypass the heavy ctor.
- `tests/v1/distributed/test_s3_l2_adapter.py::TestS3L2AdapterListKeys`
  — S3 listing: sort stability, `cache_salt` filter (incl. empty
  string), `model_name` filter, full-set pagination walk,
  `page_size`/`cursor` validation, offset-past-end handling.
- `tests/v1/multiprocess/http_apis/test_l2_keys_api.py` — endpoint
  shape: happy path, in-body failure reporting, 503 on no adapters /
  no engine, 501 on unsupported listing, 400 on malformed body /
  page_token / page_size, `_default` sentinel translation, auto-
  discovery (registry sweep picks up the module).

---

## Future Work (not in this PR)

- Implement `list_l2_keys` on additional adapters that have natural
  enumeration sources (FS, Mooncake, Dax). Until then they remain
  501.
- Per-adapter targeting on the HTTP surface when a deployment runs
  multiple L2 adapters and wants to address each by `type_name` or
  descriptor index.
- Optional `prefix` / `model_name_glob` filters once a real caller
  needs them.
- A `DELETE /l2/keys?cache_salt=...` convenience that combines
  listing + evicting for a whole tenant in one call (currently the
  caller pages through `GET /l2/keys` then issues
  `POST /l2/keys:evict`).
- Counter-based snapshot tokens so pagination across concurrent
  mutations is fully consistent (currently best-effort).
