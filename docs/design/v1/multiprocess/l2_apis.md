# L2 HTTP APIs

## Overview

Two HTTP endpoints, both auto-discovered out of
`lmcache/v1/multiprocess/http_apis/l2_api.py`:

- `DELETE /l2` — delete the KV cache for a caller-supplied list of
  keys (the keys are addresses; the cached bytes are what gets removed).
- `GET /l2/keys` — paginate keys currently resident in L2, optionally
  filtered by `model_name`.

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
def delete_l2(self, keys: list[ObjectKey]) -> dict[str, object]
def list_l2_keys(
    self,
    model_name: str | None = None,
    page_size: int = 500,
    page_token: str | None = None,
) -> dict[str, object]
    # Returns {"adapter": <type_name>, "entries": tuple[KeyEntry, ...],
    #          "next_page_token": <opaque> | None}

# L2AdapterInterface  (NEW abstract method)
def list_l2_keys(
    self,
    model_name: str | None = None,
    page_size: int = 500,
    cursor: str | None = None,
) -> KeyListPage
    # Default: raises NotImplementedError. S3L2Adapter overrides.
```

### New dataclasses (in `distributed/api.py`)

```python
@dataclass(frozen=True)
class KeyEntry:
    key: ObjectKey
    size_bytes: int

@dataclass(frozen=True)
class KeyListPage:
    entries: tuple[KeyEntry, ...]
    next_page_token: str | None   # None ⇒ listing exhausted
```

### HTTP

```
DELETE /l2
Body:  {"keys": [{"chunk_hash_hex": "...", "model_name": "...",
                  "kv_rank": <int>, "cache_salt": "<opt>"}, ...]}
200:   {"requested": N, "adapter": "<type>", "ok": <bool>,
        "error": "<opt>"}
400:   key payload violates ``ObjectKey`` invariants (bad hex,
       ``@`` in ``model_name``, etc.)
422:   Pydantic-level body shape failure
503:   engine not initialized OR no L2 adapters configured

GET /l2/keys
Query: model_name=<str>     (optional)
       page_size=<int 1..5000>   (default 500)
       page_token=<opaque str>   (omit on first call)
200:   {"adapter": "<type>",
        "entries": [{"key": <EncodedObjectKey>, "size_bytes": N}, ...],
        "next_page_token": "<opaque>" | null}
400:   invalid filter / malformed page_token
501:   primary adapter does not implement listing
503:   engine not initialized OR no L2 adapters configured
```

Both responses carry the adapter's type name in a top-level
`"adapter"` field so operators always know which adapter answered.
The `GET /l2/keys` response reports it once per page (not per entry):
every entry on a given page is from the primary adapter by
construction, so per-entry tagging would just duplicate that one
string N times.

---

## Eviction Semantics

### Single target, idempotent

`delete_l2` reads `self._l2_adapters[0]` and calls its
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

### S3 listing is served from S3 itself, via `ListObjectsV2`

The adapter issues a real `ListObjectsV2` request against the bucket
on every page call. The response XML is parsed into `(ObjectKey, size)`
pairs via :func:`_string_to_object_key` (the inverse of the adapter's
key serializer), and S3's `NextContinuationToken` becomes the next
wire `page_token`.

Rationale: the in-memory `_key_sizes` tracker only knows what *this*
LMCache instance has stored since startup. Operators running multiple
instances against the same bucket, or restarting an instance, need a
listing that reflects what's actually on S3 — not just this process's
write log.

Costs:
- **One S3 RTT per page** (vs. zero for an in-memory walk).
- Server-side prefix filter on `model_name` (when set) lets S3 skip
  irrelevant keys.
- `MaxKeys` is capped at 1000 by S3, so even when a caller requests
  `page_size=5000` the adapter clamps internally and returns at most
  1000 entries per call — the caller continues via the token.

### Filtering

The only supported filter is **`model_name`**, pushed down as
`prefix=<flattened_model_name>@`. Flattening (`/` → `_`) is applied
so the prefix matches the form `_format_safe_path` stored on S3.
`cache_salt` is intentionally NOT a filter parameter — it sits at the
*end* of the key and can't be expressed as an S3 prefix, so filtering
it would only narrow client-side without reducing the RTTs. If a
future caller needs per-tenant scoping, the simplest path is a
client-side filter on the response.

### Pagination

The wire `page_token` is S3's `NextContinuationToken`, passed through
verbatim by `StorageManager.list_l2_keys`. Callers MUST treat it as
opaque — it's a base64-ish string whose format is owned by S3.

When `IsTruncated` is `false` in the response, the adapter returns
`next_page_token=None` and the listing is complete.

### Cross-instance visibility

Because the listing is bucket-side, keys written by other LMCache
instances sharing the same prefix DO appear. Keys written by other
tools (anything whose object name doesn't conform to
`<model>@<rank>@<group>@<hash>[@<salt>]`) are silently dropped from
the response — `_string_to_object_key` raises `ValueError`, and the
parser skips entries it can't decode.

### `/` in `model_name` is stored verbatim

The adapter stores objects under their literal `_object_key_to_string`
output — no path-flattening. `_format_safe_path` only URL-encodes the
HTTP path (`/` stays as `/` in the URL, `@` becomes `%40`). S3 accepts
`/` in object keys as a legal character (it's only the AWS console
that treats `/` as a virtual-folder delimiter — purely cosmetic).

Round-trip example:

```
ObjectKey(model_name="meta-llama/Llama-3.1-8B", ...)
 → stored on S3 as literal "meta-llama/Llama-3.1-8B@..."
 → listed back as ObjectKey(model_name="meta-llama/Llama-3.1-8B", ...)
```

Operators pass HF model ids (with `/`) to the `model_name=` filter on
`GET /l2/keys` exactly as they appear in their config — the adapter
forwards them straight to S3's `prefix=` query param.

### Consistency

S3 `ListObjectsV2` is strongly consistent for new keys (read-after-write)
but offers no snapshot guarantees across paged calls — keys written
or deleted between calls may appear, disappear, or shift positions.
The contract is best-effort. Operator workflows that need an exact
snapshot should quiesce writes first.

`page_size` is clamped to `[1, 5000]` at the HTTP layer and to
`[1, 1000]` at the S3 adapter layer (S3's `MaxKeys` ceiling). Default
500 — chosen to keep a single response under typical HTTP body
soft-limits even with verbose keys.

---

## Contract Table

| Requirement | Where enforced |
|---|---|
| `chunk_hash_hex` is valid hex | `bytes.fromhex` in `EncodedObjectKey.to_object_key` raises `ValueError` |
| `model_name` / `cache_salt` invariants (no `@`, etc.) | `ObjectKey.__post_init__` |
| Per-request eviction batch cap | `_MAX_DELETE_BATCH = 10_000` in `l2_api.py` |
| `page_size` bounds | `Query(ge=1, le=_MAX_PAGE_SIZE)` |
| Listing returns lex order owned by S3 | S3's `ListObjectsV2` |
| No adapters configured → 503 | endpoint catches `ValueError("no L2 adapters …")` |
| Adapter doesn't support listing → 501 | endpoint catches `NotImplementedError` |
| Adapter delete failure → in-body, not 5xx | `delete_l2` catches per-call exceptions |
| L1 not touched on evict | documented in module + `StorageManager.delete_l2` docstrings |

---

## Caller Impact

### Existing callers of `L2AdapterInterface`

`list_l2_keys` was added as a **non-abstract** method with a default
that raises `NotImplementedError`. All existing concrete L2 adapters
inherit the default — no caller code changes needed.

The new dataclasses (`KeyEntry`, `KeyListPage`) are additive — no
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
  — S3 listing: `model_name` prefix push-down, pagination walk via
  continuation tokens, `page_size` clamp to S3's MaxKeys ceiling,
  circuit-breaker rejection, silent skipping of objects whose names
  don't conform to this adapter's key format.
- `tests/v1/multiprocess/http_apis/test_l2_api.py` — endpoint
  shape: happy path, in-body failure reporting, 503 on no adapters /
  no engine, 501 on unsupported listing, 400 on malformed body /
  page_token / page_size, auto-discovery (registry sweep picks up
  the module).

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
- A `DELETE /l2?model_name=...` (or equivalent body filter) convenience
  that combines listing + deletion for a whole model in one call
  (currently the caller pages through `GET /l2/keys` then issues
  `DELETE /l2`).
- Counter-based snapshot tokens so pagination across concurrent
  mutations is fully consistent (currently best-effort).
