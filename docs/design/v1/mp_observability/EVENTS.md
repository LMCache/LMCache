# Event Metadata Contracts

Each `EventType` has a documented metadata schema.  Producers **must** populate
these keys; subscribers **may** rely on them being present.

For the full list of event types see `event.py`.  For metrics derived from
these events see [METRICS.md](METRICS.md).

## Tenant tagging (`cache_salt`)

Every observability metric and span is tagged with a `cache_salt` attribute
identifying the tenant the event belongs to (one `cache_salt` == one user).
The salt is carried either:

- **Directly on each `ObjectKey`** — L1Manager, StorageManager, and the L2
  store controller publish `metadata["keys"]` / `metadata["succeeded_keys"]`
  / `metadata["failed_keys"]` as `list[ObjectKey]`; subscribers read
  `key.cache_salt` off each entry and group their emissions accordingly.
- **As an explicit `cache_salt: str` metadata key** — MP server events and
  L2 prefetch events are request-scoped (one salt per event), so they carry
  a single string value.

An empty string (`""`) is a valid salt value representing an unsalted
request and appears on metrics as `cache_salt=""`.

---

## L1Manager Events

| EventType | Metadata keys | Types |
|---|---|---|
| `L1_READ_RESERVED` | `keys` | `list[ObjectKey]` |
| `L1_READ_FINISHED` | `keys` | `list[ObjectKey]` |
| `L1_WRITE_RESERVED` | `keys` | `list[ObjectKey]` |
| `L1_WRITE_FINISHED` | `keys` | `list[ObjectKey]` |
| `L1_WRITE_FINISHED_AND_READ_RESERVED` | `keys` | `list[ObjectKey]` |
| `L1_KEYS_EVICTED` | `keys` | `list[ObjectKey]` |

Each `ObjectKey` carries `cache_salt`; subscribers group metric emissions by
`key.cache_salt`.

---

## StorageManager Events

| EventType | Metadata keys | Types |
|---|---|---|
| `SM_READ_PREFETCHED` | `succeeded_keys`, `failed_keys` | `list[ObjectKey]`, `list[ObjectKey]` |
| `SM_READ_PREFETCHED_FINISHED` | `succeeded_keys`, `failed_keys` | `list[ObjectKey]`, `list[ObjectKey]` |
| `SM_WRITE_RESERVED` | `succeeded_keys`, `failed_keys` | `list[ObjectKey]`, `list[ObjectKey]` |
| `SM_WRITE_FINISHED` | `succeeded_keys`, `failed_keys` | `list[ObjectKey]`, `list[ObjectKey]` |

`cache_salt` flows off each `ObjectKey`.

---

## L2 Store Controller Events

| EventType | Metadata keys | Types |
|---|---|---|
| `L2_STORE_SUBMITTED` | `adapter_index`, `key_count`, `keys` | `int`, `int`, `list[ObjectKey]` |
| `L2_STORE_COMPLETED` | `adapter_index`, `succeeded_count`, `failed_count`, `succeeded_keys`, `failed_keys` | `int`, `int`, `int`, `list[ObjectKey]`, `list[ObjectKey]` |

`cache_salt` flows off each `ObjectKey` in the key lists.

---

## L2 Prefetch Controller Events

Each prefetch request belongs to a single tenant; `cache_salt` is a single
string, copied from the originating request's keys.

| EventType | Metadata keys | Types |
|---|---|---|
| `L2_PREFETCH_LOOKUP_SUBMITTED` | `request_id`, `key_count`, `adapter_count`, `cache_salt` | `int`, `int`, `int`, `str` |
| `L2_PREFETCH_LOOKUP_COMPLETED` | `request_id`, `prefix_hit_count`, `cache_salt` | `int`, `int`, `str` |
| `L2_PREFETCH_LOAD_SUBMITTED` | `request_id`, `key_count`, `adapter_count`, `cache_salt` | `int`, `int`, `int`, `str` |
| `L2_PREFETCH_LOAD_COMPLETED` | `request_id`, `loaded_count`, `failed_count`, `cache_salt` | `int`, `int`, `int`, `str` |

---

## MP Server Lifecycle Sentinels

CPU-synchronous sentinels published by `server.py` to bracket request scope.
Published via `EventBus.publish()` (not `publish_on_stream`) so the drain
thread processes them in strict order before any GPU-callback events.

| EventType | Metadata keys | Types | Published by / when |
|---|---|---|---|
| `MP_REQUEST_START` | `cache_salt` | `str` | `MPServer.handle_request` — at request arrival, before any GPU work |
| `MP_STORE_SUBMITTED` | `device`, `cache_salt` | `str`, `str` | `MPServer.store` — CPU-synchronous, before the GPU store is enqueued |
| `MP_RETRIEVE_SUBMITTED` | `device`, `cache_salt` | `str`, `str` | `MPServer.retrieve` — CPU-synchronous, before the GPU retrieve is enqueued |
| `MP_SESSION_END` | `cache_salt` | `str` | `MPServer.handle_request` — after all CPU work; may precede GPU callbacks |

---

## MP Server Events

These events use `session_id` on the `Event` dataclass (not in `metadata`)
to correlate START/END pairs.

| EventType | Metadata keys | Types |
|---|---|---|
| `MP_STORE_START` | `device`, `cache_salt` | `str`, `str` |
| `MP_STORE_END` | `device`, `stored_count`, `cache_salt` | `str`, `int`, `str` |
| `MP_RETRIEVE_START` | `device`, `cache_salt` | `str`, `str` |
| `MP_RETRIEVE_END` | `device`, `retrieved_count`, `cache_salt` | `str`, `int`, `str` |
| `MP_LOOKUP_PREFETCH_START` | `cache_salt` | `str` |
| `MP_LOOKUP_PREFETCH_END` | `found_count`, `cache_salt` | `int`, `str` |
| `MP_LOOKUP` | `request_id`, `chunk_hashes`, `model_name`, `chunk_size`, `seq_len`, `dtypes`, `shapes`, `cache_salt` | `str`, `list[str]`, `str`, `int`, `int`, `list[str]`, `list[list[int]]`, `str` |
| `MP_VLLM_BLOCK_ALLOCATION` | `instance_id`, `model_name`, `records` | `int`, `str`, `list[BlockAllocationRecord]` (each has `req_id: str`, `new_block_ids: list[int]`, `new_token_ids: list[int]`, `cache_salt: str`) |
| `MP_VLLM_END_SESSION` | `request_id`, `cache_salt` | `str`, `str` |

---

## Trace Recording Events

A single unified event used by the `@enable_tracing` decorator (see
[trace.md](trace.md)). All instrumented call sites publish the same
`EventType` regardless of which method or layer; the `qualname` field
inside `metadata` discriminates ops.

| EventType | Metadata keys | Types |
|---|---|---|
| `TRACE_CALL` | `qualname`, `args` | `str`, `dict[str, Any]` (codec-encoded; see `lmcache.v1.mp_observability.trace.codecs`) |
