# Event Metadata Contracts

Each `EventType` has a documented metadata schema.  Producers **must** populate
these keys; subscribers **may** rely on them being present.

For the full list of event types see `event.py`.  For metrics derived from
these events see [METRICS.md](METRICS.md).

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

---

## StorageManager Events

| EventType | Metadata keys | Types |
|---|---|---|
| `SM_READ_PREFETCHED` | `succeeded_keys`, `failed_keys` | `list[ObjectKey]`, `list[ObjectKey]` |
| `SM_READ_PREFETCHED_FINISHED` | `succeeded_keys`, `failed_keys` | `list[ObjectKey]`, `list[ObjectKey]` |
| `SM_WRITE_RESERVED` | `succeeded_keys`, `failed_keys` | `list[ObjectKey]`, `list[ObjectKey]` |
| `SM_WRITE_FINISHED` | `succeeded_keys`, `failed_keys` | `list[ObjectKey]`, `list[ObjectKey]` |

---

## L2 Store Controller Events

| EventType | Metadata keys | Types |
|---|---|---|
| `L2_STORE_SUBMITTED` | `adapter_index`, `task_id`, `l2_name`, `key_count`, `total_bytes` | `int`, `int`, `str`, `int`, `int` |
| `L2_STORE_COMPLETED` | `adapter_index`, `task_id`, `l2_name`, `succeeded_count`, `failed_count`, `total_bytes` | `int`, `int`, `str`, `int`, `int`, `int` |

---

## L2 Prefetch Controller Events

| EventType | Metadata keys | Types |
|---|---|---|
| `L2_PREFETCH_LOOKUP_SUBMITTED` | `request_id`, `key_count`, `adapter_count` | `int`, `int`, `int` |
| `L2_PREFETCH_LOOKUP_COMPLETED` | `request_id`, `prefix_hit_count` | `int`, `int` |
| `L2_PREFETCH_LOAD_SUBMITTED` | `request_id`, `key_count`, `adapter_count` | `int`, `int`, `int` |
| `L2_PREFETCH_LOAD_COMPLETED` | `request_id`, `loaded_count`, `failed_count` | `int`, `int`, `int` |
| `L2_LOAD_TASK_SUBMITTED` | `request_id`, `adapter_index`, `task_id`, `l2_name`, `key_count`, `total_bytes` | `int`, `int`, `int`, `str`, `int`, `int` |
| `L2_LOAD_TASK_COMPLETED` | `request_id`, `adapter_index`, `task_id`, `l2_name`, `total_bytes` | `int`, `int`, `int`, `str`, `int` |

`L2_LOAD_TASK_*` events fire once per `(request_id, adapter_index)` pair
— unlike the request-level `L2_PREFETCH_LOAD_*` events above, which
aggregate across adapters.  Throughput subscribers that need per-adapter
attribution (e.g. `L2ThroughputSubscriber`) consume these task-level
events; key-count counters continue to consume the request-level events.

---

## MP Server Lifecycle Sentinels

CPU-synchronous sentinels published by `server.py` to bracket request scope.
Published via `EventBus.publish()` (not `publish_on_stream`) so the drain
thread processes them in strict order before any GPU-callback events.

| EventType | Metadata keys | Types | Published by / when |
|---|---|---|---|
| `MP_REQUEST_START` | *(none)* | — | `MPServer.handle_request` — at request arrival, before any GPU work |
| `MP_STORE_SUBMITTED` | `device` | `str` | `MPServer.store` — CPU-synchronous, before the GPU store is enqueued |
| `MP_RETRIEVE_SUBMITTED` | `device` | `str` | `MPServer.retrieve` — CPU-synchronous, before the GPU retrieve is enqueued |
| `MP_SESSION_END` | *(none)* | — | `MPServer.handle_request` — after all CPU work; may precede GPU callbacks |

---

## MP Server Events

These events use `session_id` on the `Event` dataclass (not in `metadata`)
to correlate START/END pairs.

| EventType | Metadata keys | Types |
|---|---|---|
| `MP_STORE_START` | `device`, `engine_id`, `model_name` | `str`, `int`, `str` |
| `MP_STORE_END` | `device`, `stored_count`, `engine_id`, `model_name`, `total_bytes` | `str`, `int`, `int`, `str`, `int` |
| `MP_RETRIEVE_START` | `device`, `engine_id`, `model_name` | `str`, `int`, `str` |
| `MP_RETRIEVE_END` | `device`, `retrieved_count`, `engine_id`, `model_name`, `total_bytes` | `str`, `int`, `int`, `str`, `int` |
| `MP_LOOKUP_PREFETCH_START` | *(none)* | — |
| `MP_LOOKUP_PREFETCH_END` | `found_count` | `int` |
| `MP_LOOKUP` | `request_id`, `chunk_hashes`, `model_name`, `chunk_size`, `seq_len`, `dtypes`, `shapes` | `str`, `list[str]`, `str`, `int`, `int`, `list[str]`, `list[list[int]]` |
| `MP_VLLM_BLOCK_ALLOCATION` | `instance_id`, `model_name`, `records` | `int`, `str`, `list[BlockAllocationRecord]` (each has `req_id: str`, `new_block_ids: list[int]`, `new_token_ids: list[int]`) |
| `MP_VLLM_END_SESSION` | `request_id` | `str` |

---

## Trace Recording Events

A single unified event used by the `@enable_tracing` decorator (see
[trace.md](trace.md)). All instrumented call sites publish the same
`EventType` regardless of which method or layer; the `qualname` field
inside `metadata` discriminates ops.

| EventType | Metadata keys | Types |
|---|---|---|
| `TRACE_CALL` | `qualname`, `args` | `str`, `dict[str, Any]` (codec-encoded; see `lmcache.v1.mp_observability.trace.codecs`) |
