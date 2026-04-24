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
| `L2_STORE_SUBMITTED` | `adapter_index`, `key_count` | `int`, `int` |
| `L2_STORE_COMPLETED` | `adapter_index`, `succeeded_count`, `failed_count` | `int`, `int`, `int` |

---

## L2 Prefetch Controller Events

| EventType | Metadata keys | Types |
|---|---|---|
| `L2_PREFETCH_LOOKUP_SUBMITTED` | `request_id`, `key_count`, `adapter_count` | `int`, `int`, `int` |
| `L2_PREFETCH_LOOKUP_COMPLETED` | `request_id`, `prefix_hit_count` | `int`, `int` |
| `L2_PREFETCH_LOAD_SUBMITTED` | `request_id`, `key_count`, `adapter_count` | `int`, `int`, `int` |
| `L2_PREFETCH_LOAD_COMPLETED` | `request_id`, `loaded_count`, `failed_count` | `int`, `int`, `int` |

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
| `MP_REQUEST_END` | *(none)* | — | `MPServer.handle_request` — after all CPU work; may precede GPU callbacks |

---

## MP Server Events

These events use `session_id` on the `Event` dataclass (not in `metadata`)
to correlate START/END pairs.

| EventType | Metadata keys | Types |
|---|---|---|
| `MP_STORE_START` | `device`, `engine_id`, `gpu_id` | `str`, `int`, `int` |
| `MP_STORE_END` | `device`, `stored_count`, `engine_id`, `gpu_id`, `total_bytes` | `str`, `int`, `int`, `int`, `int` |
| `MP_RETRIEVE_START` | `device`, `engine_id`, `gpu_id` | `str`, `int`, `int` |
| `MP_RETRIEVE_END` | `device`, `retrieved_count`, `engine_id`, `gpu_id`, `total_bytes` | `str`, `int`, `int`, `int`, `int` |
| `MP_LOOKUP_PREFETCH_START` | *(none)* | — |
| `MP_LOOKUP_PREFETCH_END` | `found_count`, `requested_tokens`, `hit_tokens` | `int`, `int`, `int` |
| `MP_LOOKUP` | `request_id`, `chunk_hashes`, `model_name`, `chunk_size`, `seq_len`, `dtypes`, `shapes` | `str`, `list[str]`, `str`, `int`, `int`, `list[str]`, `list[list[int]]` |
| `MP_VLLM_BLOCK_ALLOCATION` | `instance_id`, `model_name`, `records` | `int`, `str`, `list[BlockAllocationRecord]` (each has `req_id: str`, `new_block_ids: list[int]`, `new_token_ids: list[int]`) |
| `MP_VLLM_END_SESSION` | `request_id` | `str` |

### `MP_LOOKUP_PREFETCH_END` metadata

`found_count` is the contiguous prefix hit at chunk granularity, already
divided by `world_size` at the emit site.  `requested_tokens` and
`hit_tokens` are denormalized token-level counts so subscribers need not
know `chunk_size`:

- `requested_tokens = len(chunk_hashes) * chunk_size` on the happy path; `0`
  on the two early-exit paths in `lookup()` (no matching GPU context,
  empty `chunk_hashes`).  Sub-chunk trailing tokens are excluded — they
  cannot hit at chunk granularity.
- `hit_tokens = found_count * chunk_size`.

Together they drive the `lmcache_mp.lookup_*_tokens` counters used to
compute the L1+L2 token-level hit rate.  See
[L1_L2_HIT_RATE_PLAN.md](L1_L2_HIT_RATE_PLAN.md) for the design.

---

## Trace Recording Events

A single unified event used by the `@enable_tracing` decorator (see
[trace.md](trace.md)). All instrumented call sites publish the same
`EventType` regardless of which method or layer; the `qualname` field
inside `metadata` discriminates ops.

| EventType | Metadata keys | Types |
|---|---|---|
| `TRACE_CALL` | `qualname`, `args` | `str`, `dict[str, Any]` (codec-encoded; see `lmcache.v1.mp_observability.trace.codecs`) |

---

## Blend Server Lifecycle Sentinels

CPU-synchronous sentinels published by `blend_server_v2.py` to bracket
request scope and guard GPU callback races.  Published via `EventBus.publish()`
(not `publish_on_stream`).

| EventType | Metadata keys | Types | Published by / when |
|---|---|---|---|
| `CB_REQUEST_START` | *(none)* | — | `BlendEngineV2.cb_lookup_pre_computed` — at request arrival |
| `CB_STORE_PRE_COMPUTED_SUBMITTED` | `instance_id` | `int` | `BlendEngineV2.cb_store_pre_computed` — before GPU store enqueue |
| `CB_RETRIEVE_SUBMITTED` | `instance_id` | `int` | `BlendEngineV2.cb_retrieve_pre_computed` — before GPU retrieve enqueue |
| `CB_STORE_FINAL_SUBMITTED` | `instance_id` | `int` | `BlendEngineV2.cb_store_final` — before GPU store enqueue |
| `CB_REQUEST_END` | *(none)* | — | `BlendEngineV2.cb_lookup_pre_computed` (early return: no matches or no GPU context) **or** `BlendEngineV2.cb_store_final` — after SUBMITTED, before GPU work |

---

## Blend Server Events

These events use `session_id` on the `Event` dataclass (sourced from
`IPCCacheEngineKey.request_id`) to correlate START/END pairs.

| EventType | Metadata keys | Types |
|---|---|---|
| `CB_LOOKUP_START` | `num_tokens` | `int` |
| `CB_LOOKUP_END` | `num_tokens`, `fingerprint_hits`, `storage_hits`, `stale_chunks`, `no_gpu_context` | `int`, `int`, `int`, `int`, `bool` |
| `CB_STORE_PRE_COMPUTED_START` | `instance_id`, `num_tokens` | `int`, `int` |
| `CB_STORE_PRE_COMPUTED_END` | `instance_id`, `num_tokens`, `stored_chunks`, `success` | `int`, `int`, `int`, `bool` |
| `CB_RETRIEVE_START` | `instance_id`, `num_chunks` | `int`, `int` |
| `CB_RETRIEVE_END` | `instance_id`, `num_chunks`, `success` | `int`, `int`, `bool` |
| `CB_STORE_FINAL_START` | `instance_id`, `num_tokens` | `int`, `int` |
| `CB_STORE_FINAL_END` | `instance_id`, `num_tokens`, `stored_chunks`, `success` | `int`, `int`, `int`, `bool` |
| `CB_FINGERPRINTS_REGISTERED` | `num_chunks`, `num_tokens` | `int`, `int` |
| `CB_CHUNKS_EVICTED` | `num_chunks` | `int` |
