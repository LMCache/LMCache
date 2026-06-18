# KV Cache SDK

> Status: v1 out-of-process client for the LMCache multiprocess (MP) server, over ZMQ.

## Goal

Provide a small Python surface for moving KV cache tensors into and out of a running
LMCache MP server, addressed by **token ids**. 
The SDK drives the same machinery the engine uses, across two planes:

1. **Control plane (ZMQ message queue)** — registration, lookup/prefetch, slot
   reservation, and lock release.
2. **Data plane (POSIX shared memory)** — the KV bytes are read/written directly in the
   server's L1 SHM pool; only metadata crosses the socket.

```py
import lmcache.sdk.kvcache as lmc_sdk

ctx = lmc_sdk.connect(
	url="tcp://localhost:5555", 
	model_name="Qwen/Qwen3-8B"
)

tensor = lmc_sdk.retrieve(
	ctx, 
	tokens=[1, 2, 3, ...]
)

# do token dropping

ok = lmc_sdk.store(
	ctx, 
	kv=tensor, 
	tokens=[4, 5, 6, ...]
)

lmc_sdk.close(ctx)
```

Example of the token dropping API usage in 
[Token Dropping Example](../../../examples/token_dropping/multi_req_split.py).

The model layout must already be registered in the server by a vLLM instance that called
`REGISTER_KV_CACHE`. The SDK does **not** supply layout itself — it asks the server to
resolve it from the model name (see [Registration handshake](#registration-handshake)).

## Scope

In scope:

- Retrieve the longest whole-chunk cached prefix for a token list.
- Store a whole-chunk-aligned KV tensor for a token list.
- A single transport: **shared memory** (one copy in / one copy out).

Limitations:

- `world_size > 1` (tensor parallelism). The SDK supports `world_size == 1` only and
  raises `KVCacheSDKError` otherwise.
- Pickle transport on the SDK.

## Architecture

The SDK is a **separate process** from the LMCache server.

```
SDK process                          LMCache server process
-----------                          ----------------------
LMCacheKVCacheContext
  └ MessageQueueClient ──ZMQ──▶ MQ dispatch
                                  ├ EngineDrivenTransferModule  (PREPARE_/COMMIT_ STORE/RETRIEVE)
                                  │     └ ShmTransferStrategy
                                  │         └ StorageManager (L1 pool, locks, prefetch)
                                  └ LookupModule          (LOOKUP / QUERY_PREFETCH_STATUS)
  └ SharedMemory(name) ◀──────── same POSIX segment (data plane only)
```

- **Control plane — ZMQ.** All bookkeeping (which keys exist, prefetch, lock
  acquire/release, slot reservation) lives in the server's `StorageManager`, an in-process
  object the client cannot share. The SDK drives it exclusively through MQ requests.
- **Data plane — shared memory.** The server's L1 pool is a POSIX SHM segment. The server
  returns byte offsets into the pool, and the SDK maps those with `torch.frombuffer` for a
  single copy in/out.

## Registration handshake

`connect()` constructs an `LMCacheKVCacheContext`, whose `__init__` runs a one-time handshake.
All calls are blocking and bounded by `timeout`:

1. `GET_WORLD_SIZE [model_name]` → `world_size`. **Raises `KVCacheSDKError` if 
	`world_size != 1`.**
2. `GET_CHUNK_SIZE []` → the server chunk size.
3. `GET_SHM_POOL_INFO []` → `{shm_name, pool_size}` for the data plane.
4. `REGISTER_SDK_TRANSFER_STRATEGY [instance_id, model_name, world_size]`.
	It **resolves the `MemoryLayoutDesc` from the model registry** via 
	`resolve_model_name(model_name)` — populated when the vLLM instance called `REGISTER_KV_CACHE` 
	then registers, keyed by `instance_id`:
	- the per-instance non-GPU context (`EngineDrivenContextMetadata`: layout, block size, `use_mla`
	  derived from the layout shape), and
	- the SHM transfer strategy.

`instance_id` is the SDK process PID (`os.getpid()`).

## Public API

Exported from `lmcache.sdk`: `LMCacheKVCacheContext`, `KVCacheSDKError`.
Operations are module-level functions in `lmcache.sdk.kvcache`: `connect`, `close`,
`retrieve`, `store`.

### `connect(url, model_name, timeout=60.0) -> LMCacheKVCacheContext`

Opens a `MessageQueueClient` to `url` (TCP) and runs the [registration
handshake](#registration-handshake). The returned context is context-manager capable
(`__enter__` / `close`).

### `retrieve(ctx, tokens, cache_salt="") -> torch.Tensor | None`

Returns a **contiguous CPU** tensor of shape `[2, num_layers, hit_tokens, hidden_dim]` in
`KV_2LTD` layout, where `hit_tokens` is the longest whole-chunk cached prefix. Returns
`None` when `tokens` is empty or shorter than one chunk. Raises `KVCacheSDKError` if the
server returns no SHM slots (e.g. SHM disabled, or nothing cached after prefetch).

### `store(ctx, kv, tokens, cache_salt="") -> bool`

Stores `kv` (a 4-D `KV_2LTD` tensor) for `tokens`. The tensor is moved to a contiguous CPU
tensor and validated before transfer. Returns the server commit result (`bool`).

### `close(ctx) -> None`

Shuts down the MQ client and ZMQ context.

## Cache addressing

Both store and retrieve build an `IPCCacheServerKey`:

```py
IPCCacheServerKey(
    model_name, 
	world_size=1, # for now only work for TP==1
	worker_id=0, # for now only work for TP==1
    token_ids, 
	start=0, 
	end=<chunk-aligned>, 
	request_id, # uuid.uuid4().hex
	cache_salt,
)
```

The server resolves it to per-chunk `ObjectKey`s. **Cache identity** is
`token-chunk hashes + model_name + kv_rank(worker_id) + cache_salt`. `request_id` is a
fresh per-call id (`store-<uuid>` / `retrieve-<uuid>`) and is **not** part of cache
identity — it only keys the per-request server session and prefetch job.

- `worker_id=0` is valid because the SDK is `world_size == 1` — exactly one KV shard per
  chunk, so "worker 0" *is* the whole chunk. (For `world_size > 1` this would address only
  one of N shards; hence the `world_size == 1` guard.)
- The **lookup** request uses the `worker_id=None` (expand-to-all-workers) variant, because
  a lookup must confirm every worker's shard is present.

## Memory contract

The SDK treats the KV cache as a 4-D in-memory `torch.Tensor` in canonical `KV_2LTD`
layout `[2, num_layers, total_tokens, hidden_dim]`. Cache identity is passed explicitly as
SDK parameters: `model_name` (at connect), `tokens`, and optional `cache_salt`.

- `retrieve` reads each SHM slot as a zero-copy `torch.frombuffer` view, then
  `_assemble_contiguous` allocates the result tensor and copies each shard slice into it
  (with `world_size == 1` there is one shard per chunk). The returned tensor owns its
  memory independently of the pool; the copy completes before the segment is closed and
  before `COMMIT_RETRIEVE`.
- `store` reserves slots, copies each `chunk_size`-token slice of the input tensor into its
  SHM slot, then commits. Storage reserves writes with mode `"new"`, so chunks already
  present in cache are deduplicated and not rewritten.

## Protocol

The SDK uses the following MQ request types (defined in
`lmcache.v1.multiprocess.protocols`; that module is the authoritative wire contract):

| Request | Purpose |
| --- | --- |
| `GET_WORLD_SIZE` | Learn the registered `world_size` for the model (SDK requires `1`). |
| `GET_CHUNK_SIZE` | Learn the server chunk size. |
| `GET_SHM_POOL_INFO` | Learn the SHM pool name + size (data plane). |
| `REGISTER_SDK_TRANSFER_STRATEGY` | Register the per-instance context + SHM strategy; layout is resolved server-side from the model registry. |
| `LOOKUP` | Submit a prefix lookup and kick off the server-side prefetch (retrieve only). |
| `QUERY_PREFETCH_STATUS` | Poll prefetch completion → matched chunk count (`None` while in progress). |
| `PREPARE_RETRIEVE` | Read the cached prefix into SHM and return slot descriptors (read-locked). |
| `COMMIT_RETRIEVE` | Release the read locks held since prepare. |
| `PREPARE_STORE` | Reserve SHM slots for the missing chunks and return descriptors (write-locked). |
| `COMMIT_STORE` | Finalize the write and release the write locks. |
| `END_SESSION` | Drop the per-request server session. |

The server constructs/uses the `IPCCacheServerKey` carried in each `PREPARE_*`/`COMMIT_*`
payload; it resolves the layout for store/retrieve via the per-instance context registered
at connect time.

## Retrieve flow (SHM)

```
 SDK                         Server (Lookup + EngineDrivenTransferModule + Storage)
  │                              │
  │┐ key = IPCCacheServerKey(worker_id=0); request_id = "retrieve-<uuid>"
  │┘
  │ Phase 0 — lookup + prefetch
  │ LOOKUP [key(worker_id=None), world_size]
  │─────────────────────────────▶│ LookupModule.lookup → submit_prefetch_task (locks + loads)
  │◀─────────────────────────────│ (ack)
  │ QUERY_PREFETCH_STATUS [request_id]   (poll until non-None)
  │─────────────────────────────▶│ None while prefetch in flight; chunk count when done
  │◀─────────────────────────────│ (chunk count)
  │ Phase 1 — prepare			 |
  │ PREPARE_RETRIEVE [key, instance_id]
  │─────────────────────────────▶│ strategy.prepare_retrieve → unsafe_read → slots (read locks)
  │◀──── resp(success, slots) ───│
  │ Phase 2 — read SHM
  │┐ open SHM(shm_name); frombuffer views (0-copy)
  ││ _assemble_contiguous → copy SHM→result  ◀══ 1 COPY
  │┘ shm.close()
  │ Phase 3 — commit + end session
  │ COMMIT_RETRIEVE [key, instance_id]
  │─────────────────────────────▶│ finish_read_prefetched (release read locks)
  │ END_SESSION [request_id]
  │─────────────────────────────▶│ drop session
  │  return contiguous tensor [2, L, hit_tokens, D]   (or None on miss)
  ▼                              ▼
```

## Store flow (SHM)

```
 SDK                         Server (EngineDrivenTransferModule + Storage)
  │                              │
  │┐ kv.detach().cpu().contiguous(); _validate_store_tensor
  │┘ key = IPCCacheServerKey(worker_id=0); request_id = "store-<uuid>"
  │ Phase 1 — prepare
  │ PREPARE_STORE [key, instance_id]
  │─────────────────────────────▶│ reserve_write(obj_keys, "new") → slots for MISSING chunks
  │◀──── resp(context={slots}) ───│   (already-cached chunks are deduplicated, no slot)
  │ Phase 2 — write SHM
  │┐ open SHM(shm_name); for each slot: frombuffer view
  ││ dst.copy_(kv[..., chunk/shard slice])  ◀══ 1 COPY (CPU→SHM)
  │┘ shm.close()
  │ Phase 3 — commit + end session
  │ COMMIT_STORE [key, instance_id, cpu_data=b""]
  │─────────────────────────────▶│ finish_write (release write locks) → bool
  │◀──── ok ──────────────────────│
  │ END_SESSION [request_id]
  │─────────────────────────────▶│ drop session
  │  return ok (bool)
  ▼                              ▼
```

## Copy summary

| Flow | Copies | Notes |
| --- | --- | --- |
| Retrieve SHM | 1 | SHM → contiguous result, SDK side |
| Store SHM | 1 | CPU → SHM, SDK side |

## Error handling

Protocol/transport failures raise `KVCacheSDKError`. A cache miss is **not** an error —
`retrieve` returns `None` for empty/sub-chunk input. The SDK does not retry; callers that
need retry semantics implement it around the SDK call (repeated store calls can partially
overwrite the same cache prefix).

The lookup poll loop (`QUERY_PREFETCH_STATUS` returning `None`) waits for the server-side
prefetch to complete before `PREPARE_RETRIEVE`. Each MQ call is bounded by `timeout`.

## Constraints & known gaps

- **`world_size == 1` only.** Enforced at connect (`GET_WORLD_SIZE`), since `worker_id=0`
  addresses the whole chunk only when there is a single shard.
- **Pickle not yet.** The SDK client requires SHM slot descriptors; the pickle transport is
  not yet implemented on the client.
  Implementation plan:
  - The current LMCacheKVCacheContext will be an ABC, extended by LMCacheKVCacheContextShm and 
  	LMCacheKVCacheContextPickle.
  - or, just put if else conditions based on the response form (e.g., SHM returns slots,
  	but pickle returns data in bytes).
- Unlike original implementation, in which store() returns StoreResult(total_tokens,
  total_chunks, stored_tokens, stored_chunks), current implementation only returns boolean
  because the LMCache's PREPARE_STORE only returns boolean.
- **Model must be pre-registered.** Layout is resolved from the server registry, which is
  populated by a vLLM instance's `REGISTER_KV_CACHE`. Therefore, the LMCache server should
  call REGISTER_KV_CACHE, called by vLLM, prior to the SDK running.
