# KV Cache SDK

> Status: v2 out-of-process client for the LMCache multiprocess (MP) server.

## Goal

Provide a small Python surface for moving KV cache tensors into and out of a running
LMCache MP server, addressed by **token ids**. 
The SDK drives the same machinery the engine uses, across two planes:

1. **Control plane (ZMQ message queue)** — registration, lookup/prefetch, slot
   reservation, and lock release.
2. **Data plane** — If shm name is specified when starting lmcache, SDK will use SHM,
   otherwise, it will use Pickle through LMCache's EngineDrivenTransferContext.
   The SDK stages KV in a paged CPU buffer it allocates at connect time and lets an 
   engine-driven transfer context copy it to/from the pool.

```py
import lmcache.sdk.kvcache as lmc_sdk

ctx = lmc_sdk.connect(
	url="tcp://localhost:5555",
	http_url="http://localhost:9000",
	model_name="Qwen/Qwen3-8B",
	kv_buffer_bytes=2 << 30,  # SDK-owned paged staging buffer budget
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
`REGISTER_KV_CACHE`. The SDK reads that layout from the server's `/status` and sizes its own
CPU staging buffer to match (see [Registration handshake](#registration-handshake)).

## Scope

In scope:

- Retrieve the longest whole-chunk cached prefix for a token list.
- Store a whole-chunk-aligned KV tensor for a token list.
- Transport is **SHM or pickle**, selected by the server (whether it exposes a SHM pool),
  via the engine-driven `TransferContext`; both stage through the paged CPU buffer (see
  [Copy summary](#copy-summary)).

Limitations:

- The SDK supports `world_size == 1` only and
  raises `KVCacheSDKError` otherwise.

## Architecture

The SDK is a **separate process** from the LMCache server.

```
SDK process                          LMCache server process
-----------                          ----------------------
LMCacheKVCacheContext
  ├ paged CPU buffer + BlockPool   (KV staging)
  ├ EngineDrivenTransferContext    (scatter/gather ↔ paged buffer)
  └ MessageQueueClient ──ZMQ──▶ MQ dispatch
                                  ├ EngineDrivenTransferModule  (PREPARE_/COMMIT_ STORE/RETRIEVE)
                                  │     └ EngineDrivenContextShm / EngineDrivenContextPickle
                                  │         └ StorageManager (L1 pool, locks, prefetch)
                                  └ LookupModule          (LOOKUP / QUERY_PREFETCH_STATUS)
  └ SharedMemory(name) ◀──────── same POSIX segment (data plane only)
```

- **Control plane — ZMQ.** All bookkeeping (which keys exist, prefetch, lock
  acquire/release, slot reservation) lives in the server's `StorageManager`.
- **Data plane — SHM or pickle.** The SDK's `EngineDrivenTransferContext` scatters/gathers
  between its paged CPU buffer and the transport: the server's L1 POSIX SHM pool when one is
  configured, otherwise a pickle payload over the MQ (see [Copy summary](#copy-summary)).

## Registration handshake

`connect()` constructs an `LMCacheKVCacheContext` (`__init__`) and then calls
`register_kv_caches(kv_buffer_bytes)`. All calls are blocking and bounded by `timeout`:

1. HTTP `/conf` → `chunk_size`, `shm_name`.
2. HTTP `/status` → the `cache_context_meta` entry for `model_name`: its `world_size` and the
   GPU `kv_cache_layout` (num_layers, dtype, `tokens_per_block`, `engine_kv_format`,
   `engine_kv_concrete_shape`).
3. Allocate the staging buffer. Decode `num_kv_heads`, `block_size`, `head_dim` from the
   layout; size `num_blocks = kv_buffer_bytes / (num_layers · 2 · num_kv_heads · block_size ·
   head_dim · itemsize)`; allocate a paged CPU buffer `{layer: [num_blocks, 2, NH, BS, HS]}`
   plus a `BlockPool`.
4. `create_transfer_context(buffer)` → `EngineDrivenTransferContext.register(...)`, which sends
   `REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT`. The response carries `shm_name`/`pool_size`: when
   present the context uses `EngineDrivenContextShm`, otherwise it falls back to the pickle
   transport. Either way the SDK code path is identical.

`instance_id` is the SDK process PID (`os.getpid()`).

> **HND on CPU.** The SDK runs on CPU, and LMCache's format detector hardcodes **HND** for the
> CPU backend ([utils.py:663](../../../lmcache/v1/gpu_connector/utils.py#L663)). The buffer is
> therefore built in HND-physical order `[NB, 2, NH, BS, HS]` regardless of the engine's GPU
> layout, so detection reads NH/BS from the right slots.

## Public API

Exported from `lmcache.sdk`: `LMCacheKVCacheContext`, `KVCacheSDKError`.
Operations are module-level functions in `lmcache.sdk.kvcache`: `connect`, `close`,
`retrieve`, `store`.

### `connect(url, http_url, model_name, kv_buffer_bytes, timeout=60.0) -> LMCacheKVCacheContext`

Opens a `MessageQueueClient` to `url` (TCP), fetches config from `http_url`, and runs the
[registration handshake](#registration-handshake). `kv_buffer_bytes` budgets the paged CPU
staging buffer (floored to whole blocks). If user specified kv_buffer_bytes larger than
available, torch will throw the error.

### `retrieve(ctx, tokens, cache_salt="") -> torch.Tensor | None`

Returns a **contiguous CPU** tensor of shape `[2, num_layers, hit_tokens, hidden_dim]` in
`KV_2LTD` layout, where `hit_tokens` is the longest whole-chunk cached prefix. Returns
`None` when `tokens` is empty, shorter than one chunk, or nothing is cached after prefetch.
Protocol/transport failures raise `KVCacheSDKError`.

### `store(ctx, kv, tokens, cache_salt="") -> bool`

Stores `kv` (a 4-D `KV_2LTD` tensor) for `tokens`. The tensor is moved to a contiguous CPU
tensor, scattered into the paged buffer, and gathered to the server's SHM pool. Returns the
server commit result (`bool`).

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
- The **lookup** request uses the `worker_id=None`.

## Memory contract

The SDK treats the KV cache as a 4-D in-memory `torch.Tensor` in canonical `KV_2LTD`
layout `[2, num_layers, total_tokens, hidden_dim]`. Cache identity is passed explicitly as
SDK parameters: `model_name` (at connect), `tokens`, and optional `cache_salt`.

- `store` splits the input into `chunk_size` slices, `scatter_cpu_to_paged_kv` writes them
  into the paged buffer at freshly-allocated block ids, then `submit_store` (engine-driven)
  gathers the buffer into the transport (SHM pool or pickle payload) and commits. Storage
  reserves writes with mode `"new"`, so chunks already in cache are deduplicated.
- `retrieve` calls `submit_retrieve` (engine-driven scatters the transport into the paged
  buffer), then `gather_paged_kv_to_cpu` + `torch.cat` assemble a contiguous
  `[2, L, hit_tokens, D]` result. Block ids are returned to the `BlockPool` in `end_session`.

## Protocol

The SDK directly calls the following MQ request types because these are minimally needed
by the SDK without duplicating the whole `vllm_multi_adapter.py` code:

| Request | Purpose |
| --- | --- |
| `LOOKUP` | Submit a prefix lookup and kick off the server-side prefetch (retrieve only). |
| `QUERY_PREFETCH_STATUS` | Poll prefetch completion → matched chunk count (`None` while in progress). |
| `END_SESSION` | Drop the per-request server session. |

The server constructs/uses the `IPCCacheServerKey` for store/retrieve via the per-instance 
context registered at connect time.

The `PREPARE_*`/`COMMIT_*` calls are issued **inside** `submit_store` / `submit_retrieve`
(the engine-driven transfer context wraps them); the SDK does not call them directly. The
flows below show the SHM transport; with pickle the server returns/consumes a bytes payload
over the MQ instead of SHM slots, and the scatter/gather steps are unchanged.

## Retrieve flow

```
 SDK                                          Server
  │ key = IPCCacheServerKey(worker_id=0); request_id = "retrieve-<uuid>"
  │ Phase 0 — lookup + prefetch
  │ LOOKUP [key(worker_id=None), world_size] ─▶ LookupModule → prefetch (locks + loads)
  │ QUERY_PREFETCH_STATUS [request_id]        ◀▶ poll; chunk count when done (None in flight)
  │ Phase 1 — alloc blocks + retrieve
  │ flat_block_ids = BlockPool.alloc(n_chunks * blocks_in_chunk)
  │ transfer_ctx.submit_retrieve(...):
  │   PREPARE_RETRIEVE [key, instance_id]  ─▶ slots (read locks)
  │   scatter_cpu_to_paged_kv: SHM → paged buffer          ◀══ LMCache server COPY
  │   COMMIT_RETRIEVE [key, instance_id]   ─▶ release read locks
  │ Phase 2 — assemble
  │ gather_paged_kv_to_cpu + torch.cat: paged → contiguous ◀══ SDK 2 COPIES
  │ END_SESSION [request_id]; BlockPool.free(flat_block_ids)
  │ return [2, L, hit_tokens, D]   (or None on miss)
```
- LMCache server copies L1 (SHM pool) → SDK's paged buffer via `scatter_cpu_to_paged_kv`.
- SDK copies paged buffer → per-chunk contiguous CPU tensors via `gather_paged_kv_to_cpu`,
  transforming HND `[NB, 2, NH, BS, HS]` order to `[2, L, chunk_tokens, D]`.
- SDK copies chunks → single contiguous result: `torch.cat(chunks, dim=2).contiguous()`
  to `[2, L, hit_tokens, D]`.

## Store flow

```
 SDK                                          Server
  │ kv.detach().cpu().contiguous(); key(worker_id=0); request_id = "store-<uuid>"
  │ Phase 1 — scatter into paged buffer
  │ block_ids = BlockPool.alloc(n_chunks * blocks_in_chunk)
  │ chunks = kv.split(chunk_size, dim=2) (contiguous)       ◀══ SDK COPY
  │ scatter_cpu_to_paged_kv: chunks → paged buffer          ◀══ SDK COPY
  │ Phase 2 — store
  │ transfer_ctx.submit_store(...):
  │   PREPARE_STORE [key, instance_id]  ─▶ slots for MISSING chunks (deduped)
  │   gather_paged_kv_to_cpu: paged → SHM                   ◀══ LMCache server COPY
  │   COMMIT_STORE [key, instance_id]   ─▶ finish_write → bool
  │ END_SESSION [request_id]; BlockPool.free(block_ids)
  │ return ok (bool)
```
- SDK splits `kv_cpu` into `chunk_size` slices, making each contiguous.
- SDK scatters chunks into its paged buffer via `scatter_cpu_to_paged_kv`, because
  `submit_store()` reads from the paged buffer addressed by `block_ids`.
- `submit_store()` lands the data in the server's L1 pool:
  - SHM: `gather_paged_kv_to_cpu` writes straight into the reserved L1 slots.
  - Pickle: gather to CPU chunks, send pickled bytes over MQ, server deserializes to L1.

## Copy summary

| Flow | Copies | Notes |
| --- | --- | --- |
| Retrieve | ~3 | SHM→paged (LMCache server calls scatter), paged→chunks (SDK calls gather), `cat`→contiguous (SDK calls `torch.cat`) |
| Store | ~3 | input→chunks (SDK calls `contiguous`), chunks→paged (SDK calls scatter), paged→SHM (LMCache server calls gather) |

The paged staging buffer adds a round-trip versus a direct contiguous↔transport copy — the
cost of reusing the engine-driven `TransferContext`. Counts are for SHM; pickle replaces the
SHM hop with an MQ-carried bytes payload (de/serialization in place of a pool copy).

## Error handling

Protocol/transport failures raise `KVCacheSDKError`. A cache miss is **not** an error —
`retrieve` returns `None` for empty/sub-chunk input. The SDK does not retry; callers that
need retry semantics implement it around the SDK call (repeated store calls can partially
overwrite the same cache prefix).

The lookup poll loop (`QUERY_PREFETCH_STATUS` returning `None`) waits for the server-side
prefetch to complete before `PREPARE_RETRIEVE`. Each MQ call is bounded by `timeout`.

## Constraints & known gaps

- **`world_size == 1` only.**
- **Staging-buffer budget depends on user.** Raises `KVCacheSDKError` if SDK's paged buffer is
  not enough.
- **Model must be pre-registered with inference engine using GPU.** Layout is resolved from 
  the server registry, which is populated by a vLLM instance's `REGISTER_KV_CACHE`. Therefore, 
  the LMCache server should call REGISTER_KV_CACHE, called by vLLM, prior to the SDK running.
