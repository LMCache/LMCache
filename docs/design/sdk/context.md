# LMCache SDK

> Status: out-of-process, CPU-only client for the LMCache multiprocess (MP) server.

## Goal

A small Python surface for moving **KV cache** (and **query**) tensors in and out of a
running LMCache MP server, addressed by **token ids** — so an ML engineer can retrieve a
prefix's KV, edit it, and store it back (e.g. token-dropping inside a KV connector). One
`LMCacheSDKContext` serves either kind, selected by `LMCacheSDKCacheKind` (`KV` / `QUERY`).
Retrieve / store / close are **methods** on the context.

```py
import lmcache.sdk.kvcache as kvcache   # query tensors: lmcache.sdk.qcache

ctx = kvcache.connect(
    url="tcp://localhost:5555",       # tcp:// for ZMQ; grpc:// for gRPC
    http_url="http://localhost:9000", # HTTP url for retrieving KV cache shape
    model_name="Qwen/Qwen3-8B",
)
kv = ctx.retrieve(tokens=[1, 2, 3, ...])   # [2, L, hit_tokens, D] or None
# ... edit kv ...
ok = ctx.store(kv=kv, tokens=[4, 5, 6, ...])
ctx.close()
```

See the [token-dropping example](../../../examples/kvcache_sdk/e2e_kv_edit.ipynb).

The model layout must already be registered in the server by a vLLM instance with query
transfer enabled (`"lmcache.mp.transfer_intermediate_tensors": true`), which also creates the
Q ring that is used for: (1) gathering vLLM-generated Q tensors before transferred to LMCache,
(2) staging buffer for LMCache-driven transfer between LMCache and SDK.

## Architecture

The SDK is a **separate, CPU-only process** from the LMCache server and vLLM. It allocates and
registers no buffer. Transfers take the **LMCache-driven** path: the server copies L1 objects
to and from GPU memory over device IPC, and the GPU memory is the **QRing attached in vLLM process**
(`QRingBuffer`), borrowed while vLLM schedules no requests.

```
SDK process                        LMCache MP server                 vLLM worker
-----------                        -----------------                 -----------
LMCacheSDKContext                                                    QRingBuffer (GPU)
  ├ RingSDKTransferWrapper ──RPC─▶ QStoreModule                       ├ allocate/free
  │   SDK_RETRIEVE / SDK_STORE      ├ L1 object (ring pieces copy)    └ publish_state()
  │   (pieces of free blocks)       │   (CUDA IPC + events)               │
  ├ RingMapping ─────GET_Q_RING──▶  └ per-ring used blocks ◀──REPORT_Q_RING_STATE
  │   (maps the ring's IPC handles; checks num_alloc)
  └ RequestClient ──ZMQ/gRPC──────▶ LookupModule (LOOKUP / QUERY_PREFETCH_STATUS)
```

- **Ring state:** at the end of each step in which the ring changed, the vLLM worker reports
  its used blocks and allocation count (`num_alloc`).
- **Exclusive phase:** transfer must happen only when no prefill/decode was running. The SDK
  reads the ring state before and after; a changed `num_alloc` (vLLM took blocks) or instance ID
  (vLLM restarted) fails the transfer.

## Registration handshake

`connect()` builds the context, then `register_caches()` runs once (for the `QUERY` kind the
model name is suffixed `##query`):

1. HTTP `/config` → `chunk_size`.
2. HTTP `/status` → this kind's registration (`cache_context_meta` / `kv_cache_layout` for KV,
   `q_context_meta` / `q_ring_layout` for query) and qstore's `object_layouts`: per object
   group, its kind, window, and one chunk's tensor shapes and dtypes.
3. Check the layout is dense: one kernel group, one full-attention object group without a
   window, uncompressed and chunk-aligned; KV at `world_size == 1`.
4. `GET_Q_RING` → per rank, the ring's instance ID, IPC handles, used blocks and `num_alloc`;
   map the handles (`RingMapping`).

## Flows

**retrieve**:
- Phase 0: `LOOKUP` + poll `QUERY_PREFETCH_STATUS` until non-`None`; if 0 → `None`. 
- Phase 1, holding the ring lock: read the ring state; split the range into piece size,
`SDK_RETRIEVE` (the server waits on the SDK's event, copies the objects into the given pieces,
returns its completion event), then gather each object from its pieces into CPU rows.

**store** — validate `len(tokens) == kv.shape[2]`, truncate to whole chunks; per batch,
encode each chunk's object into free pieces, then `SDK_STORE` (the server reserves the objects
in L1, copies the pieces in, and commits). Query tensors are read-only.

## Copy summary

| Flow | Copies |
| --- | --- |
| store | 2 (CPU tensor → ring pieces, ring → L1) |
| retrieve | 2 (L1 → ring pieces, ring → CPU tensor) |

## Public API

`connect()` lives in `lmcache.sdk.kvcache` (KV) and `lmcache.sdk.qcache` (query); the
`lmcache.sdk.connect(kind, ...)` helper dispatches by `LMCacheSDKCacheKind`. Both return an
`LMCacheSDKContext`, on which retrieve / store / close are **methods**. `LMCacheSDKContext` /
`LMCacheSDKCacheKind` / `LMCacheSDKError` are exported from `lmcache.sdk`.

- **`connect(url, http_url, model_name, timeout=60.0)`** — open the request client, fetch config,
  run the handshake; returns an `LMCacheSDKContext`.
- **`ctx.retrieve(tokens, cache_salt="")`** → contiguous CPU `[2, num_layers, hit_tokens,
  hidden_dim]` for the cached prefix, or `None` (empty/sub-chunk input, or nothing cached).
- **`ctx.store(kv, tokens, cache_salt="")`** → `bool`. `kv` is `[2, L, T, D]`; `len(tokens)`
  must equal `T`; both are truncated to whole chunks before storing.
- **`ctx.close()`** — shut down the request client and release transport resources.

## Cache addressing

Both build an `IPCCacheServerKey(model_name, world_size=1, worker_id=0, token_ids, start=0,
end=<chunk-aligned>, request_id, cache_salt)`. The server resolves it to per-chunk `ObjectKey`s;
**cache identity** = token-chunk hashes + `model_name` + `kv_rank(worker_id)` + `cache_salt`.
`request_id` (`store-`/`retrieve-<uuid>`) only keys the per-request session, not cache identity.

- `worker_id=0` is valid because the SDK is `world_size == 1` (one shard per chunk).
- LOOKUP uses the `worker_id=None` (expand-to-all-workers) variant.

## Constraints & known gaps

- **Dense models only**: a single full-attention kernel group, no sliding window.
- **KV at `world_size == 1`**; query tensors at any TP (read-only).
- **The ring bounds a batch**: a transfer needs free blocks for at least one chunk's objects
  (`lmcache.q.ring_blocks`).
- **Model must be pre-registered** by a vLLM instance with query transfer enabled, and the
  server must load the `qstore` module with the LMCache-driven transfer path.
