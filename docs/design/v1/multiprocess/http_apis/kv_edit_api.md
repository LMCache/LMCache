# KV-Cache HTTP API

> Status: v1 contract. Future work is listed at the end and is out of scope
> for this PR.

## Goal

Expose three HTTP endpoints — `POST /api/kv/store`, `POST /api/kv/retrieve`,
and a cheap `POST /api/kv/lookup` — on the LMCache MP server, so that an
external developer can read and write KV cache bytes keyed by a token
sequence. This unblocks cache priming, debugging, and future editing
workflows by making `MemoryObj`s addressable from outside the inference
path.

V1 ships the HTTP transport only. Editing semantics — fake token
sequences, cache-salt namespacing, orchestration-layer version selection
— are explicitly deferred (see §6).

## 1. Plumbing

We add three new methods to `MPCacheEngine` (in `multiprocess/server.py`,
the class instance held at `app.state.engine`):

- `store_bytes_by_tokens(tokens: list[int], payload: bytes, *, model_name: str) -> StoreBytesResult`
- `retrieve_bytes_by_tokens(tokens: list[int], *, model_name: str) -> RetrieveBytesResult`
- `lookup_bytes_by_tokens(tokens: list[int], *, model_name: str) -> LookupBytesResult`

These bypass the GPU connector entirely. They reuse the existing
machinery `MPCacheEngine` already exposes as instance attributes:
`self.token_hasher.compute_chunk_hashes` for chunking and key derivation;
`self.storage_manager.allocate / put / get` for storage; and
`self.gpu_context_meta` (mapping `instance_id` → `(model_name,
world_size)`) for routing. The only new work is copying bytes into / out
of `MemoryObj.byte_array` and looping over worker shards.

Each method takes `model_name` and uses it to pick a matching registered
GPU context — the layout, dtype, and `world_size` come from that context.
Today only one model registers per server in practice, but the API and
internal routing are written so that adding multi-model support later is
purely a deployment / config change, with no method-signature breakage.

Layout: shards are `MemoryFormat.KV_2LTD` — `[2, num_layers, num_tokens, hidden_dim]`.
Aggregation across the full TP group concatenates shards along the
hidden dim (`D`), and chunks along the token dim (`T`). For MLA models,
all TP workers already share one object per chunk (see
`compute_extra_count` at `server.py:75`), so HTTP store / retrieve
operate only on rank 0's shard for those models.

We considered a CPU `GPUConnector` subclass and a ZMQ protocol op for
this layer; the engine-method path was chosen for the smallest blast
radius, lowest risk to the GPU path, and maximum reuse of code already
present in `MPCacheEngine`.

## 2. HTTP surface

Three POST endpoints, served from a new
`lmcache/v1/multiprocess/http_apis/kv_api.py`. POST is used throughout
because token sequences are too large for URL parameters.

**`POST /api/kv/store`** takes `model_name`, `tokens`, and `kv_payload` in
the request body and returns `{status, stored_tokens, stored_chunks}`.

**`POST /api/kv/retrieve`** takes `model_name` and `tokens`, returns the
binary body of whatever the engine had cached, with partial-hit metadata
in `X-LMCache-Hit-Tokens`, `X-LMCache-Hit-Chunks`, and
`X-LMCache-Total-Tokens` response headers. It returns 404 with an empty
body if no chunks hit. Retrieve is **always non-destructive** and ignores
the engine's `remove_after_retrieve` setting.

**`POST /api/kv/lookup`** takes the same body as retrieve and returns
`{hit_tokens, hit_chunks}` without moving payload bytes.

All endpoints return 400 on `model_name` mismatch, 503 if the engine is
not initialized, and (store only) 507 on quota exhaustion.

## 3. Wire format

The payload is a sequence of per-chunk `MemoryObj` byte arrays,
concatenated. Chunk size in tokens is `engine.chunk_size`; chunk size in
bytes is fixed for a given engine (homogeneous attention assumption — see
§6). The HTTP request and response use `Content-Type: application/x-lmcache-kv; v=1`
so the format can evolve later without breaking clients.

## 4. Concurrency

Same model as `/api/clear-cache`: last-writer-wins on store, no
transactional guarantees. A multi-chunk store interrupted mid-way may
leave a partial new prefix in cache. This is acceptable for v1 and
documented; the use cases this API enables (priming, debugging) tolerate
it.

## 5. Rollout

Single end-to-end PR: `MPCacheEngine` bytes methods, `kv_api.py` HTTP
routes, and tests at both layers. If the diff turns out too large for
review, split out the engine methods as a prior PR.

## 6. Future work (not in this PR)

- Editing convention: fake token sequences, orchestration-layer
  virtualization between original and edited caches, `cache_salt` exposed
  as a request field.
- Multi-model serving on a single MP server. The internal routing
  already keys by `model_name`, so this is a deployment change rather
  than an interface change.
- Hybrid-attention payload format with per-layer shape headers.
- Streaming uploads / downloads for very large payloads.
- Auth on the mutating endpoints.
