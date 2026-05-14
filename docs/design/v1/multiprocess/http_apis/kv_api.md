# KV-Cache HTTP API

> Status: v1 contract. Future work is listed at the end and is out of scope
> for this PR.

## Goal

Expose two HTTP endpoints - `POST /api/kv/store` and
`POST /api/kv/retrieve` - on the LMCache MP server so an external client
can store and retrieve KV cache
bytes keyed by a token sequence.

V1 keeps the server transport as HTTP. The client-facing interface added
by this PR is the Python SDK (`lmcache.sdk`); a CLI can be layered on the
same protocol later without changing the server contract.

## 1. Engine Surface

The HTTP API uses two bytes-level `MPCacheEngine` methods:

- `store_kv_bytes_by_tokens(...) -> StoreBytesResult`
- `retrieve_kv_bytes_by_tokens(...) -> RetrieveBytesResult`

The storage implementation lives in `lmcache/v1/multiprocess/kv_bytes.py`
and the `MPCacheEngine` methods are thin wrappers. This keeps the server
integration point small while still exposing the engine-owned dependencies
needed for token hashing, model layout resolution, and storage access.

Both methods bypass the GPU connector and reuse existing MP engine
machinery:

- `token_hasher.compute_chunk_hashes` for chunking and key derivation.
- `storage_manager.reserve_write`, `finish_write`, prefetch, and read-lock
  APIs for storage.
- `gpu_context_meta` and `gpu_contexts` to resolve the registered model
  layout, dtype, and tensor-parallel `world_size`.

Layout: v1 supports one homogeneous KV layer group in `KV_2LTD` layout,
`[2, num_layers, num_tokens, hidden_dim]`. The full hidden dimension is
split across tensor-parallel workers along `D`. MLA models already share
one object per chunk across TP workers, so the existing layout metadata is
used to decide what is valid for the registered model.

## 2. Streaming Behavior

### Store

The client streams one full-token chunk at a time. Each chunk payload has
shape `[2, num_layers, chunk_size, full_hidden_dim]` in row-major order
and the registered model KV dtype.

The server validates the stream metadata once, then processes each chunk
independently:

1. Reserve the object keys for that chunk.
2. Reinterpret the chunk bytes as a CPU tensor.
3. Split along `D` into one shard per TP worker.
4. Copy each shard directly into its reserved `MemoryObj`.
5. Finish the write reservation for that chunk.

The server never concatenates the full `[2, L, T, D]` request tensor.

### Retrieve

The engine returns a `RetrieveBytesResult` with prefix metadata plus a
lazy iterator over `KVBytesShard` objects. The HTTP layer first sends a
manifest frame, then sends one shard frame per cached `MemoryObj`.

The client is responsible for assembling the returned shard frames into
the full `[2, L, hit_tokens, full_hidden_dim]` tensor. This keeps memory
pressure on the LMCache server bounded by a single stored shard payload
during response generation.

Retrieve is non-destructive and ignores the engine's
`remove_after_retrieve` setting. Callers must consume the shard iterator
or call `RetrieveBytesResult.close()` so storage read locks are released.

## 3. Versioned Wire Protocol

The wire protocol is isolated in
`lmcache/v1/multiprocess/http_apis/kv_protocol.py`.

V1 constants:

- `PROTOCOL_VERSION = 1`
- `STREAM_MEDIA_TYPE = "application/x-lmcache-kv-stream; v=1"`

Each binary frame is:

```text
uint32_be header_length
uint64_be payload_length
header_length bytes of UTF-8 JSON header
payload_length bytes of binary payload
```

Every header contains:

- `version`: protocol version, currently `1`.
- `type`: one of the frame type strings below.
- `payload_length`: binary payload length repeated for validation.

Frame types:

- `store_manifest`: first frame in a store request. Carries `model_name`,
  `tokens`, `cache_salt`, full tensor `shape`, and `dtype`.
- `store_chunk`: store payload frame. Carries `chunk_index` and one full
  token chunk payload.
- `retrieve_manifest`: first frame in a retrieve response. Carries prefix
  metadata, chunk size, TP world size, full hit shape, shard shape, and
  dtype.
- `retrieve_shard`: retrieve payload frame. Carries `chunk_index`,
  `worker_id`, and one worker shard payload.

Protocol evolution should add new frame types or a new version in this
module. HTTP handlers and SDK code should depend on the typed encode/decode
helpers rather than open-coding frame JSON.

## 4. HTTP Surface

Two POST endpoints are served from
`lmcache/v1/multiprocess/http_apis/kv_api.py`. POST is used because token
sequences are too large for URL parameters.

`POST /api/kv/store` consumes
`Content-Type: application/x-lmcache-kv-stream; v=1`. The request body is
one `store_manifest` frame followed by ordered `store_chunk` frames. The
response is JSON:

```json
{
  "status": "ok",
  "total_tokens": 768,
  "total_chunks": 3,
  "stored_tokens": 768,
  "stored_chunks": 3
}
```

`stored_chunks` is the leading complete prefix that landed. It can be less
than `total_chunks` under write conflicts or capacity pressure.

`POST /api/kv/retrieve` consumes JSON:

```json
{
  "model_name": "meta-llama/Llama-3.1-8B-Instruct",
  "tokens": [1, 2, 3],
  "cache_salt": "",
  "protocol_version": 1
}
```

It returns `Content-Type: application/x-lmcache-kv-stream; v=1` with one
`retrieve_manifest` frame followed by zero or more `retrieve_shard` frames.
A miss still returns `200 OK`; the manifest reports `hit_chunks: 0` and an
empty full shape.

All endpoints return `400` for unknown models, unsupported protocol
versions, invalid metadata, or unsupported registered layouts, and `503`
when the engine is not initialized.

## 5. Client SDK

This PR adds `lmcache.sdk` as the supported client interface:

```python
import lmcache.sdk as lmc_sdk

kv = lmc_sdk.retrieve(
    "http://localhost:8080",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    tokens=[1, 2, 3],
)
if kv is not None:
    lmc_sdk.store(
        kv,
        "http://localhost:8080",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        tokens=[4, 5, 6],
    )
```

The SDK owns protocol framing and client-side assembly, but not file storage.
It accepts an in-memory `torch.Tensor`, streams store chunks to the server,
decodes retrieve frames, and concatenates returned worker shards into an
in-memory tensor. Cache identity (`model_name`, `tokens`, and `cache_salt`) is
always passed explicitly to the SDK call. Callers that need files can serialize
the tensor and metadata outside `lmcache.sdk`.

## 6. Concurrency

Store uses chunk-scoped write reservations. A multi-chunk store interrupted
mid-way may leave a partial new prefix in cache. This is acceptable for v1:
the intended workflows are cache priming, debugging, and offline editing,
which can tolerate prefix-granularity partial success.

Retrieve holds read locks while the lazy shard iterator is active. The HTTP
handler releases those locks when the stream finishes or is closed.

## 7. V1 Limitations

V1 rejects unsupported layouts before acquiring storage locks:

1. Exactly one KV layer group is required. Hybrid attention with multiple
   KV layer groups needs a future protocol with per-group framing.
2. The registered layout must be `KV_2LTD`.
3. Store input must cover complete token chunks. Partial trailing tokens are
   ignored for keying and must not be present in the payload tensor.

## 8. Future Work

- CLI on top of the same protocol. A future CLI can own its file format and
  feed an in-memory tensor plus explicit metadata into the SDK.
- Editing conventions: fake token sequences, orchestration-layer
  virtualization between original and edited caches, and version selection.
- Hybrid-attention payload format with per-layer-group frame metadata.
- Authentication and authorization for mutating endpoints.
- Protocol v2 when a compatibility break is worth the migration cost.
