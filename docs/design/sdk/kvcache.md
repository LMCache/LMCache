# KV Cache SDK

> Status: v1 client interface for the MP HTTP KV cache protocol.

## Goal

Provide a small Python SDK surface for moving KV cache files into and out
of an LMCache MP server:

```python
import lmcache.sdk as lmc_sdk

lmc_sdk.store("kv.pt", "http://localhost:8080")
lmc_sdk.retrieve(
    "kv-hit.pt",
    "http://localhost:8080",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    tokens=[1, 2, 3],
)
lmc_sdk.lookup(
    "http://localhost:8080",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    tokens=[1, 2, 3],
)
```

This PR intentionally does not add a CLI. A future CLI should call these
SDK functions rather than duplicate protocol code.

## File Contract

`store` accepts `.pt` or `.safetensors` packages. The package must contain
a 4-D tensor named `kv` in canonical `KV_2LTD` layout:

```text
[2, num_layers, total_tokens, hidden_dim]
```

The SDK also needs `model_name`, `tokens`, and optional `cache_salt`.
Callers can pass those values explicitly, or store them in file metadata:

- `.pt`: a mapping with `kv`, `model_name`, `tokens`, and `cache_salt`.
- `.safetensors`: tensor `kv`, with string metadata for `model_name`,
  JSON-encoded `tokens`, and `cache_salt`.

Retrieve writes the same package shape and metadata. The saved token list
is truncated to the retrieved hit prefix.

## Protocol Ownership

The SDK imports the versioned frame helpers from
`lmcache.v1.multiprocess.http_apis.kv_protocol`. It does not open-code
frame JSON or byte prefixes. This keeps SDK, HTTP handlers, and tests on
one protocol definition.

Store flow:

1. Load and validate the local package.
2. Read `/api/status` to learn `chunk_size`.
3. Send a `store_manifest` frame.
4. Stream one `store_chunk` frame per complete token chunk.

Retrieve flow:

1. Send the JSON retrieve request with `protocol_version: 1`.
2. Decode the `retrieve_manifest`.
3. Allocate the full hit-prefix tensor on CPU.
4. Copy each `retrieve_shard` payload into the correct token and
   tensor-parallel hidden-dimension slice.
5. Save the assembled package.

Lookup sends the same JSON request as retrieve and returns the metadata
JSON directly as a typed result.

## Error Handling

All public SDK functions raise `KVCacheSDKError` for invalid packages,
invalid server JSON, protocol failures, and non-2xx HTTP responses. The
SDK does not retry writes; callers that need retry semantics should do so
at the workflow layer because repeated store calls can partially overwrite
the same cache prefix.
