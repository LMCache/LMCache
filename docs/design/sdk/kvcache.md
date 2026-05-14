# KV Cache SDK

> Status: v1 client interface for the MP HTTP KV cache protocol.

## Goal

Provide a small Python SDK surface for moving KV cache tensors into and out
of an LMCache MP server without forcing applications through local storage:

```python
import lmcache.sdk as lmc_sdk

result = lmc_sdk.retrieve(
    "http://localhost:8080",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    tokens=[1, 2, 3],
)

lmc_sdk.store(
    result.package,
    "http://localhost:8080",
    tokens=[4, 5, 6],
)
```

This PR intentionally does not add a CLI. A future CLI should call these
SDK functions rather than duplicate protocol code.

## Memory Contract

The primary SDK payload is `KVCachePackage`, an in-memory object containing:

- `kv`: a 4-D tensor in canonical `KV_2LTD` layout
  `[2, num_layers, total_tokens, hidden_dim]`.
- `model_name`: registered LMCache model name.
- `tokens`: token IDs addressed by the tensor.
- `cache_salt`: optional namespace salt.

`retrieve` returns the assembled package in `result.package`. `store` accepts
that package directly and streams its tensor bytes to the server. This keeps
edit workflows in memory and avoids an unnecessary `torch.save` / `torch.load`
round trip. Callers that need durability or offline interchange should serialize
their own `KVCachePackage` fields outside the SDK.

## Protocol Ownership

The SDK imports the versioned frame helpers from
`lmcache.v1.multiprocess.http_apis.kv_protocol`. It does not open-code
frame JSON or byte prefixes. This keeps SDK, HTTP handlers, and tests on
one protocol definition.

Store flow:

1. Resolve metadata overrides and validate the in-memory package.
2. Read `/api/status` to learn `chunk_size`.
3. Send a `store_manifest` frame.
4. Stream one `store_chunk` frame per complete token chunk.

Retrieve flow:

1. Send the JSON retrieve request with `protocol_version: 1`.
2. Decode the `retrieve_manifest`.
3. Allocate the full hit-prefix tensor on CPU.
4. Copy each `retrieve_shard` payload into the correct token and
   tensor-parallel hidden-dimension slice.
5. Return the assembled package in memory.

## Error Handling

All public SDK calls raise `KVCacheSDKError` for invalid packages,
invalid server JSON, protocol failures, and non-2xx HTTP responses. The
SDK does not retry writes; callers that need retry semantics should do so
at the workflow layer because repeated store calls can partially overwrite
the same cache prefix.
