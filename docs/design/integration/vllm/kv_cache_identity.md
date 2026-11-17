# KV precision identity in the MP connector

`lmcache.mp.enable_kv_dtype_isolation` defaults to `false`. When enabled,
`get_kv_cache_format` projects vLLM's configured model dtype, attention KV
dtype, skipped layers, and two Mamba state dtypes into the engine-neutral
`lmcache.v1.common.cache_format.KVCacheFormat` schema. Its canonical JSON and
BLAKE3 digest produce a versioned `kv-dtype-v1-<128-bit hex>` token. Skipped
layers are a set; cache capacity does not affect the token.

```text
vLLM precision settings
  -> KVCacheFormat token
     -> scheduler: request_configs["lmcache.tag.kv_format"]
        -> existing identity tags -> L1 keys and persistent L2 names
     -> worker: explicit cache_format registration field
        -> registry[(model_name, world_size, token)]
```

The connector injects the tag when creating a request tracker, including after
preemption. Lookup hints and scheduler/worker metadata preserve it, so polling,
store, retrieve, and release carry one identity. A conflicting request value is
overridden with the engine's token, while other tags and request settings are
preserved. The model name and `cache_salt` are not decorated.

Layout registration needs the same token: tags alone would separate the cache
objects but let two workers with one model/world-size pair overwrite each
other's layout. Both transfer modes carry the token in an explicit
`cache_format` registration field, separate from tensor layout hints. The
vLLM worker adapter retains it for initial registration, recovery, and Q ring
registration. The gRPC field is 8 for LMCache-driven KV/Q registration and 11
for engine-driven registration. Refcounts and unregister/reap cleanup select
the same namespace.
Ordinary lookup, unfull-chunk lookup, CacheBlend, source-key resolution, and
management prefetch resolve the format from `kv_format`. Other tags do not
select layouts because request-scoped tenants, revisions, and adapters need
not register separate physical layouts.

The existing registry APIs accept `cache_format` alongside model name and
world size. Lookup, Blend, prefetch, and read-lock cleanup select it from the
request's `kv_format` tag.
CPU residency checks preserve matching tags while expanding ranks and groups;
announcement state is separate for each tagged identity. The legacy event
wire still carries only model names and hashes, so this does not add precision
or tenant filtering to router subscriptions.

The existing ObjectKey tag schema, key codecs, and storage adapters already
preserve the precision identity and need no additional fields. Callers that
omit `cache_format` select the empty layout namespace. Opting in misses
existing untagged entries; there is no legacy read fallback because their byte
encoding is unknown. ZMQ registration now carries eight payload fields, so
clients and servers must use the updated protocol even with isolation disabled.

QStore's `object_layouts` response always includes `cache_format`, including
`""` for the legacy namespace, so readers consume one schema for every layout.

This retains the configured-precision scope of #5615. It does not resolve
physical `fp8` variants across platforms or fingerprint quantization scales,
weights, every tensor layout, or every engine's equivalent configuration.
Those require a separate compatibility contract. In particular, a token is
not proof that independently configured engines or revisions can share bytes.

