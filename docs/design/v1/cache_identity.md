# Versioned cache representation identity

## Problem

The same model name and token prefix do not imply reusable KV bytes. A weight
sync changes the values, and quantization, compression, hybrid topology, or
token dropping changes their representation. Reusing a cache entry across one
of those boundaries can silently return stale or undecodable KV.

LMCache already keys each entry by model name and a rolling token digest. This
design adds the missing revision dimensions without duplicating those existing
fields.

## Identity model

`BaseCacheIdentity` describes the semantic source of the values:

- model revision;
- tokenizer revision;
- weight revision;
- optional adapter/LoRA revision.

`CacheRepresentationIdentity` describes how those values are represented:

- topology fingerprint;
- attention backend revision;
- KV dtype;
- optional quantization revision;
- optional compression revision;
- optional token-drop algorithm and policy revisions.

The drop algorithm and policy revisions form one compatibility boundary and
must therefore be set together. Every value is treated as an opaque,
case-sensitive revision. LMCache never trims or otherwise normalizes one.

`CacheIdentity` combines both halves. Its `revision` is a versioned SHA-256
digest of canonical JSON, so mapping insertion order cannot alter the result.
Callers can use `mismatched_fields()`, `is_compatible_with()`, or
`require_compatible()` before reusing a representation outside the regular
storage-key path.

## Request configuration

The identity can be attached to vLLM `kv_transfer_params`. All required fields
must be supplied once any `lmcache.cache_identity.*` field is present:

```json
{
  "kv_transfer_params": {
    "lmcache.cache_identity.model_revision": "deepseek-v4@4e6f1a2",
    "lmcache.cache_identity.tokenizer_revision": "tokenizer@8d91c70",
    "lmcache.cache_identity.weight_revision": "rollout-step-4200",
    "lmcache.cache_identity.adapter_revision": "policy-lora@17",
    "lmcache.cache_identity.topology_fingerprint": "mla:61,attn:3,tp:2",
    "lmcache.cache_identity.backend_revision": "flashmla@2.3",
    "lmcache.cache_identity.kv_dtype": "fp8_e4m3",
    "lmcache.cache_identity.quantization_revision": "per-head-scale-v2",
    "lmcache.cache_identity.drop_algorithm_id": "dapo-attention-score",
    "lmcache.cache_identity.drop_policy_revision": "keep-0.75@3"
  }
}
```

The same complete identity must accompany lookup, store, retrieve, and cleanup
operations for a request. Unknown fields, non-string values, missing required
fields, and half-specified drop policies fail closed.

LMCache materializes the structured fields into an internal
`lmcache.tag.cache_identity_revision` digest once per request. Callers should
not set that tag directly, and there is deliberately no public precomputed
revision field: the structured values remain the source of truth at the API
boundary.

## Key integration

The in-process path projects the canonical revision into one internal
`CacheEngineKey` tag. Its existing string and dictionary encodings therefore
round-trip the revision without a new key schema.

The multiprocess path derives the revision in `IPCCacheServerKey` and binds it
to every raw chunk hash with a domain-separated SHA-256 operation before
constructing `ObjectKey`. This central conversion point covers L1 and every L2
adapter without changing `ObjectKey`, adapter filenames, or coordinator wire
schemas.

When no identity is configured, both paths retain their legacy key values
exactly. This avoids invalidating existing deployments. It also means that
unversioned clients remain in the legacy compatibility domain; operators must
roll out identity configuration consistently when stale cross-revision reuse
is unacceptable.

## Scope and invariants

- Identity revisions are cache-compatibility fences, not tenant authorization.
  Use `cache_salt` for tenant isolation.
- The token digest remains in the existing chunk hash; identity only adds
  revision dimensions.
- A weight update must produce a new `weight_revision` before the updated
  worker performs cache lookup or store.
- A topology, backend, dtype, quantization, compression, or token-drop policy
  change must produce a new representation identity.
- An empty identity retains legacy behavior; a partial identity is rejected.
