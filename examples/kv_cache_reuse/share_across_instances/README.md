# Examples of across-instance KV cache sharing with vLLM + LMCache

> **In-process mode (deprecated):** These workflows use the old controller or
> `lmcache_server`. Prefer MP sharing for new deployments; see the
> [migration guide](https://docs.lmcache.ai/legacy/migration_to_mp.html).

LMCache should be able to reduce the generation time of the second and following calls.

We have examples for the following types of across-instance KV cache sharing:

- KV cache sharing through a centralized cache server: `centralized_sharing/`
- KV cache sharing through p2p cache transfer: `p2p_sharing/`
