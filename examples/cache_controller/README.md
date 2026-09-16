# Examples of Cache Controller APIs

> **In-process mode (deprecated):** These controller workflows remain available
> for existing deployments. Prefer MP for new deployments where the required
> API is supported; see the [migration guide](https://docs.lmcache.ai/legacy/migration_to_mp.html).

LMCache offers various ochestration APIs which can be used for routing (e.g., KV cache lookup) or hot context migration (e.g., KV cache move/migration).

Here are a few examples:

- [KV cache clear](clear/)
- [KV cache compress](compress/)
- [KV cache lookup](lookup/)
- [KV cache move](move/)
- [KV cache pin](pin/)

Unsupported APIs (WIP):
- [KV cache decompress](decompress/)
- [KV cache unpin](unpin/)
