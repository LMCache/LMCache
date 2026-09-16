# Examples vLLM + LMCache w. remote backends

> **Legacy in-process examples:** The linked workflows use legacy connector
> configuration. Prefer an MP L2 adapter for new deployments where supported;
> see the [migration guide](https://docs.lmcache.ai/legacy/migration_to_mp.html).

LMCache should be able to reduce the generation time of the second and following calls.

We have examples for the following backends:

- Infinistore: `infinistore/`
- Mooncake: `mooncakestore/`
- External: `external/`
