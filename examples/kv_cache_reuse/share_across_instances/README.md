# Examples of across-instance KV cache sharing with vLLM + LMCache
LMCache should be able to reduce the generation time of the second and following calls.

These directories document **in-process** sharing (`LMCacheConnectorV1`).
The recommended path is multiprocess mode: one `lmcache server` shared by
two engines. See
[Share KV cache across engines (MP mode)](https://docs.lmcache.ai/getting_started/quickstart/share_kv_cache_mp.html).

We have examples for the following types of across-instance KV cache sharing:

- KV cache sharing through a centralized cache server: `centralized_sharing/`
- KV cache sharing through p2p cache transfer: `p2p_sharing/`

In `centralized_sharing/`, the sample `curl` repeats the prompt so it
exceeds `chunk_size` (256). A one-sentence prompt is not stored by
default (`save_unfull_chunk` is false), so the second engine would miss.