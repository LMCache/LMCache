Chunk-Size Resolution
=====================

Multiprocess connectors resolve an effective chunk size at startup. This keeps
chunk boundaries aligned with each engine's KV cache geometry while allowing
one MP server to serve models that use different chunk sizes. vLLM, SGLang,
ATOM, and TensorRT-LLM support this resolution.

Effective chunk size
--------------------

``lmcache server --chunk-size`` is the server minimum. vLLM and SGLang can
optionally set a client minimum through ``lmcache.mp.chunk_size``. Every
connector reads the server minimum, applies any supported client minimum, and
rounds the result up to the engine's required alignment::

    base = max(server_minimum, client_minimum)  # client minimum is optional
    effective_chunk_size = round_up(base, required_alignment)

For vLLM and SGLang, the required alignment is the least common multiple of the
effective token span of every cacheable KV group. For vLLM attention groups
under DCP, that span is ``block_size * decode_context_parallel_size``.
Recurrent-state groups use their physical block size because their state is
replicated rather than DCP sharded. ATOM and TensorRT-LLM align to their paged
KV block size.

For example, with a server minimum of 256 tokens:

* no client minimum and an alignment of 80 resolves to 320;
* a client minimum of 128 and an alignment of 16 resolves to 256;
* a client minimum of 500 and an alignment of 16 resolves to 512; and
* a client minimum of 300 and an alignment of 80 resolves to 320.

The server minimum remains 256 in every case.

Initialization flow
-------------------

#. The connector computes the engine's required alignment.
#. The scheduler and every worker call ``GET_CHUNK_SIZE`` to read the server
   minimum, even when ``lmcache.mp.chunk_size`` is configured.
#. Each process computes the same effective chunk size locally.
#. The scheduler uses that size for lookup and scheduling. Workers include it
   when registering their KV caches.
#. The server stores the size, token hasher, and layouts under
   ``(model_name, world_size)``. Every rank sharing that key must register the
   same size.

The final unregister removes the registry entry. A later deployment may then
register the same ``(model_name, world_size)`` with a different chunk size.

Multiple MP servers
-------------------

A vLLM scheduler connected to multiple MP servers requires every server to
resolve to the same effective size. This preserves the common lookup
granularity used when per-server hit lengths are combined.

Coordinator and CacheBlend
--------------------------

Coordinator and CacheBlend still use one server-wide chunk size and do not yet
support model-specific effective sizes. When either feature is enabled, the MP
server rejects a KV-cache registration whose effective size differs from its
configured ``--chunk-size``. Configure that value to satisfy every connected
model. Normal MP prefix caching does not have this restriction.

Restart recovery
----------------

The vLLM scheduler and worker adapters keep the effective size in memory. After
an MP server restart, heartbeat recovery reads the server minimum again,
recomputes the effective value, and verifies that it matches the cached value
before KV caches are re-registered. A mismatch keeps the adapter unhealthy so
requests cannot silently use different chunk boundaries.
