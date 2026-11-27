Chunk-Size Resolution
=====================

Multiprocess connectors resolve an effective LMCache chunk size at startup.
This keeps chunk boundaries aligned with each engine's KV cache geometry
without changing the MP server's configured chunk size. vLLM, SGLang, ATOM,
and TensorRT-LLM support this resolution.

Effective chunk size
--------------------

``lmcache server --chunk-size`` is the server minimum. vLLM and SGLang can
optionally set a client minimum through ``lmcache.mp.chunk_size``. Every
connector reads the server minimum, applies any supported client minimum, and
rounds the result up to the engine's required alignment::

    base = max(server_minimum, client_minimum)
    effective_chunk_size = smallest multiple of required_alignment
                           that is >= base

When the client minimum is omitted, ``base`` is the server minimum. vLLM and
SGLang use the least common multiple of their cacheable KV group spans as the
required alignment. ATOM and TensorRT-LLM use their paged KV block size.

For vLLM attention groups under DCP, the effective span is
``block_size * decode_context_parallel_size``. Recurrent-state groups use
their physical block size because their state is replicated rather than DCP
sharded.

For example, with a server minimum of ``256``:

* an alignment of 256 resolves to an effective chunk size of 256;
* an alignment of 544 resolves to 544;
* an alignment of 544 and a client minimum of 600 resolve to 1088; and
* group spans 32 and 48 have an alignment of 96, so the effective size is 288.

The effective size is registered for ``(model_name, world_size)`` together
with its KV layout. Deployments of the same pair must use the same effective
size while that registration is active. When a vLLM deployment uses multiple
MP servers, every server must resolve the same effective size.

Initialization flow
-------------------

#. The connector computes the engine's required alignment.
#. The scheduler adapter and each worker adapter call ``GET_CHUNK_SIZE`` to
   read the MP server minimum.
#. Each adapter combines the server minimum, optional client minimum, and
   model alignment locally to obtain the effective size.
#. The connector verifies that the result is compatible with every cache
   group; a multi-server scheduler also verifies that all servers agree.
#. Workers register their KV caches and the effective size. The server keeps
   its configured minimum unchanged and stores the size with the registered
   layout.

CacheBlend and Coordinator currently require one global chunk size. When
either is enabled, registrations whose effective size differs from the MP
server's configured ``--chunk-size`` are rejected. An MP server also includes
its configured size in ``POST /instances`` so Coordinator can reject a fleet
configuration mismatch.

Restart recovery
----------------

The vLLM scheduler and worker adapters keep the effective size in memory.
Their heartbeat threads detect an unhealthy-to-healthy transition after an MP
server restart and perform recovery before marking that server healthy:

#. Read the recovered server minimum and resolve the size again from the
   original client minimum and model alignment.
#. Verify that it exactly matches the adapter's cached effective size.
#. On workers, re-register the KV cache handles and any experimental
   dispatcher state.

If any step fails, the server remains unhealthy from that adapter's point of
view and normal cache operations stay in degraded mode. Recovery is attempted
again after the next successful heartbeat. The heartbeat interval defaults to
10 seconds and can be changed with ``lmcache.mp.heartbeat_interval`` in
``kv_connector_extra_config``.
