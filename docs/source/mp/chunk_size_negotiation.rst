Multiple Chunk Sizes in MP Mode
===============================

One LMCache multiprocess (MP) server can serve model deployments with different
KV chunk sizes when using the default prefix-caching engine without a
Coordinator. The vLLM, SGLang, ATOM, and TensorRT-LLM MP integrations resolve
an effective chunk size at startup that aligns with each model's KV cache
blocks.

Effective chunk size
--------------------

``lmcache server --chunk-size`` sets the server-wide minimum in tokens and
defaults to ``256``. Each connector starts with that minimum. If a client
minimum is set, the connector takes the larger of the two. It then rounds up
to the smallest multiple of the model's required alignment. Set each minimum
as a positive integer.

vLLM and SGLang align to the least common multiple of the token spans of the
cacheable KV groups. ATOM and TensorRT-LLM align to their paged KV block size.

For vLLM attention groups under DCP, the effective span is
``block_size * decode_context_parallel_size``. Recurrent-state groups use their
physical block size because each DCP rank holds a replica of the state.

Configuring each integration
----------------------------

**vLLM** can set ``lmcache.mp.chunk_size`` in ``kv_connector_extra_config``.
For example, pass this JSON to ``vllm serve --kv-transfer-config`` to use the
connector shipped with LMCache and request a minimum of 512 tokens.

.. code-block:: json

   {
     "kv_connector": "LMCacheMPConnector",
     "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
     "kv_role": "kv_both",
     "kv_connector_extra_config": {
       "lmcache.mp.host": "localhost",
       "lmcache.mp.port": 5555,
       "lmcache.mp.chunk_size": 512
     }
   }

**SGLang** can set the same key under ``extra_config`` in the YAML file
passed to ``--lmcache-config-file``. This example requests a minimum of 300
tokens.

.. code-block:: yaml

   mp_host: 127.0.0.1
   mp_port: 5555
   extra_config:
     lmcache.mp.chunk_size: 300

**ATOM** uses the server minimum and rounds it up to a multiple of the ATOM
block size. **TensorRT-LLM** also uses the server minimum and rounds it up to
a multiple of ``KvCacheConfig.tokens_per_block``. Both adapters perform this
alignment automatically.

Chunk-size examples
-------------------

With ``--chunk-size 256``, these example configurations produce the following
effective sizes in tokens.

.. list-table::
   :header-rows: 1

   * - Integration
     - Block span or spans
     - Client minimum
     - Effective chunk size
   * - vLLM
     - 16
     - 512
     - 512
   * - vLLM (hybrid model)
     - 544
     - 600
     - 1088
   * - SGLang
     - 32 and 48 (alignment 96)
     - 300
     - 384
   * - ATOM
     - 128
     - Uses server minimum
     - 256
   * - TensorRT-LLM
     - 64
     - Uses server minimum
     - 256

In the hybrid vLLM example, one 544-token block falls below the 600-token
minimum, so the connector uses two blocks (1088 tokens).

Initialization flow
-------------------

#. The connector reads the MP server minimum with ``GET_CHUNK_SIZE``.
#. It combines the server minimum with the optional client minimum and rounds
   up to the required block alignment.
#. The connector checks the effective size against the engine's KV block
   geometry. A vLLM scheduler using ``lmcache.mp.server_urls`` also checks
   that every server it uses resolves to the same effective size.
#. Each worker registers its KV cache with the effective size. The MP server
   validates the size and records it with the KV layout.

Sharing one server
------------------

For example, two vLLM deployments with different model names and 16-token
blocks can use 256-token and 512-token chunks on the same server. Start the
server with ``--chunk-size 256``, leave the first deployment's client minimum
unset, and set it to ``512`` on the second.

All deployments registered under the same ``(model_name, world_size)`` pair
on one server must agree on the effective size. To change it, shut down all
engine deployments using that pair before starting workers with the new
size. The server releases the layout when the last worker unregisters during
shutdown or the server reaps its stale registration. If a worker exits without
unregistering, wait for cleanup under ``--worker-reap-timeout-seconds`` or
``--worker-registration-grace-seconds`` (see :doc:`configuration`). If reaping
is disabled and a stale registration remains, restart the MP server before
starting the replacement workers.

.. _mp-global-chunk-size:

CacheBlend and Coordinator
--------------------------

:doc:`CacheBlend </kv_cache_optimizations/cacheblend>` and
:doc:`Coordinator </mp/coordinator>` each require a single chunk size. With
``--engine-type blend`` or a configured ``--coordinator-url``, the MP server
rejects any KV cache registration whose effective size differs from its
``--chunk-size``. Choose a server size that is a multiple of every connected
model's required alignment and at least as large as every configured client
minimum.

When using Coordinator, also set the same ``--chunk-size`` on the Coordinator
and all MP servers in the fleet. Coordinator rejects a server registration with
HTTP 409 if the server reports a different size.

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
