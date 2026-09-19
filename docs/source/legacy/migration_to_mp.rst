.. _migration_to_mp:

Migrating from in-process to MP mode
====================================

LMCache's in-process KV-cache runtime is deprecated but remains available for
existing deployments and features that do not yet have MP support. MP mode
runs LMCache as a separately managed ``lmcache server`` and connects each
serving engine to it with that engine's MP connector. Migration is explicit:
LMCache does not start a server, translate configuration, or fall back between
modes automatically.

Start with a parallel deployment
--------------------------------

Keep the current in-process configuration and cache data while validating MP.
The following baseline uses a fresh in-memory L1 cache and matching request
address and chunk size:

.. code-block:: bash

   # Terminal 1
   lmcache server --host 127.0.0.1 --port 5555 \
       --l1-size-gb 4 --eviction-policy LRU --chunk-size 256

For vLLM 0.20.0 and newer, start a separate engine process with LMCache's
shipped connector:

.. code-block:: bash

   # Terminal 2
   vllm serve Qwen/Qwen3-8B --port 8000 \
       --no-enable-prefix-caching \
       --kv-transfer-config \
       '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"127.0.0.1","lmcache.mp.port":5555}}'

``--no-enable-prefix-caching`` is useful during migration testing so vLLM's
own GPU prefix cache cannot hide an LMCache retrieval; it is not required for
normal operation. With vLLM versions before 0.20.0,
``kv_connector_module_path`` cannot select the LMCache-shipped implementation;
``LMCacheMPConnector`` resolves to vLLM's built-in connector instead. Use a
vLLM/LMCache combination with matching protocol support. See the
:doc:`quickstart <../getting_started/quickstart>` for current connector-version
guidance.

Other serving engines use the same server but different client configuration:

* **SGLang:** use a revision that selects ``LMCacheMPConnector``, put
  ``mp_host`` and ``mp_port`` in a new LMCache YAML, and pass the file with
  ``--lmcache-config-file``. These keys address the server; removing them does
  not select the legacy connector. The server ``--chunk-size`` must be a
  multiple of SGLang's ``--page-size``. See the SGLang tab in the
  :doc:`quickstart <../getting_started/quickstart>`.
* **TensorRT-LLM:** select ``connector="lmcache-mp"`` and set
  ``server_url="tcp://127.0.0.1:5555"``. This requires a TensorRT-LLM revision
  with that connector preset and a compatible LMCache adapter, as described
  in the :doc:`quickstart <../getting_started/quickstart>`.

Map the configuration deliberately
----------------------------------

In-process YAML and environment variables configure a cache owned by each
engine process. In MP mode, storage and capacity move to server arguments;
engine configuration selects and addresses the connector.

.. list-table:: Common migration mappings
   :header-rows: 1
   :widths: 28 34 38

   * - In-process setting
     - MP equivalent
     - Migration note
   * - vLLM ``LMCacheConnectorV1`` or its dynamic wrapper
     - ``LMCacheMPConnector`` plus a running ``lmcache server``
     - Preserve the version-specific connector-module selection shown above.
   * - ``chunk_size`` or ``LMCACHE_CHUNK_SIZE``
     - Server ``--chunk-size``
     - Keep it compatible with the serving engine's page or block size.
   * - ``local_cpu`` and ``max_local_cpu_size``
     - MP L1, including ``--l1-size-gb``
     - Size the shared server pool; it is not a per-worker allocation.
   * - ``local_disk`` or a legacy filesystem backend
     - A supported MP L2 adapter, for example
       ``--l2-adapter '{"type":"fs_native","base_path":"/data/lmcache-mp"}'``
     - Start with a fresh directory and follow
       :doc:`MP L2 storage <../mp/l2_storage/index>`.
   * - ``remote_url`` or a legacy remote backend
     - The corresponding supported MP L2 adapter
     - Select and configure the adapter explicitly; this is not a URL rename.
   * - ``use_layerwise``, blending, compression, or model-specific state
     - Feature-specific MP configuration, when supported
     - Check the current MP feature page; there is no automatic key mapping.

A single MP server's L1 limit is shared by its connected engines. It is not
capacity-equivalent to repeating a legacy per-worker limit on every worker.
Choose the shared limit from the desired aggregate capacity and expected
concurrency.

Do not point MP at an existing legacy cache directory unless the selected
adapter explicitly documents format compatibility. Key and file compatibility
is not guaranteed. Use a fresh path during validation, and never delete or
format the legacy cache as part of migration.

Verify a real store and retrieval
---------------------------------

With the vLLM command above running, create two requests with a shared prefix
longer than the 256-token chunk size:

.. code-block:: bash

   python - <<'PY'
   import json

   prefix = "LMCache migration verification uses a deliberately long prefix. " * 160
   for number, suffix in ((1, "First request."), (2, "Second request.")):
       with open(f"/tmp/lmcache-request-{number}.json", "w") as request_file:
           json.dump({
               "model": "Qwen/Qwen3-8B",
               "prompt": prefix + suffix,
               "max_tokens": 8,
               "temperature": 0,
           }, request_file)
   PY

   curl http://127.0.0.1:8000/v1/completions \
       -H 'Content-Type: application/json' \
       --data-binary @/tmp/lmcache-request-1.json

Wait until the ``lmcache server`` terminal reports ``Stored <N> tokens`` before
sending the second request:

.. code-block:: bash

   curl http://127.0.0.1:8000/v1/completions \
       -H 'Content-Type: application/json' \
       --data-binary @/tmp/lmcache-request-2.json

The second request must produce ``Retrieved <N> tokens`` in the LMCache server
log. A lower time-to-first-token by itself is not proof of an LMCache hit.

Check feature and version gaps
------------------------------

.. list-table:: Current migration considerations
   :header-rows: 1
   :widths: 24 20 56

   * - Feature
     - MP status
     - Guidance
   * - CacheGen compression
     - Unavailable
     - MP CacheGen is not implemented. Continue using the in-process
       implementation if it is required; see :doc:`../mp/cachegen`.
   * - Encoder or multimodal encoder-output caching
     - Unavailable
     - MP can reuse supported decoder KV for multimodal requests, but the
       separate encoder cache remains in-process only. See
       :doc:`../non_kv_cache/encoder_cache`.
   * - SGLang MUSA MLA and layerwise transfer
     - Platform-dependent
     - MP MUSA supports MHA with separate K/V pools and requires the opt-in
       handle-transfer setup. On MUSA, multiprocess MLA is not supported;
       in-process supports non-layerwise MLA and layerwise MHA, while
       layerwise MLA is unsupported. See
       :doc:`../developer_guide/integration`.
   * - Hidden-state caching
     - Unverified in MP
     - The current hidden-state store is configured through the in-process
       engine and is not documented or wired as an MP feature. Keep the
       in-process path when hidden states are required.
   * - CacheBlend
     - Available, differently configured
     - Use the MP blend engine rather than legacy blending keys; see
       :doc:`../kv_cache_optimizations/cacheblend`.
   * - TensorRT-LLM connector presets
     - Version-dependent
     - Check that the TensorRT-LLM revision provides the ``lmcache-mp`` preset
       and is compatible with the installed LMCache adapter.

Check the MP documentation and tests before migrating an unlisted integration
or feature. The in-process runtime remains available when a required feature
has no MP equivalent.

Rollback
--------

1. Stop the serving engine that uses the MP connector.
2. Restore its original in-process connector and YAML or ``LMCACHE_*``
   environment configuration.
3. Restart the serving engine and verify a normal cache store and hit.
4. Stop the standalone LMCache server after no clients use it.

Keep the original configuration and cache data until MP validation is
complete. Rollback changes the runtime wiring; it does not require deleting
either cache.
