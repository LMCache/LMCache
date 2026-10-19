vLLM Connector Loading
======================

vLLM can load an external KV connector through ``kv_connector_module_path``.
The module and class must match the deployment mode.

Multiprocess mode
-----------------

For vLLM versions supporting the external MP connector, use:

.. code-block:: bash

   vllm serve YOUR_MODEL \
     --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both"}'

Start the standalone LMCache server first, as shown in
:doc:`/getting_started/quickstart`. See
:doc:`/getting_started/compatibility` for the vLLM version boundary and the
bundled-versus-external connector behavior.

In-process mode (deprecated)
----------------------------

``LMCacheConnectorV1`` and ``LMCacheConnectorV1Dynamic`` embed LMCache in the
serving engine. Their import compatibility and dynamic loading examples are
kept in :doc:`/legacy/api_reference/dynamic_connector`.
