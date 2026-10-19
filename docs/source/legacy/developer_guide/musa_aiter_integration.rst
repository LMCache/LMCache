MUSA In-Process Launch
======================

.. warning::

   These commands use the deprecated in-process ``LMCacheConnectorV1``.
   For shared prerequisites and the MP launch, see
   :doc:`/developer_guide/extending_lmcache/musa_aiter_integration`.

After installing LMCache, TorchMUSA, vLLM, and optionally ``musa-aiter`` in
the serving environment, configure the engine:

.. code-block:: bash

   cat > /tmp/lmcache_musa.yaml <<'YAML'
   chunk_size: 256
   local_cpu: true
   max_local_cpu_size: 2
   YAML

   export LMCACHE_CONFIG_FILE=/tmp/lmcache_musa.yaml
   export LMCACHE_MUSA_NATIVE_KV_TRANSFER=1
   # Select an available device in the current container or host.
   export MUSA_VISIBLE_DEVICES=0

   vllm serve /path/to/model \
     --served-model-name qwen3-8b --host 127.0.0.1 --port 18082 \
     --max-model-len 2048 --gpu-memory-utilization 0.35 --enforce-eager \
     --no-enable-prefix-caching \
     --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

``LMCACHE_MP_TRANSFER_MODE`` and ``LMCACHE_MUSA_HANDLE_TRANSFER`` configure
MP transfers; they do not select an in-process transfer mode. Send the
repeated requests from the shared tutorial to validate cache reuse.
