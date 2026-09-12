.. _recipe_glm5_2:

GLM 5.1/5.2/5.3
===============

A large Mixture-of-Experts model using **Dynamic Sparse Attention (DSA)**, shared
by the **GLM-5.1 / GLM-5.2 / GLM-5.3** series. Like DeepSeek-V4-Flash, the sparse-attention path
splits the model's layers into more than one KV cache group; the
``LMCacheMPConnector`` stores and retrieves each group in its own block size, so
KV reuse works without extra flags.

Validated models
----------------

- `zai-org/GLM-5.2-FP8 <https://huggingface.co/zai-org/GLM-5.2-FP8>`_ (8 GPUs)
- GLM-5.3 (NVFP4 checkpoint, 8x B200, TP8 + expert parallel, ``--kv-cache-dtype fp8_e4m3``)

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `GLM-5.2 in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#text-generation>`_
      (architecture ``GlmMoeDsaForCausalLM``). See also the
      `vLLM GLM-5.2 recipe <https://recipes.vllm.ai/zai-org/GLM-5.2>`_.

      **Status:** Validated with LMCache (GLM-5.2: vLLM 0.23.0 + LMCache 0.4.7;
      GLM-5.3: vLLM 0.28.0 + LMCache 0.5.5rc5, needle retrieval correct at 19k,
      55k and 92k tokens when the KV is restored from LMCache).

      Start the LMCache MP server:

      .. code-block:: bash

         lmcache server \
             --port 6555 \
             --max-workers 8 \
             --l1-size-gb 100 \
             --eviction-policy LRU \
             --chunk-size 1024

      |

      Start vLLM with the LMCache MP connector (8 GPUs):

      .. code-block:: bash

         vllm serve zai-org/GLM-5.2-FP8 \
             --tensor-parallel-size 8 \
             --tool-call-parser glm47 \
             --enable-auto-tool-choice \
             --reasoning-parser glm45 \
             --no-enable-prefix-caching \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.port":6555}}'

      |

      ``--tool-call-parser glm47``, ``--enable-auto-tool-choice``, and
      ``--reasoning-parser glm45`` are GLM-5.2's serving requirements (see the
      vLLM recipe). ``--no-enable-prefix-caching`` routes all KV reuse through
      LMCache rather than vLLM's in-engine prefix cache. The server's
      ``--port 6555`` must match ``lmcache.mp.port`` in the connector config;
      ``--max-workers`` is set to the tensor-parallel size. Adjust
      ``--tensor-parallel-size`` to match your hardware. For the generic
      LMCache + vLLM wiring (ports, remote hosts), see
      :doc:`../getting_started/quickstart`.

      If there are any issues with vLLM setup, please refer to the
      `vLLM Recipes <https://docs.vllm.ai/projects/recipes/en/latest/index.html>`_
      for more details.

   .. tab-item:: SGLang

      **Status:** Not validated with LMCache.

   .. tab-item:: TRT-LLM

      **Status:** Supported. See :doc:`../getting_started/quickstart` for TRT-LLM + LMCache setup.

CacheBlend support
------------------

Not validated.

Compression support
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Method
     - Status
     - Notes
   * - :doc:`CacheGen <../kv_cache_optimizations/compression/cachegen>`
     - Not validated
     -

MTP (speculative decoding) support
----------------------------------

GLM-5.2 ships a native multi-token-prediction head
(``num_nextn_predict_layers: 1``), usable for speculative decoding via
vLLM's ``--speculative-config '{"method":"mtp","num_speculative_tokens":1}'``.
LMCache accounts for the MTP draft layer's KV cache automatically.

**Status:** Not yet validated with LMCache (validation requires an
8-GPU node).

Caveats
-------

- **Dynamic Sparse Attention KV groups.** The DSA path registers a sparse-attention
  indexer k-cache (uint8, 132 bytes per token) per sparse layer beside the MLA cache.
  ``LMCacheMPConnector`` stores and restores both (the server log shows two
  ``KernelGroupInfo`` groups, ``hs=576`` and ``hs=132``); no extra flags are required.
  ``--chunk-size`` must be a multiple of the engine block size (64 for DSA models).
- **In-process connector not supported.** ``LMCacheConnectorV1`` /
  ``LMCacheConnectorV1Dynamic`` cannot carry the indexer caches and refuse to start
  on DSA models. Restoring only the MLA cache is not an option: every KV hit longer
  than the indexer top-k (2,048 tokens) then decodes garbage.
