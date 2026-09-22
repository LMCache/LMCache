.. _recipe_glm5_3:

GLM-5.3
=======

A large Mixture-of-Experts model using **Dynamic Sparse Attention (DSA)**,
continuing the :doc:`GLM-5.1 / GLM-5.2 <glm5_2>` line (architecture
``GlmMoeDsaForCausalLM``, 78 layers, ``index_topk`` 2048). Unlike
:doc:`DeepSeek-V4.1-Flash <deepseek_v41_flash>`, this checkpoint presents a
**single KV cache group** to the connector, so no per-group tuning is needed.

Validated models
----------------

- `zai-org/GLM-5.3 <https://huggingface.co/zai-org/GLM-5.3>`_ (8 GPUs)

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `GLM-5.3 in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#text-generation>`_
      (architecture ``GlmMoeDsaForCausalLM``).

      **Status:** Validated with LMCache.

      Start the LMCache MP server:

      .. code-block:: bash

         lmcache server \
             --chunk-size 256 \
             --separate-object-groups \
             --l1-size-gb 100 \
             --eviction-policy LRU

      |

      Start vLLM with the LMCache MP connector (8 GPUs):

      .. code-block:: bash

         vllm serve zai-org/GLM-5.3 \
             --tensor-parallel-size 8 \
             --enable-prefix-caching \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      **Chunk size.** GLM-5.3 reports a single KV cache group with
      ``tokens_per_block = 64``, so ``--chunk-size`` must be a multiple of
      ``64``. Both ``256`` and ``1024`` were validated; prefer the smaller
      value, because only **whole** chunks are stored and the remainder is
      recomputed on every request. For a 5040-token prompt, ``--chunk-size
      1024`` caches 4096 tokens (81%) while ``256`` caches 4864 (97%) -- a
      measured external-hit-rate difference of 40.6% vs 48.3% on that prompt.
      Choose a chunk size well below your typical prompt length.

      The connector logs the geometry it resolved at startup as
      ``group_tokens_per_block=[...], scheduler_block_size=...`` -- check that
      line if you are unsure what a given checkpoint reports.

      Adjust ``--tensor-parallel-size`` to match your hardware. For the generic
      LMCache + vLLM wiring (ports, remote hosts), see
      :doc:`../getting_started/quickstart`. GLM's serving extras
      (``--tool-call-parser``, ``--reasoning-parser``) are orthogonal to
      LMCache; see the :doc:`GLM-5.1/5.2 recipe <glm5_2>` and the vLLM GLM
      recipe for those.

   .. tab-item:: SGLang

      **Status:** Not validated with LMCache.

   .. tab-item:: TRT-LLM

      **Status:** Not validated with LMCache.

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

GLM-5.3 ships a native multi-token-prediction head
(``num_nextn_predict_layers: 1``), usable via vLLM's
``--speculative-config '{"method":"mtp","num_speculative_tokens":1}'``.
**Status:** Not yet validated with LMCache.

Caveats
-------

- **Single KV cache group.** Unlike the multi-group sparse-attention models in
  this section, the validated ``zai-org/GLM-5.3`` checkpoint reports one group
  (``group_tokens_per_block=[64]``). A ``-Flash`` variant of this architecture
  additionally keeps a per-request kpool-tail scratch group
  (``KpoolTailSpec``), which LMCache skips automatically; see
  :doc:`../mp/hybrid_models`. That variant is not covered by this validation.
- Cached pages are byte-opaque views, so cache entries must not be shared
  across engines with different attention backends or kernel block sizes.
- The LMCache validation covers **text** generation with greedy decoding; the
  cold and cache-served runs reproduced each other exactly.
