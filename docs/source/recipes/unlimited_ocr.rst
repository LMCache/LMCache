.. _recipe_unlimited_ocr:

Unlimited-OCR (multimodal R-SWA)
================================

Validated models
----------------

- `baidu/Unlimited-OCR <https://huggingface.co/baidu/Unlimited-OCR>`_

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `Official vLLM Unlimited-OCR recipe
      <https://recipes.vllm.ai/baidu/Unlimited-OCR>`_
      (architecture ``UnlimitedOCRForCausalLM``).

      **Status:** Validated with LMCache on one NVIDIA A100 (single-image
      request, MP mode) at vLLM commit ``4fc943b8676e``.

      Unlimited-OCR and R-SWA may require a recent vLLM nightly or source
      installation. If the installed release does not recognize the
      architecture, use the validated commit above or a later compatible
      revision.

      Start the LMCache MP server. The chunk size is shown explicitly because
      R-SWA caches only complete prompt chunks:

      .. code-block:: bash

         lmcache server --chunk-size 256 --l1-size-gb 20 \
             --eviction-policy LRU

      |

      Start vLLM with the LMCache-provided MP connector:

      .. code-block:: bash

         vllm serve baidu/Unlimited-OCR \
             --trust-remote-code \
             --logits-processors \
                 vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor \
             --no-enable-prefix-caching \
             --mm-processor-cache-gb 4 \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector", "kv_role":"kv_both"}'

      |

      ``--mm-processor-cache-gb`` must remain positive. Do **not** copy the
      ``0`` value from the baseline vLLM recipe: with both prefix caching and
      the multimodal processor cache disabled, vLLM can assign renderer-local
      image identifiers that repeat after a frontend restart or across
      replicas. The R-SWA MP connector rejects this unsafe configuration at
      startup rather than risk a stale cross-image cache hit.

      Follow the `official vLLM recipe
      <https://recipes.vllm.ai/baidu/Unlimited-OCR>`_ for the required
      ``<image>`` prompt prefix, ``skip_special_tokens=False``, and per-request
      n-gram processor arguments. Those model-side request requirements do not
      change when LMCache is enabled.

      The validated 277-token prompt stored one 256-token chunk on the cold
      request. Repeating the request looked up and retrieved those 256 tokens,
      with no second store. The remaining 21 prompt tokens were recomputed.
      See :ref:`rswa-prompt-caching` for the policy and its safety rationale.

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

Caveats
-------

- **MP mode only.** R-SWA is supported only through the LMCache-provided
  external MP connector. The in-process connector rejects this model; see
  :ref:`rswa-prompt-caching`.
- **Prompt KV only.** Decode KV is never cached. Resumable requests bypass
  LMCache, and prompts shorter than one LMCache chunk have nothing to store.
- **Single-image validation.** The end-to-end validation above covered one
  image. Multi-image requests have not been validated with LMCache.
- **KV cache only.** Vision-encoder outputs are a separate cache: see
  :doc:`../non_kv_cache/encoder_cache` (in-process mode only; not available for
  this MP setup).
