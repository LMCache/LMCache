.. _recipe_qwen3_5:

Qwen3_5ForConditionalGeneration
===============================

A hybrid architecture interleaving Mamba / Gated-DeltaNet (GDN) linear-attention
layers with full-attention layers. LMCache reinterprets the recurrent state
caches as opaque pages at registration time; see :doc:`../mp/hybrid_models`.

Validated models
----------------

- `Qwen/Qwen3.5-0.8B <https://huggingface.co/Qwen/Qwen3.5-0.8B>`_

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `Qwen3.5 in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#text-generation>`_
      (architecture ``Qwen3_5ForConditionalGeneration``).

      **Status:** Validated with LMCache.

      Start the LMCache MP server. ``--chunk-size`` must be a multiple of
      vLLM's unified block size for the model — vLLM logs ``Setting attention
      block size to N tokens`` at startup; for Qwen3.5-0.8B, ``N = 544``:

      .. code-block:: bash

         lmcache server --chunk-size 544 --l1-size-gb 100 --eviction-policy LRU

      |

      **Qwen3.5-0.8B** (1 GPU):

      .. code-block:: bash

         vllm serve Qwen/Qwen3.5-0.8B \
             --enable-prefix-caching \
             --mamba-cache-mode align \
             --max-num-batched-tokens 544 \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      ``--mamba-cache-mode align`` is required (GDN does not support the
      ``all`` mode). ``--max-num-batched-tokens`` must be at least the unified
      block size and below twice it — LMCache raises at engine startup
      otherwise. ``align`` snapshots the Mamba state only at scheduler-step
      ends, so each prefill step must advance exactly one block for every
      block boundary to hold a reusable snapshot.

      For the generic LMCache + vLLM wiring (ports, remote hosts, in-process
      mode), see :doc:`../mp/quickstart`.

   .. tab-item:: SGLang

      **Status:** Not validated with LMCache.

   .. tab-item:: TRT-LLM

      **Status:** Not supported. LMCache TRT-LLM integration is in progress.

CacheBlend support
------------------

Not supported: the hybrid groups' cached pages are byte-opaque (see Caveats).

Compression support
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Method
     - Status
     - Notes
   * - :doc:`CacheGen <../kv_cache_optimizations/compression/cachegen>`
     - Not supported
     - Hybrid groups' cached pages are byte-opaque.

Caveats
-------

- Generation is **not bit-exact** between a cached and a fresh run: GDN
  backends do not support vLLM's batch-invariant mode. Expect score-level
  equivalence, not token-level (the CI gate is the ``hma_lm_eval_qwen3_5``
  gsm8k store-vs-retrieve comparison).
- Cached pages for the Mamba and full-attention groups are byte-opaque views,
  so content-aware processing does not apply, and cache entries must not be
  shared across engines with different attention backends or kernel block
  sizes.
- vLLM's Mamba prefix caching in ``align`` mode is experimental.
