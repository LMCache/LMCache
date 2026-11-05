.. _recipe_qwen3_5:

Qwen3.5 / Qwen3.6 / Qwen3.8 series
==================================

A hybrid architecture interleaving Mamba / Gated-DeltaNet (GDN) linear-attention
layers with full-attention layers, shared by the **Qwen3.5, Qwen3.6 and
Qwen3.8** series (all use the ``Qwen3_5ForConditionalGeneration``
architecture). LMCache reinterprets the recurrent state caches as opaque pages
at registration time; see :doc:`../mp/hybrid_models` for the general handling
of Mamba / linear-attention models.

Validated models
----------------

- `Qwen/Qwen3.8-27B <https://huggingface.co/Qwen/Qwen3.8-27B>`_ (1 GPU)
- `Qwen/Qwen3.6-27B <https://huggingface.co/Qwen/Qwen3.6-27B>`_ (1 GPU)
- `Qwen/Qwen3.5-0.8B <https://huggingface.co/Qwen/Qwen3.5-0.8B>`_ (1 GPU)

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `Qwen3.5 in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#text-generation>`_
      (architecture ``Qwen3_5ForConditionalGeneration``).

      **Status:** Validated with LMCache.

      Every model in this family needs the same settings: the ``align`` Mamba
      cache mode, prefix caching, and ``--separate-object-groups`` on the
      LMCache server. The chunk size is negotiated from vLLM's *unified block
      size*, which is model-specific — vLLM logs
      ``Setting attention block size to N tokens`` at startup:

      .. list-table::
         :header-rows: 1
         :widths: 50 25 25

         * - Model
           - Unified block size ``N``
           - GPUs
         * - ``Qwen/Qwen3.8-27B``
           - 784
           - 1
         * - ``Qwen/Qwen3.6-27B``
           - 784
           - 1
         * - ``Qwen/Qwen3.5-0.8B``
           - 544
           - 1

      **Qwen3.6-27B** (1 GPU, ``N = 784``):

      .. code-block:: bash

         lmcache server --chunk-size 784 --separate-object-groups \
             --l1-size-gb 100 --eviction-policy LRU

      .. code-block:: bash

         vllm serve Qwen/Qwen3.6-27B \
             --enable-prefix-caching \
             --mamba-cache-mode align \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      **Qwen3.8-27B** (1 GPU, ``N = 784``): same block size as
      Qwen3.6-27B, so the commands above apply unchanged apart from the model
      id.

      |

      **Qwen3.5-0.8B** (1 GPU, ``N = 544``): identical to the above, with
      ``--chunk-size 544``. A lower configured chunk-size minimum, including
      the default 256, negotiates to 544; see
      :doc:`/mp/chunk_size_negotiation`.

      ``--mamba-cache-mode align`` is required (GDN does not support the
      ``all`` mode). ``--separate-object-groups`` (server) is required for
      hybrid models so the Mamba layers get their own cache objects. If vLLM
      reports *"max_num_seqs exceeds available Mamba cache blocks"*, lower
      ``--max-num-seqs`` to ≤ that count (each decode sequence needs one Mamba
      block) or raise ``--gpu-memory-utilization``.

      For the generic LMCache + vLLM wiring (ports, remote hosts), see
      :doc:`../getting_started/quickstart`.

   .. tab-item:: SGLang

      **Status:** Not validated with LMCache.

   .. tab-item:: TRT-LLM

      **Status:** Supported. See :doc:`../getting_started/quickstart` for TRT-LLM + LMCache setup.

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
- ``Qwen/Qwen3.6-27B`` and ``Qwen/Qwen3.8-27B`` are vision-language models
  (they load a vision tower); the LMCache validation covers **text**
  generation (the ``hma_lm_eval_qwen3_5`` gsm8k store-vs-retrieve gate runs on
  Qwen3.5 / Qwen3.6). Caching of image/video KV is not validated.
