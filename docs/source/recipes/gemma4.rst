.. _recipe_gemma4:

Gemma 4
=======

Validated models
----------------

- `google/gemma-4-31B-it <https://huggingface.co/google/gemma-4-31B-it>`_
- `google/gemma-4-12B-it <https://huggingface.co/google/gemma-4-12B-it>`_
- `google/gemma-4-E4B-it <https://huggingface.co/google/gemma-4-E4B-it>`_

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `Gemma 4 in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#multimodal-language-models>`_
      (architectures ``Gemma4ForConditionalGeneration`` for 31B/E4B and
      ``Gemma4UnifiedForConditionalGeneration`` for 12B).

      **Status:** Validated with LMCache.

      Start the LMCache MP server:

      .. code-block:: bash

         lmcache server --l1-size-gb 100 --eviction-policy LRU

      |

      Start vLLM with the LMCache MP connector:

      .. code-block:: bash

         vllm serve google/gemma-4-31B-it \
             --tensor-parallel-size 2 \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      The smaller ``google/gemma-4-12B-it`` and ``google/gemma-4-E4B-it`` run on
      a single GPU:

      .. code-block:: bash

         vllm serve google/gemma-4-12B-it \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      Adjust ``--tensor-parallel-size`` to match your hardware. For the
      generic LMCache + vLLM wiring (ports, remote hosts),
      see :doc:`../getting_started/quickstart`.

      If there are any issues with vLLM setup, please refer to the
      `vLLM Recipes <https://docs.vllm.ai/projects/recipes/en/latest/index.html>`_
      for more details.

   .. tab-item:: SGLang

      **Status:** Not validated with LMCache.

   .. tab-item:: TRT-LLM

      **Status:** Supported. See :doc:`../getting_started/quickstart` for TRT-LLM + LMCache setup.

CacheBlend support
------------------

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

Gemma 4 supports MTP speculative decoding through separate **assistant
checkpoints** (``google/gemma-4-<size>-it-assistant``), which vLLM loads as
the draft model. The draft layers carry their own KV cache; LMCache detects
them from vLLM's ``speculative_config`` and stores/retrieves the draft-layer
KV together with the target model's -- no extra LMCache flags are required.

**Status:** Validated with LMCache (vLLM MP connector):

- ``google/gemma-4-31B-it`` + ``google/gemma-4-31B-it-assistant`` (2 GPUs)
- ``google/gemma-4-12B-it`` + ``google/gemma-4-12B-it-assistant`` (1 GPU,
  needs ``--enforce-eager``; see below)
- ``google/gemma-4-E4B-it`` + ``google/gemma-4-E4B-it-assistant`` (1 GPU)

Add to the ``vllm serve`` command shown above:

.. code-block:: bash

   --speculative-config \
       '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":1}' \
   --attention-backend TRITON_ATTN

.. warning::

   ``--attention-backend TRITON_ATTN`` is **required when MTP is enabled**.
   The target model alone auto-selects the Triton backend (FlashAttention
   does not support Gemma 4's 512-dim global-attention heads), but the
   assistant draft model's backend selection picks FlashAttention and the
   engine crashes at startup with *"FlashAttention forward only supports
   head dimension at most 256"*. Pinning the backend explicitly covers both
   models.

Validation evidence (gsm8k store-vs-retrieve, 100 samples, exact score match
required under ``VLLM_BATCH_INVARIANT=1``): 31B scored 0.78 in both the
computed and the LMCache-retrieved run with the MTP acceptance rate unchanged
(0.911 vs 0.910); 12B scored 0.28 in both runs (acceptance 0.854 vs 0.853);
E4B scored 0.69 in both runs with acceptance rate identical (0.807).
Cold-vs-warm TTFT improved 3.7x (31B) / 1.7x (12B) / 1.4x (E4B) with MTP
enabled throughout.

.. note::

   The 12B Unified assistant requires ``--enforce-eager`` on the vLLM
   nightly tested: its draft head applies a token-suppression list held in
   an unpinned CPU tensor (``gemma4_mtp.py``, ``compute_logits``), which
   aborts CUDA graph capture with *"Cannot copy between CPU and CUDA
   tensors during CUDA graph capture"*. This is an upstream vLLM issue,
   unrelated to LMCache; 31B and E4B assistants are unaffected.

Caveats
-------

- **Hybrid KV cache with heterogeneous block sizes.** Gemma 4 interleaves
  sliding-window and full-attention layers whose head dimensions differ
  (sliding 256, full 512), so vLLM unifies the physical page size by giving the
  two attention types different ``block_size``\ s (e.g. ``google/gemma-4-E4B-it``:
  sliding 32, full 16). LMCache stores and retrieves each KV cache group in its
  own block size; no extra flags are required.
- **Cross-layer KV sharing.** ``google/gemma-4-E4B-it`` reuses some layers' KV
  caches across layers. LMCache stores the cache-owning layers only; the sharing
  layers' KV lives in the same blocks and is restored automatically.
