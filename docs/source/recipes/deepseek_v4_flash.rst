.. _recipe_deepseek_v4_flash:

DeepSeek-V4-Flash
=================

Validated models
----------------

- `deepseek-ai/DeepSeek-V4-Flash <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash>`_

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `DeepSeek-V4-Flash in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#text-generation>`_
      (architecture ``DeepseekV4ForCausalLM``).

      **Status:** Validated with LMCache.

      **Installing vLLM:** DeepSeek-V4-Flash needs the sparse-MLA attention
      backends and the ``fp8_ds_mla`` KV cache kernels, so install vLLM by
      following its own recipe rather than a bare ``pip install vllm``:
      `vLLM DeepSeek-V4-Flash recipe
      <https://docs.vllm.ai/projects/recipes/en/latest/index.html>`_
      (also mirrored at https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash).

      .. warning::

         Use the **latest vLLM release**, not the ``main``/dev branch. The
         current vLLM development branch is broken for DeepSeek-V4-Flash (the
         ``fp4`` MoE experts are misdispatched and the real weights fail to
         load). Pin to the latest tagged release as the vLLM recipe instructs.

      Start the LMCache MP server:

      .. code-block:: bash

         lmcache server --l1-size-gb 100 --eviction-policy LRU

      |

      Start vLLM with the LMCache MP connector (8 GPUs):

      .. code-block:: bash

         vllm serve deepseek-ai/DeepSeek-V4-Flash \
             --tensor-parallel-size 8 \
             --enable-expert-parallel \
             --kv-cache-dtype fp8_ds_mla \
             --trust-remote-code \
             --tokenizer-mode deepseek_v4 \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      ``--kv-cache-dtype fp8_ds_mla`` and ``--tokenizer-mode deepseek_v4`` are
      required for this model; ``--enable-expert-parallel`` distributes the MoE
      experts across the tensor-parallel ranks. Adjust
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

DeepSeek-V4-Flash ships a native multi-token-prediction head
(``num_nextn_predict_layers: 1`` in its config), which vLLM uses for
speculative decoding. The MTP head has its own KV cache layer; LMCache
detects it from vLLM's ``speculative_config`` and stores/retrieves the
draft-layer KV together with the target model's -- no extra LMCache flags
are required.

**Status:** Validated with LMCache (vLLM MP connector, TP4).

Add the speculative config to the ``vllm serve`` command shown above:

.. code-block:: bash

   --speculative-config '{"method":"mtp","num_speculative_tokens":1}'

Validation evidence (gsm8k store-vs-retrieve, 100 samples): scores match
within sampling stderr (0.95 computed vs 0.96 LMCache-retrieved), and the
MTP acceptance rate is unchanged when the prefix KV is served by LMCache
(0.947 store run vs 0.952 retrieve run) -- i.e. the draft-layer KV survives
the store/retrieve round-trip intact. Cold-vs-warm TTFT improved 6.6x with
MTP enabled throughout.

MTP-specific caveats:

- Generation is **not bit-exact** between a cached and a fresh run: the
  MXFP4 MoE kernels do not support vLLM's batch-invariant mode. Expect
  score-level equivalence, not token-level.
- On the vLLM nightly tested, engine startup with CUDA graphs crashed for
  this model -- also with a vanilla ``vllm serve`` (no LMCache, no MTP), so
  this is the dev-branch breakage warned about above, not an LMCache or MTP
  interaction. ``--enforce-eager`` was used for validation; on tagged vLLM
  releases (per the vLLM recipe) it should not be needed.

Caveats
-------

- **Requires the latest vLLM release.** The vLLM dev branch is currently broken
  for this model (see the warning above) -- use a tagged release installed via
  the vLLM recipe.
- **Sparse-MLA hybrid KV cache.** DeepSeek-V4-Flash interleaves several KV
  cache groups with different block geometries (the compressed MLA latents are
  stored as ``fp8``/``uint8`` while the sparse-attention indexer groups are
  ``float32``), so the groups do not share a single block size. LMCache stores
  and retrieves each group in its own block size; no extra flags are required
  beyond ``--kv-cache-dtype fp8_ds_mla``.
