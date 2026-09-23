.. _recipe_deepseek_v41_flash:

DeepSeek-V4.1-Flash
===================

A sparse-MLA Mixture-of-Experts model with a **sliding-window cache on every
layer** plus a compressed KV tier on a few *source* layers, so its layers split
into several KV cache groups with different block geometries. The
``LMCacheMPConnector`` stores and retrieves each group in its own block size.
Unlike :doc:`DeepSeek-V4-Flash <deepseek_v4_flash>`, the V4.1 compressors keep a
small per-request scratch ring that LMCache skips automatically; see
:doc:`../mp/hybrid_models`.

Validated models
----------------

- `deepseek-ai/DeepSeek-V4.1-Flash <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash>`_ (8 GPUs)

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `DeepSeek-V4.1-Flash in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#text-generation>`_
      (architecture ``DeepseekV41ForCausalLM``).

      **Status:** Validated with LMCache.

      .. warning::

         DeepSeek-V4.1 support is **not in any tagged vLLM release yet**. It
         landed on vLLM ``main`` after ``v0.29.0`` was cut, so a
         ``pip install vllm`` does not have it. Use a nightly wheel from
         https://wheels.vllm.ai or build from a revision that contains
         ``vllm/models/deepseek_v41/``.

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

         vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
             --tensor-parallel-size 8 \
             --enable-expert-parallel \
             --kv-cache-dtype fp8_ds_mla \
             --enable-prefix-caching \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      ``--kv-cache-dtype fp8_ds_mla`` is required: the sparse-MLA decode kernels
      read a packed ``uint8`` KV record, and an unspecific ``auto`` / ``fp8``
      resolves to it anyway. ``--tokenizer-mode deepseek_v41`` is selected
      automatically from the checkpoint and only needs to be passed explicitly
      if you override the tokenizer.

      **Chunk size.** The server's ``--chunk-size`` must be a multiple of every
      non-scratch group's ``tokens_per_block``. DeepSeek-V4.1-Flash uses ``32``
      for its sliding-window groups and ``64`` for the compressed-KV / indexer
      group, so any multiple of ``64`` works; ``256`` is a good default.
      ``--separate-object-groups`` keeps the per-group registration explicit,
      which is what this multi-geometry model needs.

      ``--enable-expert-parallel`` distributes the MoE experts across the
      tensor-parallel ranks, as on :doc:`DeepSeek-V4-Flash <deepseek_v4_flash>`;
      it was validated here and leaves the KV cache geometry unchanged. Adjust
      ``--tensor-parallel-size`` to match your hardware. For the generic
      LMCache + vLLM wiring (ports, remote hosts), see
      :doc:`../getting_started/quickstart`.

   .. tab-item:: SGLang

      **Status:** Not validated with LMCache.

   .. tab-item:: TRT-LLM

      **Status:** Not validated with LMCache.

CacheBlend support
------------------

Not supported: the sparse-MLA groups' cached pages are byte-opaque packed
records.

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
     - Packed ``fp8_ds_mla`` records are byte-opaque.

MTP (speculative decoding) support
----------------------------------

DeepSeek-V4.1-Flash ships multi-token-prediction layers
(``num_nextn_predict_layers: 3``). **Status:** Not yet validated with LMCache.

Caveats
-------

- **The KV record format depends on the GPU architecture, not just the model.**
  On datacenter Blackwell (SM100) the sparse-decode kernels use DeepSeek's V4.1
  record -- all 512 dims as fp8 with one UE8M0 scale per 32 dims, 528 bytes per
  token on 512-byte pages. Every other architecture (including Hopper/SM90)
  keeps the V4 record: 584 bytes per token on 576-byte pages. A cache entry
  written on one architecture therefore **must not** be shared with an engine
  running the other, the same way entries must not be shared across different
  attention backends or kernel block sizes.
- ``nvfp4_ds_mla`` (an NVFP4 compressed cache) requires the SM100 records and
  is rejected elsewhere.
- **Multiple KV cache groups.** Every layer owns a sliding-window cache
  (window 128) and the *kv-source* layers additionally own a compressed cache
  paired with a sparse indexer cache; ratio-2 source layers pack two tokens
  into one stored state. LMCache registers each distinct geometry as its own
  group; no extra flags are needed beyond those above.
- **Compressor scratch state is not cached.** The per-request compressor ring
  (``CircularBufferSpec``) is marked non-prefix-cacheable by vLLM and is
  skipped by LMCache end to end, so it is recomputed rather than restored.
- ``deepseek-ai/DeepSeek-V4.1-Flash`` is a vision-language checkpoint; the
  LMCache validation covers **text** generation. Caching of image KV is not
  validated.
