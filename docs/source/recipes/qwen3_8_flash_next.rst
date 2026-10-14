.. _recipe_qwen3_8_flash_next:

Qwen3.8-Flash-Next
==================

A hybrid architecture interleaving **Gated-DeltaNet (GDN) linear-attention**
layers with **Qwen Sparse Attention (QSA)** full-attention layers
(architecture ``Qwen4ExpForConditionalGeneration``, one full-attention layer
every four). QSA picks its history from compressed keys, so while a compression
group is still open its raw keys sit in a small per-request scratch ring that
LMCache skips; see :doc:`../mp/hybrid_models` for the general handling of
Mamba / linear-attention models.

.. note::

   Despite the name, this model does **not** share the
   ``Qwen3_5ForConditionalGeneration`` architecture used by the
   :doc:`Qwen3.5 / Qwen3.6 / Qwen3.8 series <qwen3_5>`; it has its own
   architecture and its own unified block size.

Validated models
----------------

- `Qwen/Qwen3.8-Flash-Next <https://huggingface.co/Qwen/Qwen3.8-Flash-Next>`_ (8 GPUs)

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `Qwen3.8-Flash-Next in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#text-generation>`_
      (architecture ``Qwen4ExpForConditionalGeneration``).

      **Status:** Validated with LMCache.

      Like the other GDN hybrids, this model needs three matched settings: the
      ``align`` Mamba cache mode, prefix caching, and a chunk size equal to
      vLLM's *unified block size*. vLLM logs that block size at startup as
      ``Setting attention block size to N tokens``:

      .. list-table::
         :header-rows: 1
         :widths: 50 25 25

         * - Model
           - Unified block size ``N``
           - GPUs
         * - ``Qwen/Qwen3.8-Flash-Next``
           - 400
           - 8

      Start the LMCache MP server (``--chunk-size`` = ``N``):

      .. code-block:: bash

         lmcache server \
             --chunk-size 400 \
             --separate-object-groups \
             --l1-size-gb 100 \
             --eviction-policy LRU

      |

      Start vLLM with the LMCache MP connector (8 GPUs, ``N = 400`` →
      ``2N-1 = 799``):

      .. code-block:: bash

         vllm serve Qwen/Qwen3.8-Flash-Next \
             --tensor-parallel-size 8 \
             --enable-prefix-caching \
             --mamba-cache-mode align \
             --max-num-batched-tokens 799 \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      ``--mamba-cache-mode align`` is required (GDN does not support the ``all``
      mode). ``--separate-object-groups`` (server) is required for hybrid models
      so the linear-attention layers get their own cache objects.
      ``--max-num-batched-tokens`` must be **at least** ``N``; prefer ``2N-1``,
      whose spare ``N-1`` budget lets decodes co-schedule with a prefill block —
      setting it to exactly ``N`` serializes execution to one request at a time.
      Values ``>= 2N`` raise prefill throughput but snapshot only the last block
      of each step. See the :doc:`Qwen3.5 recipe <qwen3_5>` for the full
      discussion of this trade-off, which applies unchanged here.

      Adjust ``--tensor-parallel-size`` to match your hardware. For the generic
      LMCache + vLLM wiring (ports, remote hosts), see
      :doc:`../getting_started/quickstart`.

   .. tab-item:: SGLang

      **Status:** Not validated with LMCache.

   .. tab-item:: TRT-LLM

      **Status:** Not validated with LMCache.

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

MTP (speculative decoding) support
----------------------------------

The checkpoint ships multi-token-prediction layers (``mtp``,
``mtp_num_hidden_layers``). **Status:** Not yet validated with LMCache.

Caveats
-------

- **QSA scratch state is not cached.** The per-request QSA compressor ring
  (``CircularBufferSpec``) is marked non-prefix-cacheable by vLLM and is
  skipped by LMCache end to end, so it is recomputed rather than restored. It
  shows up as a ``0`` entry in the connector's logged
  ``group_tokens_per_block``, which is expected.
- Cached pages for the linear-attention and full-attention groups are
  byte-opaque views, so content-aware processing does not apply, and cache
  entries must not be shared across engines with different attention backends
  or kernel block sizes.
- vLLM's Mamba prefix caching in ``align`` mode is experimental.
- Generation is not guaranteed bit-exact between a cached and a fresh run,
  since GDN backends do not support vLLM's batch-invariant mode. (The
  validation run for this recipe did reproduce the cold output exactly, but
  treat score-level equivalence as the contract.)
- ``Qwen/Qwen3.8-Flash-Next`` is a vision-language checkpoint; the LMCache
  validation covers **text** generation. Caching of image/video KV is not
  validated.
