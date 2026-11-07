.. _recipe_glm5_3_flash:

GLM-5.3-Flash
=============

A natively multimodal hybrid Mixture-of-Experts model (architecture
``Glm5NextForConditionalGeneration``, 45 layers). It interleaves 34
**Kimi Delta Attention (KDA)** linear-attention layers with 11 **NoPE sparse
MLA** layers whose top-k indexer reads from a compressed key pool. Unlike the
single-group :doc:`GLM-5.3 <glm5_3>`, the KDA layers keep a recurrent **state
cache** instead of a paged key/value cache, so GLM-5.3-Flash is served like the
other Mamba / linear-attention hybrids; see :doc:`../mp/hybrid_models` for the
general handling.

Validated models
----------------

- `zai-org/GLM-5.3-Flash <https://huggingface.co/zai-org/GLM-5.3-Flash>`_
  (native FP8 checkpoint, 4× NVIDIA H200, TP 4)

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `GLM-5.3-Flash vLLM recipe
      <https://recipes.vllm.ai/zai-org/GLM-5.3-Flash>`_
      (architecture ``Glm5NextForConditionalGeneration``).

      **Status:** Validated with LMCache on vLLM 0.33.0.

      .. note::

         **Version requirements.** vLLM needs native ``Glm5Next`` support
         (0.29 or newer); validation used **vLLM 0.33.0**. On the LMCache side,
         GLM-5.3-Flash support ships from **0.5.6**. vLLM 0.33 pins
         ``torch==2.14.0``, and the published LMCache 0.5.6 and 0.5.7rc4 wheels
         do not load their CUDA extension under that PyTorch (``undefined
         symbol`` on import of ``lmcache.cuda_ops``). With vLLM 0.33, build
         LMCache from source (see :doc:`../getting_started/installation`).
         PyTorch 2.14 headers require a fully C++20-capable host compiler,
         such as GCC 12; GCC 10 fails with ``C++20 or later compatible
         compiler is required to use PyTorch``.

      As a KDA hybrid, GLM-5.3-Flash needs prefix caching on in vLLM and
      ``--separate-object-groups`` on the LMCache server. ``--chunk-size`` does
      not need to be set: the connector negotiates the server's minimum (default
      ``256``) up to a multiple of vLLM's *unified block size* ``N`` (see
      :ref:`mamba-block-size`), which vLLM logs at startup as
      ``Setting attention block size to N tokens``:

      .. list-table::
         :header-rows: 1
         :widths: 50 25 25

         * - Model
           - Unified block size ``N``
           - GPUs (TP)
         * - ``zai-org/GLM-5.3-Flash``
           - 1152
           - 4

      ``N`` was measured at TP 4 only. Like Kimi K3's, it can change with the
      tensor-parallel size, so after changing ``--tensor-parallel-size``
      re-read ``N`` from that log line.

      Start the LMCache MP server:

      .. code-block:: bash

         lmcache server \
             --separate-object-groups \
             --l1-size-gb 100 \
             --eviction-policy LRU

      |

      Start vLLM with the LMCache MP connector (4 GPUs):

      .. code-block:: bash

         vllm serve zai-org/GLM-5.3-Flash \
             --tensor-parallel-size 4 \
             --enable-prefix-caching \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

      |

      At startup the server logs the negotiated chunk size, and the connector
      logs the geometry it resolved:

      .. code-block:: text

         Negotiated LMCache MP chunk size to 1152 (configured_min=256, required_alignment=1152)
         Resolved LMCache MP geometry: group_tokens_per_block=[1152, 1152, 1152, 1152, 1152, 0], scheduler_block_size=1152, ...

      The four leading groups are the KDA state groups. The fifth holds the
      sparse-MLA latent cache together with the compressed indexer cache
      (``slots_per_block=288``, one slot per four tokens). The trailing ``0``
      is the per-request kpool-tail scratch group (``KpoolTailSpec``), which
      vLLM marks non-prefix-cacheable and LMCache skips.

      **Why these settings** — LMCache-side (required):

      - ``--enable-prefix-caching`` is **required**: prefix caching must be on
        for LMCache to store and reuse the KDA recurrent state, and the
        connector rejects Mamba groups that keep no per-block state snapshots.
        vLLM 0.33 selects the ``align`` Mamba cache mode
        automatically when prefix caching is on, and ``--mamba-cache-mode`` is
        a deprecated no-op there. On an older vLLM that still honors the flag,
        pass ``--mamba-cache-mode align``.
      - ``--separate-object-groups`` (server) is **required** for hybrid
        Mamba / linear-attention models: it gives the KDA layers their own cache
        objects.
      - Leave ``--chunk-size`` at its default. The connector rounds the minimum
        up to ``N``; see :doc:`/mp/chunk_size_negotiation`. Only **whole**
        chunks are stored, so the last partial chunk of every prompt (up to
        1151 tokens) is recomputed, and prompts shorter than 1152 tokens get
        no external hit.

      Model-side:

      - On Hopper, keep the default BF16 KV cache: the vLLM recipe notes that
        FP8 KV cache for this model is Blackwell-only. FP8 KV cache with
        LMCache is not validated.
      - The model is multimodal. For text-only serving,
        ``--limit-mm-per-prompt '{"image":0,"video":0}'`` runs vLLM in
        text-only mode. LMCache validation covers text generation only.
      - GLM's serving extras (``--tokenizer-mode glm5_next``,
        ``--tool-call-parser glm47``, ``--reasoning-parser glm47``) are
        orthogonal to LMCache; see the vLLM recipe linked above.

      No attention-backend or ``--no-disable-hybrid-kv-cache-manager`` flag is
      needed; ``LMCacheMPConnector`` advertises hybrid support and vLLM
      auto-selects the KDA and sparse-MLA backends. For the generic LMCache +
      vLLM wiring (ports, remote hosts), see
      :doc:`../getting_started/quickstart`.

      **Validation.** With vLLM launched under ``VLLM_SERVER_DEV_MODE=1``, an
      8507-token document-QA prompt was sent cold, vLLM's local prefix cache
      was reset (``POST /reset_prefix_cache``), and the prompt was sent again,
      so the reuse came only from LMCache. Each condition ran 16 times at
      temperature 0:

      .. list-table::
         :header-rows: 1
         :widths: 40 20 40

         * - Condition
           - LMCache hit (tokens)
           - Mean log-prob of the most likely first token
         * - Cold (unique ``cache_salt`` per request)
           - 0
           - -0.93
         * - Served from LMCache after reset
           - 8064 of 8507
           - -0.94

      The two distributions are not statistically different (Welch's t-test,
      p = 0.83), and every answer in both conditions was correct. Token-level
      outputs vary between runs in *both* conditions; see Caveats.

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
   * - :doc:`CacheGen <../mp/cachegen>`
     - Not supported
     - Hybrid groups' cached pages are byte-opaque.

MTP (speculative decoding) support
----------------------------------

The checkpoint ships one multi-token-prediction layer
(``num_nextn_predict_layers: 1``), usable via vLLM's
``--speculative-config '{"method":"mtp","num_speculative_tokens":5}'``.
**Status:** Not yet validated with LMCache.

Caveats
-------

- **Not bit-exact, and batch-invariant mode is unavailable.** vLLM's output for
  this model varies between identical temperature-0 requests even with no cache
  involved. ``VLLM_BATCH_INVARIANT=1`` does not start on vLLM 0.33: no sparse-MLA
  backend supports batch invariance (``FLASHINFER_MLA_SPARSE_SM90``,
  ``FLASH_ATTN_MLA_SPARSE`` and ``FLASHMLA_SPARSE`` report ``batch invariance
  not supported``). Validate LMCache with a score- or log-prob-level comparison
  against cold runs, not a token-level diff.
- **Do not use a prefix-caching-off run as the reference.** Turning prefix
  caching off also drops the ``align`` Mamba cache mode, which changes the KDA
  numerics more than LMCache does (higher run-to-run variance in validation).
  Compare against cold runs of the same prefix-caching-on engine, for example
  with a unique ``cache_salt`` per request.
- The kpool-tail scratch state is not cached: it is recomputed on every
  request, which is expected.
- Cached pages for the KDA and sparse-MLA groups are byte-opaque views, so
  content-aware processing (CacheGen, CacheBlend) does not apply, and cache
  entries must not be shared across engines with different attention backends
  or kernel block sizes.
- vLLM's Mamba prefix caching in ``align`` mode is experimental upstream.
- Validation covered single requests on one H200 node at TP 4 with a CPU (L1)
  cache. Concurrent load, L2 backends, PD disaggregation and multimodal inputs
  are not validated.
