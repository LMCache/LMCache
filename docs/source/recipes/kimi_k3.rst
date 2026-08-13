.. _recipe_kimi_k3:

Kimi K3
=======

Moonshot AI's flagship hybrid Mixture-of-Experts model. Like
:doc:`Kimi-Linear <kimi_linear>`, it interleaves **Kimi Delta Attention (KDA)**
linear-attention layers with **Multi-head Latent Attention (MLA)**
full-attention layers. The KDA layers keep a recurrent **state cache** (a
convolution + delta-net state) instead of a paged key/value cache; LMCache
reinterprets that state as an opaque page at registration time, so prefix
caching and KV reuse work end to end. See :doc:`../mp/hybrid_models` for the
general handling of Mamba / linear-attention models.

Validated models
----------------

- `moonshotai/Kimi-K3 <https://huggingface.co/moonshotai/Kimi-K3>`_
  (8× NVIDIA B300)
- `moonshotai/Kimi-K3 <https://huggingface.co/moonshotai/Kimi-K3>`_ MXFP4
  (8× AMD MI355X, via the InferenceX AgentX benchmark; fp8 KV cache,
  DSpark speculative decoding)

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Engine documentation:**
      `Kimi K3 in vLLM supported models
      <https://docs.vllm.ai/en/latest/models/supported_models.html#text-generation>`_.

      **Status:** Validated with LMCache. Verified on NVIDIA B300 (8 GPUs).

      .. note::

         **Version requirements.** Kimi K3 support has not landed in a stable
         vLLM release yet — use the upstream pre-release Docker image for now
         (a stable vLLM release with K3 support is expected to follow, likely
         0.26.1 or 0.27):

         .. code-block:: bash

            docker pull vllm/vllm-openai:kimi-k3

         On the LMCache side, use a **nightly build from 2026-07-27 or newer**
         (see :doc:`../getting_started/installation`); stable LMCache releases
         include K3 support starting from **0.5.3**.

      As a Mamba / linear-attention hybrid, Kimi K3 needs the same three
      settings as the other KDA / GDN hybrids: the ``align`` Mamba cache mode,
      prefix caching, and a chunk size that is a multiple of **every** engine
      KV group's ``tokens_per_block`` (see :ref:`mamba-block-size` for how
      the unified block size ``N`` is determined). For Kimi K3 at
      ``--tensor-parallel-size 8`` the validated values are:

      .. list-table::
         :header-rows: 1
         :widths: 34 22 22 22

         * - Platform (TP 8)
           - Attention block ``N``
           - KDA state group
           - Minimum ``--chunk-size``
         * - NVIDIA B300 (bf16 KV)
           - 768
           - 768
           - 768
         * - AMD MI355X (fp8 KV, ``TRITON_MLA``)
           - 1536
           - 3072
           - 3072

      ``N`` depends on the tensor-parallel size, the KV cache dtype, and the
      attention backend (the KDA state is sharded across TP ranks while the
      MLA cache is not, and vLLM sizes the attention block so the attention
      page is at least as large as the mamba page). **Do not copy a number
      from this table for a different stack.** After changing
      ``--tensor-parallel-size``, ``--kv-cache-dtype``, or the platform,
      re-read ``N`` from vLLM's ``Setting attention block size to N tokens``
      startup log line and re-derive ``--chunk-size`` from it. The KDA state
      group can register a *larger* ``tokens_per_block`` than ``N`` (the
      MI355X row above registers 3072 against ``N = 1536``); LMCache rejects
      a chunk size that is not a multiple of every group's block size at
      engine startup, and the error message names the offending group and
      value — take the final chunk size from that, not from ``N`` alone.

      Start the LMCache MP server (``--chunk-size`` = ``N`` = 768):

      .. code-block:: bash

         lmcache server \
             --port 6555 \
             --chunk-size 768 \
             --separate-object-groups \
             --max-workers 4 \
             --l1-size-gb 100 \
             --eviction-policy LRU

      |

      Start vLLM with the LMCache MP connector (8 GPUs):

      .. code-block:: bash

         export VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1

         vllm serve moonshotai/Kimi-K3 \
             --trust-remote-code \
             --load-format dummy \
             --moe-backend auto \
             --gpu-memory-utilization 0.95 \
             --tensor-parallel-size 8 \
             --no-enable-flashinfer-autotune \
             --enable-auto-tool-choice \
             --tool-call-parser kimi_k3 \
             --reasoning-parser kimi_k3 \
             --enable-prefix-caching \
             --mamba-cache-mode align \
             --kv-transfer-config \
             '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.port":6555}}'

      |

      **Why these settings** — LMCache-side (required):

      - ``--mamba-cache-mode align`` and ``--enable-prefix-caching`` are
        **required**. ``align`` is the only Mamba cache mode the KDA backend
        supports, and prefix caching must be on for LMCache to store and reuse
        the recurrent state.
      - ``--separate-object-groups`` (server) is **required** for hybrid
        Mamba / linear-attention models: it gives the KDA layers their own
        cache objects.
      - ``--chunk-size`` (server) must be a multiple of **every** engine KV
        group's ``tokens_per_block`` (see the table above) — on stacks where
        all groups share the unified block size, ``--chunk-size N`` is the
        simplest choice. LMCache raises at engine startup if it is not.
      - ``--max-num-batched-tokens`` no longer needs to be set explicitly.
        The constraint is only that it be **at least** ``N`` (``align``
        snapshots the KDA state on block boundaries at the end of a
        scheduler step), which vLLM's default already satisfies. Earlier
        revisions of this recipe pinned ``1500`` as a workaround from before
        ``--separate-object-groups`` allowed values ``≥ 2N``; that pin is
        obsolete. Smaller values within ``[N, 2N)`` snapshot every block
        boundary (finest reuse) at the cost of prefill throughput. See
        :doc:`../mp/hybrid_models` for the full rationale.
      - The server's ``--port 6555`` must match ``lmcache.mp.port`` in the
        connector config.

      Model-side (Kimi K3 serving requirements):

      - ``VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1`` enables the fused
        latent-MoE tail path for Kimi K3.
      - ``--tool-call-parser kimi_k3``, ``--reasoning-parser kimi_k3``, and
        ``--enable-auto-tool-choice`` wire up K3's tool-calling and reasoning
        output formats.
      - ``--trust-remote-code`` loads Kimi K3's custom modeling code;
        ``--moe-backend auto`` lets vLLM pick the MoE kernel backend;
        ``--no-enable-flashinfer-autotune`` skips FlashInfer autotuning at
        startup.
      - ``--load-format dummy`` initializes random weights so the serving
        stack can be validated without downloading the full checkpoint —
        **drop it to serve the real weights**.
      - ``--tensor-parallel-size 8`` shards the weights across eight GPUs.
        Adjust it to your hardware — but note it changes ``N`` and the two
        derived flags (see above).

      No attention-backend or ``--no-disable-hybrid-kv-cache-manager`` flag is
      needed; ``LMCacheMPConnector`` advertises hybrid support and vLLM
      auto-selects the KDA and MLA backends. For the generic LMCache + vLLM
      wiring (ports, remote hosts), see :doc:`../getting_started/quickstart`.

      If there are any issues with vLLM setup, please refer to the
      `vLLM Recipes <https://docs.vllm.ai/projects/recipes/en/latest/index.html>`_
      for more details.

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

Operational notes from sustained-load validation
------------------------------------------------

Learned from running this recipe under the InferenceX AgentX trace-replay
benchmark on 8× MI355X (100k–330k-token contexts, ~1 hour of sustained load,
DRAM offload as the external tier):

- **Size the L1 against** ``/dev/shm``, **not free DRAM.** The MP server's L1
  pool is backed by POSIX shared memory in every transfer mode, so the usable
  ceiling is the ``/dev/shm`` tmpfs mount (Linux default: 50% of RAM), not
  total host memory. Pre-flight the budget: on the engine-driven path an
  oversized L1 silently falls back to the (much slower) pickle transport; on
  the ``lmcache_driven`` path the server-side capacity check does not run, so
  writes can hit ``SIGBUS`` when the tmpfs fills.
- **Pin** ``--supported-transfer-mode lmcache_driven`` **for benchmarks.**
  The default ``auto`` advertises both transfer paths; pinning one makes the
  measured data path deterministic across runs.
- ``--enable-extra-logging`` (server) logs cumulative per-device store and
  retrieve token counters, which pair with vLLM's
  ``vllm:external_prefix_cache_queries`` / ``hits`` metrics to attribute hit
  rates: external hits are counted only on tokens the local GPU prefix cache
  missed, so the two hit populations are disjoint and the external hit rate
  reads as "share of locally-missed tokens served by LMCache".

Caveats
-------

- **Pre-release engine support.** Until vLLM ships K3 in a stable release, pin
  both sides: the ``vllm/vllm-openai:kimi-k3`` Docker image on the vLLM side
  and an LMCache nightly from 2026-07-27 or newer (stable from 0.5.3).
- **Concurrent load on hybrid models requires the sliding-window
  lock-release fix.** Versions up to and including 0.5.4rc1 over-release
  read locks when freeing the vLLM-hit prefix: for linear-attention (KDA)
  object groups, the lookup only retains locks on the trailing window, and
  the extra releases decrement locks held by concurrent requests on the same
  content-addressed chunks. Under L1 eviction pressure this lets evicted
  state pages be reclaimed mid-retrieve — observed as ``finish read on
  non-read-locked key`` warnings followed by corrupted generations or GPU
  ``hipErrorIllegalAddress`` / ``HSA_STATUS_ERROR_EXCEPTION`` faults. Use a
  release containing the fix for any multi-request workload.
- Generation is **not guaranteed bit-exact** between a cached and a fresh run
  under concurrent load: KDA / GDN linear-attention backends do not support
  vLLM's batch-invariant mode, so kernel results can vary with batch
  composition. Validate with a score-level comparison, not a token-level diff.
- Cached pages for the KDA and MLA groups are byte-opaque views, so
  content-aware processing (CacheGen, CacheBlend) does not apply, and cache
  entries must not be shared across engines with different attention backends
  or kernel block sizes.
- vLLM's Mamba prefix caching in ``align`` mode is experimental upstream.
