Hybrid-Attention Models
=======================

Some models interleave more than one attention type across their layers — most
commonly **sliding-window attention** on most layers and **full attention** on a
few. vLLM serves these with its *hybrid KV cache manager*, which splits the
model's layers into multiple **KV cache groups** (one per attention behavior).

The LMCache multiprocess connector (``LMCacheMPConnector``) supports these
hybrid models: it stores and retrieves the KV cache for every group, so prefix
caching and KV reuse work the same way they do for plain models.

.. contents::
   :local:
   :depth: 2

What Works
----------

Models whose layers all use **standard paged attention** — including hybrids
that mix sliding-window and full attention — are supported with no special
configuration. Examples:

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Model family
     - Attention layout
     - Status
   * - Gemma 2 / Gemma 3
     - Interleaved sliding-window + full
     - Supported
   * - gpt-oss
     - Interleaved sliding-window + full
     - Supported
   * - Qwen3.5 (and other Gated-DeltaNet hybrids)
     - Interleaved Mamba/GDN + full
     - Supported (see below)
   * - Llama, Qwen2/Qwen3 (dense), Mistral, …
     - Single attention type
     - Supported

Just point vLLM at the LMCache server as usual (see :doc:`quickstart`); LMCache
detects the model's KV cache groups automatically at registration time.

.. note::

   Because ``LMCacheMPConnector`` advertises hybrid support to vLLM, vLLM keeps
   its hybrid KV cache manager **enabled** for these models (it does not fall
   back to a single unified group). You do not need
   ``--no-disable-hybrid-kv-cache-manager`` or any related flag.

Mamba / Linear-Attention Hybrids
--------------------------------

Models that interleave **Mamba / Gated-DeltaNet layers** with full attention
(e.g. ``Qwen/Qwen3.5-0.8B``) are supported. Their recurrent state caches are
reinterpreted as opaque pages at registration time, so prefix caching and KV
reuse work end to end. They need three extra flags:

#. vLLM must run with prefix caching and the ``align`` Mamba cache mode::

       vllm serve Qwen/Qwen3.5-0.8B \
           --enable-prefix-caching --mamba-cache-mode align \
           --kv-transfer-config \
           '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

#. The LMCache server's ``--chunk-size`` must be a multiple of vLLM's unified
   block size for the model (vLLM logs ``Setting attention block size to N
   tokens`` at startup; for Qwen3.5-0.8B, ``N = 544``)::

       lmcache server --chunk-size 544 --l1-size-gb 100 --eviction-policy LRU

#. ``--max-num-batched-tokens`` must be at least the unified block size and
   below twice it (LMCache raises at engine startup otherwise; setting it
   equal to the block size is the simple choice)::

       vllm serve ... --max-num-batched-tokens 544

   ``align`` mode snapshots the Mamba state only at the *end* of each
   scheduler step; a larger budget would let one step skip block boundaries,
   leaving no snapshot for LMCache to store at those prefixes.

Caveats:

- Generation is **not bit-exact** between a cached and a fresh run: GDN
  backends do not support vLLM's batch-invariant mode. Expect score-level
  equivalence, not token-level.
- The cached pages are byte-opaque, so content-aware features (CacheGen
  compression, CacheBlend) do not apply, and cache entries must not be shared
  across engines with different attention backends or kernel block sizes.

See the :doc:`Qwen3.5 recipe <../recipes/qwen3_5>` for the validated
end-to-end commands.

What Is Not Supported Yet
-------------------------

- **DeepSeek-V4-style compressed / indexer caches** are not yet handled by the
  multiprocess connector.

Verifying Correctness
---------------------

To convince yourself that a hybrid model's KV is being cached and reused
correctly, you can compare a cold run against a run served from LMCache:

#. Run an evaluation (e.g. ``lm_eval`` on ``gsm8k``) against vLLM + LMCache.
   This computes the KV cache and **stores** it in LMCache.
#. Reset *only* vLLM's local prefix cache, leaving the LMCache-managed cache
   intact (requires launching vLLM with ``VLLM_SERVER_DEV_MODE=1``)::

       curl -X POST http://localhost:8000/reset_prefix_cache

   Omit the ``reset_external=true`` query parameter so the LMCache cache is
   preserved.
#. Re-run the same evaluation. vLLM now misses in its local cache, so the prefix
   KV is **retrieved** from LMCache. The score should match the first run.

The project ships this as the ``hma_lm_eval`` continuous-integration test (see
``.buildkite/k3_tests/multiprocess``).

See Also
--------

- :doc:`quickstart` — launching the LMCache server and a vLLM client.
- Design notes on how groups are detected and addressed:
  ``docs/design/integration/vllm/hybrid-kv-cache-groups.md`` in the source tree.
