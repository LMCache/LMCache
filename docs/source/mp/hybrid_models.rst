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

What Is Not Supported Yet
-------------------------

- **Mamba / linear-attention hybrids** (e.g. Qwen3-Next, Qwen3.5, and other
  Gated-DeltaNet models). These layers keep a recurrent *state cache* (a
  convolution + SSM state) instead of a paged key/value cache, which LMCache's
  transfer path cannot represent today. Such models will fail to register with
  the LMCache server. Tracking support is future work.
- **DeepSeek-V4-style compressed / indexer caches** are likewise not yet
  handled by the multiprocess connector.

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
