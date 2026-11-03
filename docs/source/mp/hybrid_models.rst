Hybrid Attention Models
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

Validated hybrid models
-----------------------

Recipe pages for the validated hybrid-attention architectures:

.. list-table::
   :header-rows: 1
   :widths: 34 34 32

   * - Model
     - Attention layout
     - Recipe
   * - Gemma 3
     - Sliding-window + full
     - :doc:`/recipes/gemma3`
   * - Gemma 4
     - Sliding-window + full
     - :doc:`/recipes/gemma4`
   * - gpt-oss
     - Sliding-window + full
     - :doc:`/recipes/gpt_oss`
   * - Qwen3.5 / Qwen3.6 / Qwen3.8 series
     - Mamba / GDN + full
     - :doc:`/recipes/qwen3_5`
   * - Qwen3.8-Flash-Next
     - GDN linear-attention + QSA sparse full (scratch ring)
     - :doc:`/recipes/qwen3_8_flash_next`
   * - Kimi-Linear
     - KDA linear-attention + MLA full
     - :doc:`/recipes/kimi_linear`
   * - Kimi K3
     - KDA linear-attention + MLA full
     - :doc:`/recipes/kimi_k3`
   * - DeepSeek-V4-Flash
     - Sparse-MLA (multiple KV groups)
     - :doc:`/recipes/deepseek_v4_flash`
   * - DeepSeek-V4.1-Flash
     - Sparse-MLA + per-layer sliding window (9 KV groups, scratch ring)
     - :doc:`/recipes/deepseek_v41_flash`
   * - GLM 5.1/5.2
     - Dynamic Sparse Attention (multiple KV groups)
     - :doc:`/recipes/glm5_2`
   * - GLM-5.3
     - Dynamic Sparse Attention (single KV group)
     - :doc:`/recipes/glm5_3`
   * - MiniMax-M3
     - Sparse attention + lightning indexer (mixed KV formats in one group)
     - :doc:`/recipes/minimax_m3`

.. toctree::
   :hidden:
   :maxdepth: 1

   /recipes/gemma3
   /recipes/gemma4
   /recipes/gpt_oss
   /recipes/qwen3_5
   /recipes/qwen3_8_flash_next
   /recipes/kimi_linear
   /recipes/kimi_k3
   /recipes/deepseek_v4_flash
   /recipes/deepseek_v41_flash
   /recipes/glm5_2
   /recipes/glm5_3
   /recipes/minimax_m3

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

Just point vLLM at the LMCache server as usual (see :doc:`/getting_started/quickstart`); LMCache
detects the model's KV cache groups automatically at registration time.

Per-request scratch groups are excluded from caching, including on older vLLM
forks that mark them with ``participates_in_prefix_caching=False`` instead of
``prefix_cacheable=False``.

.. note::

   Because ``LMCacheMPConnector`` advertises hybrid support to vLLM, vLLM keeps
   its hybrid KV cache manager **enabled** for these models (it does not fall
   back to a single unified group). You do not need
   ``--no-disable-hybrid-kv-cache-manager`` or any related flag.

Object-group separation
-----------------------

At KV-cache registration LMCache can bucket a hybrid model's layers into
**object groups** — the unit it stores and retrieves as one object. With
``--separate-object-groups`` each distinct cross-chunk attention window becomes
its own object group: full-attention layers form one group, and each
sliding-window size (mamba / GDN included) forms another. The default is
**off** — every layer shares a single full-attention object group.

.. code-block:: bash

   # default: a single full-attention object group for all layers
   lmcache server --chunk-size 256 --l1-size-gb 100

   # one object group per attention window (required for mamba / GDN hybrids)
   lmcache server --chunk-size 256 --l1-size-gb 100 --separate-object-groups

For a non-hybrid model (a single attention behavior) the setting makes no
difference — every layer resolves to one object group either way. For a
**Mamba / linear-attention hybrid it is required** (see below): it gives the
linear-attention layers their own cache objects so their recurrent state is
stored and loaded independently of the full-attention layers.

Mamba / Linear-Attention Hybrids
--------------------------------

Models that interleave **Mamba / Gated-DeltaNet (GDN) linear-attention layers**
with full attention — the Qwen3.5, Qwen3.6 and Qwen3.8 series
(``Qwen/Qwen3.5-0.8B``, ``Qwen/Qwen3.6-27B``, ``Qwen/Qwen3.8-27B``, …),
Qwen3-Next, Kimi-Linear
(``moonshotai/Kimi-Linear-48B-A3B-Instruct``), Kimi K3
(``moonshotai/Kimi-K3``), and other GDN hybrids — are supported.
Unlike a paged key/value cache, their linear-attention layers keep a recurrent
**state cache** (a convolution + SSM state). LMCache reinterprets that state as
an opaque page at registration time, so prefix caching and KV reuse work end to
end without any model-specific transfer code.

This section is the **general procedure for any such model**; the settings
are identical across models.

.. _mamba-block-size:

Unified block size and chunk size
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a Mamba / GDN hybrid, vLLM forces **one** block size ``N`` across all KV
cache groups, chosen large enough that an attention page is at least as big as
a Mamba state page. It depends on the model's head dimensions and GDN state
size (and, when the recurrent state is sharded, on the tensor-parallel size),
so it is model-specific. vLLM prints it once at startup::

    INFO ... interface.py:670] Setting attention block size to 784 tokens to
    ensure that attention page size is >= mamba page size.

Nothing has to be derived from ``N`` by hand: the LMCache server's
``--chunk-size`` is a minimum, and the connector negotiates the smallest
multiple of ``N`` that is at least that value (``N = 784`` resolves the default
minimum of 256 to 784, while ``N = 544`` resolves it to 544). See
:doc:`chunk_size_negotiation` for the full lifecycle and mixed-model
constraints. For reference, values grow with model size:

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Model
     - Unified block size ``N``
     - GPUs
   * - ``Qwen/Qwen3.6-27B``
     - 784
     - 1
   * - ``Qwen/Qwen3.8-27B``
     - 784
     - 1
   * - ``Qwen/Qwen3.5-0.8B``
     - 544
     - 1
   * - ``moonshotai/Kimi-Linear-48B-A3B-Instruct``
     - 944
     - 2
   * - ``moonshotai/Kimi-K3``
     - 768
     - 8

Required flags
^^^^^^^^^^^^^^

#. **LMCache server** ``--separate-object-groups`` — gives the linear-attention
   layers their own cache objects so their recurrent state is stored and loaded
   independently of the full-attention layers. ``--chunk-size`` can stay at its
   default; the connector negotiates it up to a multiple of ``N``::

       lmcache server --chunk-size 256 --separate-object-groups \
           --l1-size-gb 100 --eviction-policy LRU

#. **vLLM** ``--mamba-cache-mode align --enable-prefix-caching`` — ``align`` is
   mandatory (GDN backends do not support the ``all`` mode)::

       vllm serve <model> \
           --enable-prefix-caching --mamba-cache-mode align \
           --kv-transfer-config \
           '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

No ``--no-disable-hybrid-kv-cache-manager`` or attention-backend flag is needed;
``LMCacheMPConnector`` advertises hybrid support and vLLM auto-selects the GDN
backend.

Caveats
^^^^^^^

- Generation is **not bit-exact** between a cached and a fresh run: GDN
  backends do not support vLLM's batch-invariant mode. Validate with a
  **score-level** comparison (see `Verifying Correctness`_), not a token-level
  diff.
- The cached pages are **byte-opaque**, so content-aware features (CacheGen
  compression, CacheBlend) do not apply, and cache entries must not be shared
  across engines with different attention backends or kernel block sizes.
- Several of these models are **vision-language** (they load a vision tower).
  The validated, supported path is **text** KV caching; image/video KV caching
  is not validated.
- vLLM's Mamba prefix caching in ``align`` mode is marked experimental upstream.

See the :doc:`Qwen3.5 / Qwen3.6 / Qwen3.8 recipe <../recipes/qwen3_5>` for the
validated end-to-end commands and the per-model block sizes.

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

- :doc:`/getting_started/quickstart` — launching the LMCache server and a vLLM client.
- Design notes on how groups are detected and addressed:
  ``docs/design/integration/vllm/hybrid-kv-cache-groups.md`` in the source tree.
