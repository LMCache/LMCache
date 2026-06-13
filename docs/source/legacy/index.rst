.. _legacy:

Legacy (In-Process Mode)
========================

.. warning::

   The pages in this section document **in-process mode**, where LMCache runs
   *inside* the inference engine process (e.g. via ``LMCacheConnectorV1`` on
   vLLM). This mode is **deprecated** in favor of :doc:`Multiprocess (MP) mode
   <../mp/index>`, which is the recommended way to run LMCache.

   These pages are preserved because several features below are **not yet
   available in MP mode**. As MP gains parity, the corresponding documentation
   will graduate out of this section. If you are starting fresh, use
   :doc:`../mp/index`.

Features still exclusive to in-process mode
-------------------------------------------

The following are documented here because they have **no MP equivalent yet**:

- **P2P KV cache sharing** -- :doc:`../kv_cache/p2p_sharing`
- **Async loading** -- :doc:`../kv_cache/async_loading`
- **Encoder cache / multimodality** -- :doc:`../non_kv_cache/encoder_cache`,
  :doc:`../api_reference/multimodality`
- **Disaggregated prefill** -- :doc:`../disaggregated_prefill/nixl/index`
- **CacheGen compression** -- :doc:`../kv_cache_optimizations/compression/index`
- **Cache management** (``move`` / ``pin`` / ``compress``) --
  :doc:`../kv_cache_management/index`
- **Advanced eviction policies** (MRU / LFU / FIFO) --
  :doc:`../kv_cache/caching_policies`

Features available in both modes
--------------------------------

The pages below describe the in-process configuration. For MP, the equivalent
lives under the :doc:`../mp/index` section:

- **Storage backends** -- in MP these are **L2 adapters**; see
  :doc:`../mp/l2_storage`.
- **Eviction (LRU / noop)** -- configured on the ``lmcache server``; see
  :doc:`../mp/configuration`.
