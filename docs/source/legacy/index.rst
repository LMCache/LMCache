.. _legacy:

Legacy (In-Process Mode)
========================

.. warning::

   These pages document LMCache's original **in-process mode**, where LMCache
   ran *inside* the inference engine process (e.g. via ``LMCacheConnectorV1`` on
   vLLM). In-process mode is **deprecated**; new deployments should use
   :doc:`Multiprocess (MP) mode <../mp/index>`.

Background
----------

LMCache began as an in-process library embedded directly in the serving engine.
The **multiprocess refactor** moved LMCache into a standalone ``lmcache server``
and made an **asynchronous prefetching architecture** the default -- a
``LOOKUP`` followed by background L2→L1 loads -- alongside process isolation,
shared caching across engine instances, and multi-tier (L1/L2) storage.

MP is now the recommended mode and is on track to support essentially
everything in-process mode did. The pages below are kept for users still on
in-process mode and as a historical reference while MP closes any remaining
gaps; where a feature already has an MP equivalent, prefer the MP docs.

.. toctree::
   :maxdepth: 2

   getting_started/quickstart/index
   kv_cache/index
   disaggregated_prefill/nixl/index
   kv_cache_optimizations/index
   kv_cache_management/index
   api_reference/index
   internal_api_server/index
   controller/index
   non_kv_cache/index
   production/index
   developer_guide/index
