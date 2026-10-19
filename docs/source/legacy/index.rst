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
   :hidden:
   :maxdepth: 1

   /legacy/getting_started/quickstart/index
   /legacy/kv_cache/storage_backends/index
   /legacy/kv_cache/async_loading
   /legacy/kv_cache/caching_policies
   /legacy/kv_cache/p2p_sharing
   /legacy/non_kv_cache/hidden_states
   /legacy/non_kv_cache/encoder_cache
   /legacy/disaggregated_prefill/nixl/index
   /legacy/kv_cache_optimizations/compression/index
   /legacy/kv_cache_optimizations/layerwise
   /legacy/kv_cache_management/index
   /legacy/kv_cache_optimizations/blending
   /legacy/api_reference/multimodality
   /legacy/api_reference/dynamic_connector
   /legacy/api_reference/configurations
   /legacy/internal_api_server/internal_api_server
   /legacy/controller/index
   /legacy/production/observability/index
   /legacy/production/kubernetes_deployment
   /legacy/production/docker_deployment
   /legacy/production/performance_tuning
   /legacy/production/kv_cache_events
   /legacy/developer_guide/architecture
   /legacy/developer_guide/musa_aiter_integration
   /legacy/developer_guide/integration
   /legacy/developer_guide/usage/basic_check
   /legacy/developer_guide/extending_lmcache/storage_plugins
   /legacy/developer_guide/extending_lmcache/remote_storage_plugins
