.. LMCache documentation master file, created by
   sphinx-quickstart on Mon Sep 30 10:39:18 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. role:: raw-html(raw)
    :format: html

Welcome to LMCache!
=====================

.. figure:: ./assets/lmcache-logo_crop.png
  :width: 60%
  :align: center
  :alt: LMCache
  :class: no-scaled-link

.. raw:: html

   <p style="text-align:center; font-size:24px;">
   <strong> Supercharge Your LLM with the Fastest KV Cache Layer. </strong>
   </p>

.. note::
   We are currently in the process of upgrading our documentation to provide better guidance and examples. Some sections may be under construction. Thank you for your patience!

.. raw:: html

   <p style="text-align:center">
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/LMCache/LMCache" data-show-count="true" data-size="large" aria-label="Star">Star</a>
   <a class="github-button" href="https://github.com/LMCache/LMCache/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
   <a class="github-button" href="https://github.com/LMCache/LMCache/fork" data-show-count="true" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
   </p>

.. raw:: html

   <p style="text-align:justify">
   LMCache lets LLMs prefill each text only once. By storing the KV caches of all reusable texts, LMCache can reuse the KV caches of any reused text (not necessarily prefix) in any serving engine instance. 
   It thus reduces prefill delay, i.e., time to first token (TTFT), as well as saves the precious GPU cycles and memory.

   By combining LMCache with vLLM, LMCaches achieves 3-10x delay savings and GPU cycle reduction in many LLM use cases, including multi-round QA and RAG.
   </p>


For more information, check out the following:

* `LMCache blogs <https://lmcache.github.io>`_
* `Join LMCache slack workspace <https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-3zxjao8h0-lRfBfnLqbALOtLsWn2ITxA>`_
* Our papers:

  * `CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving <https://dl.acm.org/doi/10.1145/3651890.3672274>`_
  * `CacheBlend: Fast Large Language Model Serving with Cached Knowledge Fusion <https://arxiv.org/abs/2405.16444>`_
  * `Do Large Language Models Need a Content Delivery Network? <https://arxiv.org/abs/2409.13761>`_

:raw-html:`<br />`


Documentation
-------------


.. toctree::
   :maxdepth: 2
   :caption: Welcome to LMCache

   self

.. toctree::
   :maxdepth: 1
   :caption: Contributing

   developer_guide/contributing

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart
   getting_started/benchmarking
   getting_started/kv_cache_calculator

:raw-html:`<br />`

.. toctree::
   :maxdepth: 1
   :caption: Multiprocess Mode

   mp/index
   mp/quickstart
   mp/configuration
   mp/hybrid_models
   mp/l2_storage
   mp/serde
   mp/deployment
   mp/operator
   mp/coordinator
   mp/http_api
   mp/observability
   mp/tracing_and_debugging
   mp/architecture
   mp/frontend_dashboard
   mp/disaggregated_prefill
   mp/kv_cache_management
   mp/p2p

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2
   :caption: Recipes

   recipes/index

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2
   :caption: KV Cache Optimizations

   kv_cache_optimizations/blending

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2
   :caption: Use LMCache in production

   production/docker_deployment
   production/kubernetes_deployment
   production/kv_cache_events
   production/observability/index
   production/performance_tuning

:raw-html:`<br />`

.. toctree::
   :maxdepth: 1
   :caption: CLI

   cli/index
   cli/server
   cli/coordinator
   cli/describe
   cli/ping
   cli/query
   cli/bench
   cli/kvcache
   cli/quota
   cli/trace
   cli/tool

:raw-html:`<br />`

.. toctree::
   :caption: Developer Guide

   developer_guide/docker_file
   developer_guide/architecture
   developer_guide/integration
   developer_guide/extending_lmcache/index
   developer_guide/cli
   developer_guide/usage/index

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference/configurations

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2
   :caption: Community

   community/meetings
   community/blogs

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2
   :caption: Legacy (In-Process Mode)

   legacy/index
   getting_started/quickstart/index
   kv_cache/storage_backends/index
   kv_cache/async_loading
   kv_cache/caching_policies
   kv_cache/p2p_sharing
   non_kv_cache/encoder_cache
   disaggregated_prefill/nixl/index
   disaggregated_prefill/shared_storage
   kv_cache_optimizations/compression/index
   kv_cache_optimizations/layerwise
   kv_cache_management/index
   api_reference/multimodality
   api_reference/storage_backends
   api_reference/dynamic_connector
   internal_api_server/internal_api_server
   controller/index

:raw-html:`<br />`