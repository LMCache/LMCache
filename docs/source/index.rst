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
* `Join LMCache slack workspace <https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ>`_
* Our papers:

  * `CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving <https://dl.acm.org/doi/10.1145/3651890.3672274>`_
  * `CacheBlend: Fast Large Language Model Serving with Cached Knowledge Fusion <https://arxiv.org/abs/2405.16444>`_
  * `Do Large Language Models Need a Content Delivery Network? <https://arxiv.org/abs/2409.13761>`_

:raw-html:`<br />`


Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart/index
   getting_started/troubleshoot
   getting_started/faq

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2
   :caption: KV Cache offloading and sharing

   kv_cache/storage_backends/index
   kv_cache/caching_policies

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2
   :caption: Disaggregated prefill

   disaggregated_prefill/nixl/index
   disaggregated_prefill/shared_storage

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2
   :caption: KV Cache management

   kv_cache_management/controller
   kv_cache_management/lookup
   kv_cache_management/persist
   kv_cache_management/clear
   kv_cache_management/move
   kv_cache_management/compress
   kv_cache_management/check_finish

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2
   :caption: KV Cache Optimizations

   kv_cache_optimizations/compression/index
   kv_cache_optimizations/blending

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2
   :caption: Use LMCache in production

   production/docker_deployment
   production/kubernetes_deployment

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer_guide/contributing
   developer_guide/docker_file
   developer_guide/usage/index

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference/configurations
   api_reference/storage_backends
   api_reference/dynamic_connector
   api_reference/multimodality
   
:raw-html:`<br />`

.. toctree::
   :maxdepth: 2
   :caption: Community

   community/meetings
   community/blogs

raw-html:`<br />`
   
