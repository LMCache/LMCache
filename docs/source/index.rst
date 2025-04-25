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
   <strong> Redis for LLMs. </strong>
   </p>


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

:raw-html:`<br />`

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

   kv_cache/cpu_ram
   kv_cache/local_storage
   kv_cache/redis
   kv_cache/infinistore
   kv_cache/mooncake
   kv_cache/valkey

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

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2
   :caption: KV Cache Optimizations

   kv_cache_optimizations/compression
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
   developer_guide/dockerfile

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference/configurations
   api_reference/storage_backends

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2
   :caption: Community

   community/meetings
   community/blogs
