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

.. rst-class:: hero-tagline

**A KV Cache Management Layer for Scalable LLM Inference.**

.. note::
   We are currently in the process of upgrading our documentation to provide better guidance and examples. Some sections may be under construction. Thank you for your patience!

.. raw:: html

   <p style="text-align:center">
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/LMCache/LMCache" data-show-count="true" data-size="large" aria-label="Star">Star</a>
   <a class="github-button" href="https://github.com/LMCache/LMCache/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
   <a class="github-button" href="https://github.com/LMCache/LMCache/fork" data-show-count="true" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
   </p>

.. container:: hero-description

   LMCache is a **KV cache management layer** for LLM inference. It turns KV cache from a temporary state into reusable *AI-native knowledge* that can be *stored* persistently, *reused* across multiple serving engines, *monitored* with an observability stack, and *transformed* for better generation quality. As a result, LMCache **reduces TTFT** (time-to-first-token) and **improves throughput**, especially for long-context agentic, multi-turn conversation, and knowledge-augmented workloads (e.g., RAG).

   LMCache is **vendor-neutral**. It can be used as a KV cache layer for a range of mainstream open-source serving engines, inference frameworks, hardware vendors, storage systems, and infrastructure providers. The vendor neutrality allows users to freely switch between serving engines and storage vendors, while reusing the stored KV caches.


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

   getting_started/index

:raw-html:`<br />`

.. toctree::
   :maxdepth: 3

   interacting_with_server

:raw-html:`<br />`

.. toctree::
   :maxdepth: 3

   recipes/index

:raw-html:`<br />`

.. toctree::
   :maxdepth: 4

   mp/l2_storage/index

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2

   distributed_kv_cache

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2

   production/index

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2

   mp/observability/index

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2

   community/index

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2

   kv_cache_optimizations/index

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2

   developer_guide/index

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2

   non_kv_cache/index

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2

   legacy/index

:raw-html:`<br />`