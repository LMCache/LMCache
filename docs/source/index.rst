.. LMCache documentation master file, created by
   sphinx-quickstart on Mon Sep 30 10:39:18 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. role:: raw-html(raw)
    :format: html

Welcome to LMCache!
=====================

.. figure:: https://people.cs.uchicago.edu/~junchenj/docs/lmcache-logo.png
  :width: 60%
  :align: center
  :alt: LMCache
  :class: no-scaled-link

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
   It thus reduces prefill delay, i.e., time to first token (TTFT), as well as saves the precious GPU cycles.

   By combining LMCache with vLLM, LMCaches achieves 3-10x delay savings and GPU cycle reduction in many LLM use cases, including multi-round QA and RAG.
   </p>

:raw-html:`<br />`

What's next?
=====================

Follow these links to get started with LMCache:

* :ref:`speedup` 
* :ref:`installation`
* :ref:`docker`

:raw-html:`<br />`

Documentation
=====================

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/speedup
   getting_started/installation   
   getting_started/docker

.. toctree::
   :maxdepth: 1
   :caption: Detailed Examples

   examples/launching
   examples/save_decode
   examples/kv_blending
   examples/backend
   examples/measuring_improvements

.. toctree::
   :maxdepth: 1
   :caption: Models

   models/models

.. toctree::
   :maxdepth: 1
   :caption: Configuration

   configuration/config

.. toctree::
   :maxdepth: 1
   :caption: Developer Documentation

   developer_tutorial/overview
   developer_tutorial/LLM_Engine
   developer_tutorial/LMCache_Engine
   developer_tutorial/LMCache_Backend

.. toctree::
   :maxdepth: 1
   :caption: Advanced

   advanced/lmcache.blend
   advanced/lmcache.server
   advanced/lmcache.storage_backend
