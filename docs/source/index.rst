.. LMCache documentation master file, created by
   sphinx-quickstart on Mon Sep 30 10:39:18 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. role:: raw-html(raw)
    :format: html

Welcome to LMCache!
=====================

.. figure:: https://github.com/user-attachments/assets/a0809748-3cb1-4732-9c5a-acfa90cc72d1
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

Documentation
=====================


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart
   getting_started/docker

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/launching
   examples/backend

.. toctree::
   :maxdepth: 1
   :caption: Configuration

   configuration/config

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   developer/lmcache
