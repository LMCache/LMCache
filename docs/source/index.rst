.. LMCache documentation master file, created by
   sphinx-quickstart on Mon Sep 30 10:39:18 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

LMCache
=====================

LMCache lets LLMs prefill each text only once. By storing the KV caches of all reusable texts, LMCache can reuse the KV caches of any reused text (not necessarily prefix) in any serving engine instance. 
It thus reduces prefill delay, i.e., time to first token (TTFT), as well as saves the precious GPU cycles.

By combining LMCache with vLLM, LMCaches achieves 3-10x delay savings and GPU cycle reduction in many LLM use cases, including multi-round QA and RAG.


.. toctree::
   :maxdepth: 1
   :caption: Documentation

   getting_started/installation
   getting_started/quickstart
   getting_started/sharing_kv_cache
   developer/lmcache
   
