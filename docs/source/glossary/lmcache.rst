.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: lmcache
   question: What is LMCache?
   aliases: LM Cache
   related: kv-cache, cache-tiering, prefix-caching, pd-disaggregation
   tier: 2
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/lmcache
   schema_type: DefinedTerm
   seo_title: What is LMCache? | LMCache
   meta_description: LMCache is an open-source KV-cache layer for LLM serving.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-lmcache:

LMCache
=======

.. meta::
   :description: LMCache is an open-source KV-cache layer for LLM serving.
   :keywords: LMCache, LM Cache

**LMCache is an open-source KV-cache layer for LLM serving.** It treats the KV cache as reusable data that can live beyond a single GPU-memory allocation and, when useful, beyond a single request or inference-engine process.

It moves KV data through a storage hierarchy spanning GPU memory, CPU memory, local storage, and remote backends, so that when a later request reuses cached context the serving system can retrieve the corresponding state rather than rebuilding it through prefill. That can cut repeated computation, improve time to first token, and leave GPU memory free for active work.

LMCache integrates with the inference engine rather than replacing it — most commonly vLLM, through connector hooks in the engine's own scheduler.

---

.. seealso::

   :doc:`kv-cache` | :doc:`cache-tiering` | :doc:`prefix-caching` | :doc:`pd-disaggregation`
