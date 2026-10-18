.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: cache-hit
   question: What is a cache hit or cache miss?
   aliases: cache miss, hit, miss
   related: cache-hit-rate, prefix-caching, chunk-size, cacheblend
   tier: 2
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/cache-hit
   schema_type: DefinedTerm
   seo_title: What is a cache hit or cache miss? | LMCache
   meta_description: A cache hit means the key/value tensors for a stretch of tokens were already computed and can be reused; a cache miss means there is nothing to reuse...
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-cache-hit:

Cache hit / cache miss
======================

.. meta::
   :description: A cache hit means the key/value tensors for a stretch of tokens were already computed and can be reused; a cache miss means there is nothing to reuse...
   :keywords: Cache hit / cache miss, cache miss, hit, miss

**A cache hit means the key/value tensors for a stretch of tokens were already computed and can be reused; a cache miss means there is nothing to reuse and the model has to run that portion of prefill from scratch**, exactly as if caching were not there at all.

Ordinary prefix caching requires an *exact* prefix match. If one earlier token differs, everything after it misses. That strictness is what :doc:`CacheBlend <cacheblend>` tries to relax.

A miss is not a penalty — it is simply the uncached baseline. But a benchmark that does not disclose its hit rate is reporting a number whose most important variable is hidden.

.. seealso::

   :doc:`cache-hit-rate` | :doc:`prefix-caching` | :doc:`chunk-size` | :doc:`cacheblend`
