.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable; they are the
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
   meta_description: A cache hit means the Key and Value tensors for a stretch of tokens are already stored and can be reused; a cache miss means they are not, so that part...
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-cache-hit:

Cache hit / cache miss
======================

.. meta::
   :description: A cache hit means the Key and Value tensors for a stretch of tokens are already stored and can be reused; a cache miss means they are not, so that part...
   :keywords: Cache hit / cache miss, cache miss, hit, miss

**A cache hit means the Key and Value tensors for a stretch of tokens are already stored and can be reused; a cache miss means they are not, so that part of prefill must run.**

Ordinary prefix caching requires an *exact* prefix match. If one earlier token differs, everything after it misses, and because matching works on whole blocks or :doc:`chunks <chunk-size>`, the chunk that contains the difference misses as well. That strictness is what :doc:`CacheBlend <cacheblend>` tries to relax.

A miss costs about what the same request would cost with no cache at all, plus the time spent on the lookup. A benchmark that does not disclose its hit rate is therefore reporting a number whose most important variable is hidden.

.. seealso::

   :doc:`cache-hit-rate` | :doc:`prefix-caching` | :doc:`chunk-size` | :doc:`cacheblend`
