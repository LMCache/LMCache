.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: eviction-policy
   question: What is a cache eviction policy?
   aliases: eviction, cache_policy, LRU
   related: kv-cache, cache-hit-rate, cache-tiering, chunk-size
   tier: 2
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/eviction-policy
   schema_type: DefinedTerm
   seo_title: What is a cache eviction policy? | LMCache
   meta_description: An eviction policy decides what to discard when a cache pool is full.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-eviction-policy:

Eviction policy
===============

.. meta::
   :description: An eviction policy decides what to discard when a cache pool is full.
   :keywords: Eviction policy, eviction, cache_policy, LRU

**An eviction policy decides what to discard when a cache pool is full.** LMCache defaults to LRU — least recently used goes first.

Eviction is where a cache stops being a pure win. Once a useful cache is gone and no copy exists elsewhere, the next matching request pays full prefill again. Policy choice, pool size, and workload shape jointly determine how often that happens, which makes eviction one of the direct levers on :doc:`cache hit rate <cache-hit-rate>`.

.. seealso::

   :doc:`kv-cache` | :doc:`cache-hit-rate` | :doc:`cache-tiering` | :doc:`chunk-size`
