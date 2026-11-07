.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: cache-tiering
   question: What is KV cache offloading or tiering?
   aliases: KV offloading, cache tiers, tiered storage, L1 L2 L3 cache
   related: kv-cache, interconnects, ttft, total-cost-of-ownership
   tier: 2
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/cache-tiering
   schema_type: DefinedTerm
   seo_title: What is KV cache offloading or tiering? | LMCache
   meta_description: Cache tiering is moving KV data out of GPU memory into progressively cheaper, slower storage, so it stays reusable instead of being thrown away.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-cache-tiering:

Cache tiering and offloading
============================

.. meta::
   :description: Cache tiering is moving KV data out of GPU memory into progressively cheaper, slower storage, so it stays reusable instead of being thrown away.
   :keywords: Cache tiering and offloading, KV offloading, cache tiers, tiered storage, L1 L2 L3 cache

**Cache tiering is moving KV data out of GPU memory into progressively cheaper, slower storage, so it stays reusable instead of being thrown away.**

The usual hierarchy: GPU memory, then CPU RAM (L1 in LMCache's configuration, ``max_local_cpu_size``, 5 GB by default), then local disk (L2, ``max_local_disk_size``), then a remote store (L3, ``remote_url``).

The bet is economic — cheap storage substituting for expensive GPU memory and compute. It only pays off when retrieval is faster than recomputation, which depends on how much context is reused, where the cache landed, transfer bandwidth, and whether the lookup hits at all. Retrieving a cache is not free and is not always faster than rebuilding it.

That is the real systems question, and it is not "should we cache." It is what to keep, where to keep it, when to move it, and how to make it reachable by the engine that needs it.

.. seealso::

   :doc:`kv-cache` | :doc:`interconnects` | :doc:`ttft` | :doc:`total-cost-of-ownership`
