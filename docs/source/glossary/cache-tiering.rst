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
   meta_description: Cache tiering is moving KV data out of GPU memory into progressively cheaper, slower storage, so that it stays reusable instead of being discarded.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-cache-tiering:

Cache tiering and offloading
============================

.. meta::
   :description: Cache tiering is moving KV data out of GPU memory into progressively cheaper, slower storage, so that it stays reusable instead of being discarded.
   :keywords: Cache tiering and offloading, KV offloading, cache tiers, tiered storage, L1 L2 L3 cache

**Cache tiering is moving KV data out of GPU memory into progressively cheaper, slower storage, so that it stays reusable instead of being discarded.**

In LMCache's multiprocess mode, there are two tiers below GPU memory. L1 is CPU memory by default, and its size (``--l1-size-gb``) has no default, so it must be set. L2 is one or more secondary storage backends, such as a local file system or an S3-compatible object store, configured with ``--l2-adapter``. LMCache copies KV data from L1 to L2 in the background and loads it back into L1 when a request needs it.

The bet is economic — cheap storage substituting for expensive GPU memory and compute. It is worth it only when retrieval is faster than recomputation, which depends on how much context is reused, where the cache landed, transfer bandwidth, and whether the lookup hits at all.

The systems questions are therefore what to keep, where to keep it, when to move it, and how to make it reachable by the engine that needs it.

.. seealso::

   :doc:`kv-cache` | :doc:`interconnects` | :doc:`ttft` | :doc:`total-cost-of-ownership`
