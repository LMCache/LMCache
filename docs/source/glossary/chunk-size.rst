.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: chunk-size
   question: What is chunk size in LMCache?
   aliases: chunk_size, cache granularity
   related: cache-hit-rate, kv-cache, eviction-policy, cacheblend
   tier: 2
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/chunk-size
   schema_type: DefinedTerm
   seo_title: What is chunk size in LMCache? | LMCache
   meta_description: Chunk size is the granularity at which cached KV data is stored and looked up — 256 tokens by default in LMCache (chunk_size).
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-chunk-size:

Chunk size
==========

.. meta::
   :description: Chunk size is the granularity at which cached KV data is stored and looked up — 256 tokens by default in LMCache (chunk_size).
   :keywords: Chunk size, chunk_size, cache granularity

**Chunk size is the granularity at which cached KV data is stored and looked up** — 256 tokens by default in LMCache (``chunk_size``).

It is a genuine tradeoff rather than a value to maximize. Smaller chunks match more partial prefixes, raising hit rate on workloads whose overlap does not fall on tidy boundaries, but add per-chunk lookup and metadata overhead. Larger chunks are cheaper to manage and transfer, but a single differing token near a boundary wastes more of the chunk.

.. seealso::

   :doc:`cache-hit-rate` | :doc:`kv-cache` | :doc:`eviction-policy` | :doc:`cacheblend`
