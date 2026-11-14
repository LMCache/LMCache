.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable; they are the
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
   meta_description: Chunk size is the granularity at which cached KV data is stored and looked up — 256 tokens by default in LMCache (the server's --chunk-size setting).
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-chunk-size:

Chunk size
==========

.. meta::
   :description: Chunk size is the granularity at which cached KV data is stored and looked up — 256 tokens by default in LMCache (the server's --chunk-size setting).
   :keywords: Chunk size, chunk_size, cache granularity

**Chunk size is the granularity at which cached KV data is stored and looked up** — 256 tokens by default in LMCache (the server's ``--chunk-size`` setting).
The setting is a minimum: when vLLM connects, the connector can raise it so that chunk boundaries line up with the model's KV cache layout.

It is a tradeoff rather than a value to maximize.
Smaller chunks match more of a partially shared prefix, which raises hit rate on workloads whose shared text does not end on a chunk boundary, but they add per-chunk lookup and metadata overhead.
Larger chunks are cheaper to manage and transfer, but a token that differs anywhere inside a chunk makes the whole chunk unusable, so a larger chunk loses more of the matching tokens that come before the difference.

.. seealso::

   :doc:`cache-hit-rate` | :doc:`kv-cache` | :doc:`eviction-policy` | :doc:`cacheblend`
