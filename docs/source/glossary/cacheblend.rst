.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: cacheblend
   question: What is CacheBlend?
   aliases: cache blending, enable_blending
   related: prefix-caching, cache-hit, kv-cache, ttft
   tier: 2
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/cacheblend
   schema_type: DefinedTerm
   seo_title: What is CacheBlend? | LMCache
   meta_description: CacheBlend is an LMCache technique for reusing cached KV data across chunks that are *not* in the position they were cached in — most usefully in RAG...
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-cacheblend:

CacheBlend
==========

.. meta::
   :description: CacheBlend is an LMCache technique for reusing cached KV data across chunks that are *not* in the position they were cached in — most usefully in RAG...
   :keywords: CacheBlend, cache blending, enable_blending

**CacheBlend is an LMCache technique for reusing cached KV data across chunks that are *not* in the position they were cached in** — most usefully in RAG workloads, where independently retrieved document chunks get concatenated in whatever order retrieval returned them.

The problem it addresses: in a causal transformer, every token's key and value are built from that token's representation *after attending to everything before it*. A chunk cached in isolation never attended to whatever now precedes it once it is spliced into a new prompt, so every one of that chunk's tokens carries a cached key/value reflecting the wrong preceding context. That mismatch is the approximation.

The fix: rather than recompute the whole spliced-in chunk, CacheBlend picks one designated layer, computes the true key there for every token in the combined sequence, ranks all tokens by how far their cached key deviates, and fully recomputes only the worst ~15% through the remaining layers. The other ~85% stay frozen at their cached values.

For a 32-layer model that is roughly 18% of the cost of recomputing from scratch, and the ratio improves on deeper models. It is explicitly a speed-for-accuracy trade: the bet is that the tokens judged to deviate most were the ones that mattered most.

.. note::

   **Status note:** CacheBlend is off by default (``enable_blending``) and currently supports Llama, Qwen2, and Qwen3 only. Verify against the current release before publishing.

.. seealso::

   :doc:`prefix-caching` | :doc:`cache-hit` | :doc:`kv-cache` | :doc:`ttft`
