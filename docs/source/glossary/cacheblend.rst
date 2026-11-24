.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable; they are the
   contract between this file and the published page.

   slug: cacheblend
   question: What is CacheBlend?
   aliases: cache blending
   related: prefix-caching, cache-hit, kv-cache, ttft
   tier: 2
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/cacheblend
   schema_type: DefinedTerm
   seo_title: What is CacheBlend? | LMCache
   meta_description: CacheBlend is a technique for reusing the cached KV data of a text chunk even when the chunk is not at the start of the prompt — most usefully in RAG...
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-cacheblend:

CacheBlend
==========

.. meta::
   :description: CacheBlend is a technique for reusing the cached KV data of a text chunk even when the chunk is not at the start of the prompt — most usefully in RAG...
   :keywords: CacheBlend, cache blending

**CacheBlend is a technique for reusing the cached KV data of a text chunk even when the chunk is not at the start of the prompt** — most usefully in RAG workloads, where independently retrieved document chunks get concatenated in whatever order retrieval returned them.
It was introduced in the CacheBlend paper (Yao et al., EuroSys 2025).

The problem it addresses: in a causal transformer, every token's Key and Value are built from that token's representation *after attending to everything before it*.
A chunk cached on its own never attended to the text that precedes it in the new prompt, so its cached Keys and Values lack that cross-attention and differ from what a full prefill would produce.
The chunk also sits at a new position, so its cached Keys must be re-rotated to that position (RoPE).
Reusing the cached data without further work therefore gives only an approximate result.

Instead of recomputing the whole chunk, CacheBlend runs every token through the layers up to an early check layer.
At that layer it compares each token's freshly computed Key with its cached Key and selects the tokens whose Keys deviate most.
Only those tokens, a small fraction of the reused tokens, are recomputed through the remaining layers; the other tokens keep their cached Keys and Values.
In the paper, the selected set narrows gradually over several layers rather than at a single check layer.

With an early check layer on a 32-layer model, this is roughly 18–20 percent of the layer computation of a full prefill, and the fraction falls as models get deeper.
Loading the cached data from storage is a separate cost, which the paper overlaps with the recomputation.
CacheBlend trades a small amount of accuracy for speed, on the assumption that the tokens whose Keys deviate most at the check layer are the ones whose errors matter most.
In the paper's evaluation, recomputing 5–18 percent of tokens reduced F1 or Rouge-L by at most 0.002.

.. seealso::

   :doc:`prefix-caching` | :doc:`cache-hit` | :doc:`kv-cache` | :doc:`ttft`
