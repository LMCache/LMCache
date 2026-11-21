.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable; they are the
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
   meta_description: CacheBlend is a technique, implemented in LMCache, for reusing the cached KV data of a text chunk even when the chunk is not at the start of the prompt...
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-cacheblend:

CacheBlend
==========

.. meta::
   :description: CacheBlend is a technique, implemented in LMCache, for reusing the cached KV data of a text chunk even when the chunk is not at the start of the prompt...
   :keywords: CacheBlend, cache blending, enable_blending

**CacheBlend is a technique, implemented in LMCache, for reusing the cached KV data of a text chunk even when the chunk is not at the start of the prompt** — most usefully in RAG workloads, where independently retrieved document chunks get concatenated in whatever order retrieval returned them.
It was introduced in the CacheBlend paper (Yao et al., EuroSys 2025).

The problem it addresses: in a causal transformer, every token's Key and Value are built from that token's representation *after attending to everything before it*.
A chunk cached on its own never attended to the text that precedes it in the new prompt, so its cached Keys and Values lack that cross-attention and differ from what a full prefill would produce.
The chunk also sits at a new position, so its cached Keys must be re-rotated to that position (RoPE).
Reusing the cached data without further work therefore gives only an approximate result.

Instead of recomputing the whole chunk, CacheBlend runs every token through the layers up to an early check layer.
At that layer it compares each token's freshly computed Key with its cached Key and selects the tokens whose Keys deviate most.
Only those tokens, about 15 percent of the reused tokens by default, are recomputed through the remaining layers; the other tokens keep their cached Keys and Values.
LMCache exposes this as a single check layer (``cb.check_layer``, default 1) and a recompute ratio (``cb.recomp_ratio``, default 0.15).
The paper narrows the selected set gradually over several layers instead.

With an early check layer on a 32-layer model, this is roughly 18–20 percent of the layer computation of a full prefill, and the fraction falls as models get deeper.
Loading the cached data from storage is a separate cost, which the paper overlaps with the recomputation.
CacheBlend trades a small amount of accuracy for speed, on the assumption that the tokens whose Keys deviate most at the check layer are the ones whose errors matter most.
In the paper's evaluation, recomputing 5–18 percent of tokens reduced F1 or Rouge-L by at most 0.002.

.. note::

   **Status note:** In LMCache's multiprocess server, CacheBlend is off by default; you enable it with ``lmcache server --engine-type blend``, and it requires a chunk size of 256.
   The older in-process implementation (``enable_blending``) supported only Llama, Qwen2, and Qwen3 model architectures.
   Verify current model support against the release before publishing.

.. seealso::

   :doc:`prefix-caching` | :doc:`cache-hit` | :doc:`kv-cache` | :doc:`ttft`
