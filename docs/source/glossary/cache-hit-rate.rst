.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable; they are the
   contract between this file and the published page.

   slug: cache-hit-rate
   question: What is cache hit rate?
   aliases: hit rate, KV cache hit rate, prefix cache hit rate
   related: cache-hit, prefix-caching, eviction-policy, chunk-size
   tier: 3
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/cache-hit-rate
   schema_type: DefinedTerm
   seo_title: What is cache hit rate? | LMCache
   meta_description: Cache hit rate is the share of requested tokens that were served from cache rather than recomputed.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-cache-hit-rate:

Cache hit rate
==============

.. meta::
   :description: Cache hit rate is the share of requested tokens that were served from cache rather than recomputed.
   :keywords: Cache hit rate, hit rate, KV cache hit rate, prefix cache hit rate

**Cache hit rate is the share of requested tokens that were served from cache rather than recomputed.**
Every latency or throughput gain claimed for a KV cache depends on it.

KV caching only helps when parts of a prompt repeat across requests — a shared system prompt, a repeated document, an earlier turn in the same conversation.
If none of a request's tokens are in the cache, caching does not reduce that request's prefill work at all.

Engines that reuse only a matching prefix report this number as the *prefix cache hit rate*.
vLLM, for example, exposes the counters ``vllm:prefix_cache_hits`` and ``vllm:prefix_cache_queries``, and the hit rate is their ratio.
LMCache's token-level hit rate also counts only the matching prefix, in units of whole chunks.
CacheBlend is different, because it can also reuse cached chunks that are not part of a prefix; LMCache therefore reports blend hits with their own metrics.

The saved prefill work grows with the hit rate.
A workload with heavy prompt overlap saves far more total TTFT and compute than a workload of unique one-time prompts, where the hit rate is near zero.

Hit rate depends on the workload's actual repetition pattern and also on cache pool size, chunk size, and eviction policy.
A benchmark that neither discloses nor controls the hit rate therefore reports a number that nobody can reproduce.

.. seealso::

   :doc:`cache-hit` | :doc:`prefix-caching` | :doc:`eviction-policy` | :doc:`chunk-size`
