.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: cache-hit-rate
   question: What is cache hit rate?
   aliases: hit rate, KV cache hit rate
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
   :keywords: Cache hit rate, hit rate, KV cache hit rate

**Cache hit rate is the share of requested tokens that were served from cache rather than recomputed.** It is the mediating variable behind every other caching claim.

KV caching only helps when parts of a prompt repeat across requests — a shared system prompt, a repeated document, an earlier turn in the same conversation. Hit rate is the switch that determines whether caching does anything at all for a given request.

Higher hit rate compounds the benefit rather than merely extending it. A workload with heavy prompt overlap saves far more cumulative TTFT and throughput than a workload of all-unique one-shot prompts, where hit rate sits near zero and caching has essentially nothing to work with.

Nobody asks "what's my cache hit rate" the way they ask "how fast is my app" — but every latency and throughput claim routes through it. Hit rate is sensitive both to the workload's actual repetition pattern and to pool sizing, chunk size, and eviction policy, which means a benchmark that neither discloses nor controls it is reporting an unrepeatable number.

.. seealso::

   :doc:`cache-hit` | :doc:`prefix-caching` | :doc:`eviction-policy` | :doc:`chunk-size`
