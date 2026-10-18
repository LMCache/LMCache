.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: prefix-caching
   question: What is prefix caching?
   aliases: prefix cache, prefix reuse, cross-request caching
   related: kv-cache, cache-hit-rate, cache-aware-routing, cacheblend
   tier: 2
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/prefix-caching
   schema_type: DefinedTerm
   seo_title: What is prefix caching? | LMCache
   meta_description: Prefix caching is reusing a KV cache across separate requests, not just within one generation.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-prefix-caching:

Prefix caching
==============

.. meta::
   :description: Prefix caching is reusing a KV cache across separate requests, not just within one generation.
   :keywords: Prefix caching, prefix cache, prefix reuse, cross-request caching

**Prefix caching is reusing a KV cache across separate requests, not just within one generation.**

Reusing the cache inside a single generation is just the normal decode path. Reusing it between requests is a different problem. A chat interface shows one continuous conversation, but the backend usually receives each turn as a fresh request containing the whole history so far. If the serving system still holds the matching prefix — and can recognize and reach it — it can skip prefill for those tokens and process only the new suffix. If the cache was discarded, evicted, or left behind on a different engine, the entire history gets processed again.

The workloads where this matters most:

- Chat requests that repeatedly carry the same system prompt and conversation history.
- Agents making a series of calls that carry forward instructions, tool results, and intermediate state.
- RAG applications reusing the same retrieved documents across related requests.
- Many users or requests sharing an identical opening prefix.

.. seealso::

   :doc:`kv-cache` | :doc:`cache-hit-rate` | :doc:`cache-aware-routing` | :doc:`cacheblend`
