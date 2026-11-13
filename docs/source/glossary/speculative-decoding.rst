.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable; they are the
   contract between this file and the published page.

   slug: speculative-decoding
   question: What is speculative decoding?
   aliases: draft model, speculative sampling
   related: decode, inter-token-latency, decode-throughput
   tier: 4
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/speculative-decoding
   schema_type: DefinedTerm
   seo_title: What is speculative decoding? | LMCache
   meta_description: Speculative decoding cheaply proposes several tokens ahead, often with a smaller *draft* model, and the main model then verifies them in a single pass.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-speculative-decoding:

Speculative decoding
====================

.. meta::
   :description: Speculative decoding cheaply proposes several tokens ahead, often with a smaller *draft* model, and the main model then verifies them in a single pass.
   :keywords: Speculative decoding, draft model, speculative sampling

**Speculative decoding cheaply proposes several tokens ahead, often with a smaller *draft* model, and the main model then verifies them in a single pass.**

Other proposal methods exist besides a draft model; vLLM, for example, also supports EAGLE, multi-token prediction (MTP), and n-gram matching. Speculative decoding reduces inter-token latency mainly when the server is under low to medium load, and its gains shrink as load rises. It works separately from KV cache reuse, and the two can run together. It is listed here because it moves the same metrics that caching does, which means a speedup in a benchmark is not always caused by caching. LMCache does not implement speculative decoding.

.. seealso::

   :doc:`decode` | :doc:`inter-token-latency` | :doc:`decode-throughput`
