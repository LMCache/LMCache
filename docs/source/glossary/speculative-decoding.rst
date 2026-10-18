.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
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
   meta_description: Speculative decoding uses a smaller "draft" model to propose several tokens ahead, which the main model then verifies in a single pass.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-speculative-decoding:

Speculative decoding
====================

.. meta::
   :description: Speculative decoding uses a smaller "draft" model to propose several tokens ahead, which the main model then verifies in a single pass.
   :keywords: Speculative decoding, draft model, speculative sampling

**Speculative decoding uses a smaller "draft" model to propose several tokens ahead, which the main model then verifies in a single pass.**

It can meaningfully improve inter-token latency and decode throughput, and it is entirely independent of whether any KV cache is being reused. Worth naming because it belongs in any honest list of what moves these metrics — caching is not the only lever, and LMCache does not do this.

.. seealso::

   :doc:`decode` | :doc:`inter-token-latency` | :doc:`decode-throughput`
