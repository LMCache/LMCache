.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: autoregressive-generation
   question: What is autoregressive generation?
   aliases: autoregressive decoding, autoregression
   related: decode, causal-mask, kv-cache, inter-token-latency
   tier: 1
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/autoregressive-generation
   schema_type: DefinedTerm
   seo_title: What is autoregressive generation? | LMCache
   meta_description: Autoregressive generation is the loop a model runs to produce text: generate one token, append it to the sequence, then use the longer sequence to...
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-autoregressive-generation:

Autoregressive generation
=========================

.. meta::
   :description: Autoregressive generation is the loop a model runs to produce text: generate one token, append it to the sequence, then use the longer sequence to...
   :keywords: Autoregressive generation, autoregressive decoding, autoregression

**Autoregressive generation is the loop a model runs to produce text: generate one token, append it to the sequence, then use the longer sequence to generate the next.**

Every pass through the loop reads the entire context built so far, but computes and appends only one new Key and Value. Nothing earlier is redone — provided the cache is there.

This loop is why output length costs so much more than input length. Prompt tokens can be processed together in one parallel pass; output tokens are strictly sequential, one model invocation each.

.. seealso::

   :doc:`decode` | :doc:`causal-mask` | :doc:`kv-cache` | :doc:`inter-token-latency`
