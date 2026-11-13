.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable; they are the
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

Every pass through the loop reads the Keys and Values for the entire context built so far, but computes new Keys and Values only for the one new token.
Nothing earlier is recomputed, as long as the cache still holds the earlier Keys and Values.

This loop is why each output token takes so much more time than each input token.
Prompt tokens can be processed together in one parallel pass.
In standard decoding, output tokens are produced one at a time, with one model invocation for each token.
Techniques such as speculative decoding can produce several tokens per invocation of the main model, but the underlying loop is still sequential.

.. seealso::

   :doc:`decode` | :doc:`causal-mask` | :doc:`kv-cache` | :doc:`inter-token-latency`
