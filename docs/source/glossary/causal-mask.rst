.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: causal-mask
   question: What is a causal mask?
   aliases: causal masking, causal attention
   related: attention, prefill, autoregressive-generation, kv-cache
   tier: 1
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/causal-mask
   schema_type: DefinedTerm
   seo_title: What is a causal mask? | LMCache
   meta_description: A causal mask blocks a token from attending to positions that come after it, enforcing that a model generating text left to right can use itself and...
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-causal-mask:

Causal mask
===========

.. meta::
   :description: A causal mask blocks a token from attending to positions that come after it, enforcing that a model generating text left to right can use itself and...
   :keywords: Causal mask, causal masking, causal attention

**A causal mask blocks a token from attending to positions that come after it**, enforcing that a model generating text left to right can use itself and everything before it, but nothing ahead.

This one-way flow is what makes caching possible. Because a later token cannot change what an earlier token saw, extending a prefix with a new token does not invalidate the Keys and Values already computed for the earlier positions. The new token attends to them; the model never has to build them again.

The mask is also what lets prefill process an entire prompt in one parallel pass without any position peeking ahead.

.. seealso::

   :doc:`attention` | :doc:`prefill` | :doc:`autoregressive-generation` | :doc:`kv-cache`
