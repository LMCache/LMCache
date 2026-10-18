.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: attention
   question: What is attention in a transformer?
   aliases: attention mechanism, self-attention
   related: query-key-value, attention-head, causal-mask, kv-cache
   tier: 1
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/attention
   schema_type: DefinedTerm
   seo_title: What is attention in a transformer? | LMCache
   meta_description: Attention is the mechanism that lets each token in a sequence draw information from the other tokens around it.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-attention:

Attention
=========

.. meta::
   :description: Attention is the mechanism that lets each token in a sequence draw information from the other tokens around it.
   :keywords: Attention, attention mechanism, self-attention

**Attention is the mechanism that lets each token in a sequence draw information from the other tokens around it.**

To predict what comes next, a model needs more than the latest token in isolation; it needs the context before it. In "The cat sat down because it was tired," the information associated with "cat" is what makes "it" meaningful. Attention gives the model a way to compare the current token against earlier ones and blend in whatever matters for this step.

This happens at every layer, not just for obvious references like pronouns, and different attention heads learn to emphasize different relationships — nearby syntax, long-range references, formatting patterns.

Attention is the reason a KV cache exists at all. Because each token must look back over everything before it, the model needs the earlier tokens' intermediate state available at every step — so it is worth keeping.

.. seealso::

   :doc:`query-key-value` | :doc:`attention-head` | :doc:`causal-mask` | :doc:`kv-cache`
