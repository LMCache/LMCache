.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: attention-head
   question: What is an attention head?
   aliases: attention heads, multi-head attention, MHA
   related: attention, query-key-value, grouped-query-attention
   tier: 1
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/attention-head
   schema_type: DefinedTerm
   seo_title: What is an attention head? | LMCache
   meta_description: An attention head is one independent copy of the attention computation, running on its own slice of each token's representation.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-attention-head:

Attention head
==============

.. meta::
   :description: An attention head is one independent copy of the attention computation, running on its own slice of each token's representation.
   :keywords: Attention head, attention heads, multi-head attention, MHA

**An attention head is one independent copy of the attention computation, running on its own slice of each token's representation.** A layer runs many of them in parallel — this is multi-head attention.

Splitting the work this way lets different heads specialize: one may track syntactic neighbors, another long-range references, another document structure. The total amount of data stored per token does not change by splitting into heads; it is the same width, divided up.

Head counts matter for cache sizing. The number of *key/value* heads, not query heads, is what multiplies into KV-cache size.

.. seealso::

   :doc:`attention` | :doc:`query-key-value` | :doc:`grouped-query-attention`
