.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable; they are the
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
   meta_description: An attention head is one parallel instance of attention within a layer, projecting each token into its own smaller query/key/value space.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-attention-head:

Attention head
==============

.. meta::
   :description: An attention head is one parallel instance of attention within a layer, projecting each token into its own smaller query/key/value space.
   :keywords: Attention head, attention heads, multi-head attention, MHA

**An attention head is one parallel instance of attention within a layer, projecting each token into its own smaller query/key/value space.** A layer runs many of them in parallel, which is called multi-head attention.

Splitting the work this way lets different heads specialize: one may track syntactic neighbors, another long-range references, another document structure. In many models the heads together are exactly as wide as the model's hidden size, so splitting into heads divides the existing width instead of adding to it. Llama 3 8B, for example, has 32 heads of 128 dimensions each, which equals its hidden size of 4,096.

Head counts matter for cache sizing. In standard multi-head and grouped-query attention, the number of *key/value* heads, not query heads, is what multiplies into KV cache size.

.. seealso::

   :doc:`attention` | :doc:`query-key-value` | :doc:`grouped-query-attention`
