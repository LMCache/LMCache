.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: grouped-query-attention
   question: What is grouped-query attention (GQA)?
   aliases: GQA, grouped query attention
   related: attention-head, kv-cache, query-key-value
   tier: 1
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/grouped-query-attention
   schema_type: DefinedTerm
   seo_title: What is grouped-query attention (GQA)? | LMCache
   meta_description: Grouped-query attention is an architecture choice where several Query heads share a single Key/Value head, rather than each Query head having its own.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-grouped-query-attention:

Grouped-query attention (GQA)
=============================

.. meta::
   :description: Grouped-query attention is an architecture choice where several Query heads share a single Key/Value head, rather than each Query head having its own.
   :keywords: Grouped-query attention (GQA), GQA, grouped query attention

**Grouped-query attention is an architecture choice where several Query heads share a single Key/Value head**, rather than each Query head having its own.

In standard multi-head attention the counts are equal. Under GQA there are fewer K/V heads than Q heads, which directly reduces how much key/value data has to be stored per token and read on every decode step.

This is one of the largest single levers on KV-cache footprint, and it is decided by the model architect, not by the serving stack. Llama 3 8B, for instance, has 32 Query heads and 8 Key/Value heads.

.. seealso::

   :doc:`attention-head` | :doc:`kv-cache` | :doc:`query-key-value`
