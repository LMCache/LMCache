.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: query-key-value
   question: What are Query, Key, and Value in attention?
   aliases: QKV, Q K V, query key value
   related: attention, kv-cache, attention-head, grouped-query-attention
   tier: 1
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/query-key-value
   schema_type: DefinedTerm
   seo_title: What are Query, Key, and Value in attention? | LMCache
   meta_description: Query, Key, and Value are three vectors derived from each token's current representation, and together they work like a lookup.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-query-key-value:

Query, Key, and Value (QKV)
===========================

.. meta::
   :description: Query, Key, and Value are three vectors derived from each token's current representation, and together they work like a lookup.
   :keywords: Query, Key, and Value (QKV), QKV, Q K V, query key value

**Query, Key, and Value are three vectors derived from each token's current representation, and together they work like a lookup.** Each token asks a question, offers a label to be matched against, and holds the content it hands over if it gets picked.

- **Query** — what this token is looking for.
- **Key** — what this token offers, for other tokens comparing against it.
- **Value** — the information this token passes along if it is attended to.

To decide how much "it" should attend to "cat," the model compares "it"'s Query against "cat"'s Key. Repeated against every earlier token, that produces a set of attention weights, and the token's output is built by blending the earlier tokens' Values in proportion to those weights. Keys decide *how much* to attend; Values are the content that actually gets used.

These are learned numerical operations, not literal questions or stored definitions — but the lookup framing is the right mental model.

Only two of the three get cached. A Query is used once, at the moment its own token is processed, and can be discarded. The Keys and Values are what every later token keeps needing. Hence: **KV** cache.

.. seealso::

   :doc:`attention` | :doc:`kv-cache` | :doc:`attention-head` | :doc:`grouped-query-attention`
