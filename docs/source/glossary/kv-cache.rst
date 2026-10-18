.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: kv-cache
   question: What is a KV cache?
   aliases: KV caching, key-value cache, KV store
   related: prefill, decode, query-key-value, prefix-caching, cache-hit-rate
   tier: 2
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/kv-cache
   schema_type: DefinedTerm
   seo_title: What is a KV cache? | LMCache
   meta_description: A KV cache is the stored collection of Key and Value tensors a model has already computed for the tokens it has seen, kept so those tokens never have...
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-kv-cache:

KV cache
========

.. meta::
   :description: A KV cache is the stored collection of Key and Value tensors a model has already computed for the tokens it has seen, kept so those tokens never have...
   :keywords: KV cache, KV caching, key-value cache, KV store

**A KV cache is the stored collection of Key and Value tensors a model has already computed for the tokens it has seen, kept so those tokens never have to be processed again.** It spans every cached token position at every transformer layer.

You can think of it as the model's working memory of an exchange. It only grows as the session continues.

Without it, producing the 100th token means recomputing Keys and Values for all 99 before it, and the cost compounds: total work across a reply grows roughly with the square of its length. A 100-token reply costs on the order of 5,000 recomputations; a 1,000-token reply costs roughly 500,000 — ten times the length, about a hundred times the work. With a cache, each step contributes exactly one new Key and Value, and cost grows linearly.

The tradeoff is memory. Cache footprint scales with cached tokens, layers, KV heads, head dimension, and bytes per value, and it grows separately for every active sequence whose state cannot be shared:

::

   2 × layers × tokens × kv_heads × head_dim × bytes_per_value


For Llama 3 8B at BF16, that works out to **128 KiB per token per sequence** — about 12.5 MiB for 100 tokens, and roughly 1 GiB for 8,192 tokens, before allocator overhead. That memory competes directly with model weights, activations, and everything else the GPU has to hold, which is what makes cache placement a real systems problem rather than a free win.

.. seealso::

   :doc:`prefill` | :doc:`decode` | :doc:`query-key-value` | :doc:`prefix-caching` | :doc:`cache-hit-rate`
