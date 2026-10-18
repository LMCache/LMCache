.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: quantization
   question: What is quantization in LLM inference?
   aliases: precision, dtype, fp8, kv cache dtype
   related: model-weights, kv-cache, cache-hit-rate
   tier: 4
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/quantization
   schema_type: DefinedTerm
   seo_title: What is quantization in LLM inference? | LMCache
   meta_description: Quantization is storing and computing numbers at lower precision to save memory and time.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-quantization:

Quantization and precision
==========================

.. meta::
   :description: Quantization is storing and computing numbers at lower precision to save memory and time.
   :keywords: Quantization and precision, precision, dtype, fp8, kv cache dtype

**Quantization is storing and computing numbers at lower precision to save memory and time.** Two separate settings get confused constantly:

- **Weight precision** (``--dtype``: fp16/bf16/fp8/fp4) — how the model's own parameters are stored and computed. Lower precision generally means faster matrix multiplication on hardware with dedicated support, speeding up both prefill and decode, independent of caching entirely.
- **KV-cache precision** (``--kv-cache-dtype``, a separate flag) — how the cached key/value tensors are stored, independent of weight precision. A lower-precision KV cache takes less memory per token, so more tokens fit in the same pool, directly raising the ceiling on hit rate — at the cost of a small dequantization step on every read.

Precision is also hardware-dependent in a way that surprises people. fp8 matrix multiplication is accelerated on GPUs with dedicated fp8 tensor cores; an older GPU without that support emulates it in software and can end up *slower* than just running bf16. The same flag behaves differently on different hardware.

And it is an approximation, not a free win. Below a certain point, depending on the model, lower precision measurably degrades output quality.

.. seealso::

   :doc:`model-weights` | :doc:`kv-cache` | :doc:`cache-hit-rate`
