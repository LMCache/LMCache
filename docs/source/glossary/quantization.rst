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

**Quantization is storing and computing numbers at lower precision to save memory and time.** Two separate settings are often confused:

- Weight precision: how the model's own parameters are stored and computed. In vLLM, ``--dtype`` selects only 16- or 32-bit formats (fp16, bf16, fp32). Lower-precision weights such as fp8 or fp4 come from a quantized model checkpoint or from the ``--quantization`` option. On hardware with dedicated support, lower precision makes matrix multiplication faster, which speeds up both prefill and decode regardless of caching.
- KV cache precision (``--kv-cache-dtype``, a separate flag): how the cached key/value tensors are stored, independent of weight precision. A lower-precision KV cache takes less memory per token, so more tokens fit in the same pool, which raises the ceiling on hit rate. Depending on the attention backend, the kernel either converts the cached values back to higher precision when it reads them or runs attention directly in fp8.

Precision is also hardware-dependent. vLLM runs fp8 computation only on NVIDIA GPUs with compute capability 8.9 or higher (Ada Lovelace, Hopper, Blackwell). On older GPUs such as Turing and Ampere, it runs an fp8 model as weight-only fp8: the weights are stored in fp8, but the arithmetic runs at 16-bit precision, so the compute speedup of fp8 is not available. The same setting therefore behaves differently on different hardware.

Lower precision is an approximation, so it can cost output quality. Below a certain precision, which depends on the model, the quality loss becomes measurable.

.. seealso::

   :doc:`model-weights` | :doc:`kv-cache` | :doc:`cache-hit-rate`
