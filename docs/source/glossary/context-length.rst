.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: context-length
   question: What is context length?
   aliases: context window, max model length, max_model_len
   related: token, kv-cache, prefill, quantization
   tier: 1
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/context-length
   schema_type: DefinedTerm
   seo_title: What is context length? | LMCache
   meta_description: Context length is the maximum number of tokens a model can attend over in a single request — prompt plus generated output together.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-context-length:

Context length
==============

.. meta::
   :description: Context length is the maximum number of tokens a model can attend over in a single request — prompt plus generated output together.
   :keywords: Context length, context window, max model length, max_model_len

**Context length is the maximum number of tokens a model can attend over in a single request** — prompt plus generated output together.

In a serving deployment this is also a memory decision, not just a model capability. The engine reserves cache space against the configured ceiling (``--max-model-len`` in vLLM), so setting it higher than a workload actually needs consumes memory that could otherwise hold cached state for more concurrent requests.

Longer contexts are exactly where cache reuse pays off most, and also where the memory pressure that makes reuse hard comes from.

---

.. seealso::

   :doc:`token` | :doc:`kv-cache` | :doc:`prefill` | :doc:`quantization`
