.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable; they are the
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

In a serving deployment, the configured context length also affects memory.
In vLLM the setting is ``--max-model-len``, and the engine refuses to start if its KV cache memory cannot hold at least one request of that full length.
vLLM does not reserve that much memory for every request; it allocates cache blocks as each request grows.
Lowering the limit is one of the settings vLLM's documentation suggests when GPU memory is short.

Long contexts are where cache reuse saves the most computation, and they are also the source of the memory pressure that makes keeping a cache difficult.

---

.. seealso::

   :doc:`token` | :doc:`kv-cache` | :doc:`prefill` | :doc:`quantization`
