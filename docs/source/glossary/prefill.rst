.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable; they are the
   contract between this file and the published page.

   slug: prefill
   question: What is prefill in LLM inference?
   aliases: prefill phase, prompt processing
   related: decode, kv-cache, ttft, prefill-throughput
   tier: 2
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/prefill
   schema_type: DefinedTerm
   seo_title: What is prefill in LLM inference? | LMCache
   meta_description: Prefill is the phase where the model processes the prompt, before it generates anything.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-prefill:

Prefill
=======

.. meta::
   :description: Prefill is the phase where the model processes the prompt, before it generates anything.
   :keywords: Prefill, prefill phase, prompt processing

**Prefill is the phase where the model processes the prompt, before it generates anything.**

Because every prompt token is already known, their computations can run in parallel within each layer: one matrix multiplication produces every token's Query, Key, and Value at once, and another produces every position's attention scores against every other position, in a single pass.
The :doc:`causal mask <causal-mask>` still prevents any position from using future context.

As the prompt moves through the model, the engine produces and stores a Key and Value for every prompt token at every layer.
That is the initial KV cache — the model's understanding of everything in the prompt, saved for reuse.

For prompts that are not very short, prefill is compute-bound, and it is the main compute cost in :doc:`time to first token <ttft>`.
Cache reuse skips the part of prefill that a cached prefix covers; the uncached remainder still has to be processed.

.. seealso::

   :doc:`decode` | :doc:`kv-cache` | :doc:`ttft` | :doc:`prefill-throughput`
