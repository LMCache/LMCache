.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable; they are the
   contract between this file and the published page.

   slug: decode
   question: What is decode in LLM inference?
   aliases: decode phase, generation phase
   related: prefill, autoregressive-generation, inter-token-latency, decode-throughput
   tier: 2
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/decode
   schema_type: DefinedTerm
   seo_title: What is decode in LLM inference? | LMCache
   meta_description: Decode is the phase where the model generates output, one token at a time.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-decode:

Decode
======

.. meta::
   :description: Decode is the phase where the model generates output, one token at a time.
   :keywords: Decode, decode phase, generation phase

**Decode is the phase where the model generates output, one token at a time.**

For each new token, every layer computes the Query, Key, and Value for that one position only, which is a much smaller computation than the whole-prompt pass of prefill. The Query attends to the Keys already stored for that layer, uses the corresponding Values, and continues through the rest of the model. The new Key and Value are then appended to the cache and the loop repeats.

A single decode step is memory-bandwidth-bound rather than compute-bound, because it reads the model weights and the cached Keys and Values from GPU memory but does little arithmetic for each byte it reads. Engines therefore batch decode steps from many requests together. This hardware profile differs from that of prefill, which is why some deployments separate the two phases (:doc:`P/D disaggregation <pd-disaggregation>`).

Caching does not make every decode step cost the same. As the sequence grows, the new Query has more cached Keys and Values to read and attend over. What caching removes is re-running the model for old positions to recreate state the engine already computed.

.. seealso::

   :doc:`prefill` | :doc:`autoregressive-generation` | :doc:`inter-token-latency` | :doc:`decode-throughput`
