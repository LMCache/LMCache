.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable; they are the
   contract between this file and the published page.

   slug: expert-parallelism
   question: What is expert parallelism?
   aliases: EP, MoE parallelism
   related: tensor-parallelism, pipeline-parallelism
   tier: 4
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/expert-parallelism
   schema_type: DefinedTerm
   seo_title: What is expert parallelism? | LMCache
   meta_description: Expert parallelism places different experts of a mixture-of-experts model on different GPUs, routing each token to the GPUs that hold the experts its...
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-expert-parallelism:

Expert parallelism (EP)
=======================

.. meta::
   :description: Expert parallelism places different experts of a mixture-of-experts model on different GPUs, routing each token to the GPUs that hold the experts its...
   :keywords: Expert parallelism (EP), EP, MoE parallelism

**Expert parallelism places different experts of a mixture-of-experts model on different GPUs**, routing each token to the GPUs that hold the experts its router selected.

The communication pattern is an all-to-all exchange: each token is sent out to the GPUs that hold its selected experts, and the results are sent back. This pattern differs in shape from TP's per-layer combine, and it applies only to MoE architectures.

EP frees memory indirectly: each GPU holds only a fraction of the expert weights, which leaves more room for KV cache. EP does not shard attention, however, so how KV data is distributed depends on how the attention layers are parallelized (data, tensor, or pipeline parallel), not on EP itself.

.. seealso::

   :doc:`tensor-parallelism` | :doc:`pipeline-parallelism`
