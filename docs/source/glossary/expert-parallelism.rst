.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
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
   meta_description: Expert parallelism places different experts of a mixture-of-experts model on different GPUs, routing each token to whichever GPU holds the expert its...
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-expert-parallelism:

Expert parallelism (EP)
=======================

.. meta::
   :description: Expert parallelism places different experts of a mixture-of-experts model on different GPUs, routing each token to whichever GPU holds the expert its...
   :keywords: Expert parallelism (EP), EP, MoE parallelism

**Expert parallelism places different experts of a mixture-of-experts model on different GPUs**, routing each token to whichever GPU holds the expert its router selected.

The communication pattern is an all-to-all shuffle of tokens out to their assigned GPU and back — different in shape from TP's per-layer combine, and applicable only to MoE architectures.

Whichever parallelism type is used, splitting a model across N GPUs also splits the total memory pool across them, changing how much room is left per GPU for cached KV data.

.. seealso::

   :doc:`tensor-parallelism` | :doc:`pipeline-parallelism`
