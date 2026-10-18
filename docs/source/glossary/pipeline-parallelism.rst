.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: pipeline-parallelism
   question: What is pipeline parallelism?
   aliases: PP
   related: tensor-parallelism, expert-parallelism, interconnects
   tier: 4
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/pipeline-parallelism
   schema_type: DefinedTerm
   seo_title: What is pipeline parallelism? | LMCache
   meta_description: Pipeline parallelism assigns different whole layers to different GPUs, so a token's forward pass flows through them like an assembly line — GPU 0...
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-pipeline-parallelism:

Pipeline parallelism (PP)
=========================

.. meta::
   :description: Pipeline parallelism assigns different whole layers to different GPUs, so a token's forward pass flows through them like an assembly line — GPU 0...
   :keywords: Pipeline parallelism (PP), PP

**Pipeline parallelism assigns different whole layers to different GPUs**, so a token's forward pass flows through them like an assembly line — GPU 0 handles layers 1–20, GPU 1 handles 21–40.

Communication happens only at stage handoffs, far less than TP requires, so PP tolerates slower interconnects and can span separate machines. The cost is pipeline "bubbles" — idle GPU time — unless enough requests are kept in flight to keep every stage busy.

.. seealso::

   :doc:`tensor-parallelism` | :doc:`expert-parallelism` | :doc:`interconnects`
