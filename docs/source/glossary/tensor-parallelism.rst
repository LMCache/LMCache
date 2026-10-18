.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: tensor-parallelism
   question: What is tensor parallelism?
   aliases: TP, tensor-parallel-size
   related: pipeline-parallelism, expert-parallelism, interconnects, kv-cache
   tier: 4
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/tensor-parallelism
   schema_type: DefinedTerm
   seo_title: What is tensor parallelism? | LMCache
   meta_description: Tensor parallelism splits each layer's weight matrix across several GPUs, so each GPU computes part of one matrix multiplication and the GPUs then...
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-tensor-parallelism:

Tensor parallelism (TP)
=======================

.. meta::
   :description: Tensor parallelism splits each layer's weight matrix across several GPUs, so each GPU computes part of one matrix multiplication and the GPUs then...
   :keywords: Tensor parallelism (TP), TP, tensor-parallel-size

**Tensor parallelism splits each layer's weight matrix across several GPUs**, so each GPU computes part of one matrix multiplication and the GPUs then exchange and combine their partial results before moving to the next layer.

Because that combining step happens at *every single layer*, TP needs a very fast GPU-to-GPU link — NVLink rather than PCIe — to avoid being dominated by communication. This is why TP is typically confined to GPUs inside one physical machine.

It is not a free speedup. On a slow interconnect, adding more TP can make inter-token latency worse rather than better.

.. seealso::

   :doc:`pipeline-parallelism` | :doc:`expert-parallelism` | :doc:`interconnects` | :doc:`kv-cache`
