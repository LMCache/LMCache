.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: qps
   question: What is QPS in inference serving?
   aliases: queries per second, requests per second, RPS
   related: decode-throughput, concurrency, tail-latency
   tier: 3
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/qps
   schema_type: DefinedTerm
   seo_title: What is QPS in inference serving? | LMCache
   meta_description: QPS is completed requests per second.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-qps:

QPS (queries per second)
========================

.. meta::
   :description: QPS is completed requests per second.
   :keywords: QPS (queries per second), queries per second, requests per second, RPS

**QPS is completed requests per second.** Of all the throughput metrics, it maps most directly onto the question an operator actually asks: how many users can this deployment serve?

.. seealso::

   :doc:`decode-throughput` | :doc:`concurrency` | :doc:`tail-latency`
