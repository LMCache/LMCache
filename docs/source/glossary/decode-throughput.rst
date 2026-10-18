.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: decode-throughput
   question: What is decode throughput?
   aliases: output token throughput, generation throughput
   related: prefill-throughput, qps, concurrency, interactivity
   tier: 3
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/decode-throughput
   schema_type: DefinedTerm
   seo_title: What is decode throughput? | LMCache
   meta_description: Decode throughput is the aggregate rate of output tokens per second across all concurrent requests.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-decode-throughput:

Decode throughput
=================

.. meta::
   :description: Decode throughput is the aggregate rate of output tokens per second across all concurrent requests.
   :keywords: Decode throughput, output token throughput, generation throughput

**Decode throughput is the aggregate rate of output tokens per second across all concurrent requests.**

It measures steady-state generation capacity, which is a different axis from latency — and the two trade off against each other as concurrency changes.

.. seealso::

   :doc:`prefill-throughput` | :doc:`qps` | :doc:`concurrency` | :doc:`interactivity`
