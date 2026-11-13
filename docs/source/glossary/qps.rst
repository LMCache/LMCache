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
   meta_description: QPS (queries per second) is a request rate; in LLM serving, one query is one request, such as a single chat-completion call.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-qps:

QPS (queries per second)
========================

.. meta::
   :description: QPS (queries per second) is a request rate; in LLM serving, one query is one request, such as a single chat-completion call.
   :keywords: QPS (queries per second), queries per second, requests per second, RPS

**QPS (queries per second) is a request rate; in LLM serving, one query is one request, such as a single chat-completion call.**

Benchmarks use the name for two different numbers, so check which one a result reports. The first is the rate at which the benchmark *sends* requests, which sets the offered load; vLLM's ``--request-rate`` and the ``--mrc-qps`` option of ``lmcache bench engine`` work this way. The second is the rate at which requests *complete*, often reported as requests per second or request throughput. While the server keeps up, the two rates are equal. When it cannot keep up, requests wait in a queue, latency grows, and the completed rate stays below the send rate.

Requests differ in prompt length and output length, so a QPS figure is comparable only between benchmarks that use the same workload.

.. seealso::

   :doc:`decode-throughput` | :doc:`concurrency` | :doc:`tail-latency`
