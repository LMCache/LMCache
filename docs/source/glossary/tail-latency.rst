.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: tail-latency
   question: What is tail latency (p50, p90, p99)?
   aliases: p99, p90, percentile latency, tail
   related: ttft, inter-token-latency, concurrency
   tier: 3
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/tail-latency
   schema_type: DefinedTerm
   seo_title: What is tail latency (p50, p90, p99)? | LMCache
   meta_description: Tail latency is the latency experienced by the slowest requests, reported as percentiles rather than an average.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-tail-latency:

Tail latency (p50 / p90 / p99)
==============================

.. meta::
   :description: Tail latency is the latency experienced by the slowest requests, reported as percentiles rather than an average.
   :keywords: Tail latency (p50 / p90 / p99), p99, p90, percentile latency, tail

**Tail latency is the latency experienced by the slowest requests, reported as percentiles rather than an average.** p99 means 99% of requests were at least this fast, and 1% were slower.

Real request latencies are right-skewed: most are fast, but a subset run much slower because of cache misses, resource contention, eviction, or retries, stretching a long tail to the right. A mean barely moves in response to that tail. Percentiles look directly at where it sits.

This matters because the user who landed in the slow 1% had a genuinely bad experience regardless of how good the average looked. Production SLOs are usually written against tail latency for exactly this reason — which means a benchmark reporting only mean TTFT can look substantially better than what real users feel.

.. seealso::

   :doc:`ttft` | :doc:`inter-token-latency` | :doc:`concurrency`
