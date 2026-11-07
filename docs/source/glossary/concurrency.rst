.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: concurrency
   question: How do concurrency and batching affect inference performance?
   aliases: batching, batch size, continuous batching
   related: decode-throughput, interactivity, tail-latency, qps
   tier: 3
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/concurrency
   schema_type: DefinedTerm
   seo_title: How do concurrency and batching affect inference performance? | LMCache
   meta_description: Concurrency is how many requests a server works on at once; batching is how it groups them so the GPU processes them together.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-concurrency:

Concurrency and batching
========================

.. meta::
   :description: Concurrency is how many requests a server works on at once; batching is how it groups them so the GPU processes them together.
   :keywords: Concurrency and batching, batching, batch size, continuous batching

**Concurrency is how many requests a server works on at once; batching is how it groups them so the GPU processes them together.**

The two headline metrics trade off along this axis. Low concurrency means each request gets more GPU attention: higher interactivity, lower total throughput. High concurrency means the GPU juggles more requests: lower interactivity per user, higher total throughput.

The practical consequence for reading benchmarks: a result reporting only one of these, without stating the concurrency level it was measured at, is not comparable to another result measured at a different concurrency level — even on identical hardware.

.. seealso::

   :doc:`decode-throughput` | :doc:`interactivity` | :doc:`tail-latency` | :doc:`qps`
