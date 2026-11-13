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

LLM servers such as vLLM and SGLang use *continuous batching*, which the Orca paper introduced as iteration-level scheduling. The server rebuilds the batch at every generation step instead of once per batch. As a result, a request that finishes leaves the batch immediately, and a waiting request can join at the next step instead of waiting for the whole batch to finish.

Concurrency sets the trade-off between the two headline metrics. As concurrency rises, total decode throughput rises until the GPU's compute is saturated, but each user receives tokens more slowly (lower interactivity), and TTFT grows because new requests share the GPU with more running requests.

For this reason, a result that reports only throughput or only latency, without the concurrency level it was measured at, cannot be compared with a result measured at a different concurrency level, even on identical hardware.

.. seealso::

   :doc:`decode-throughput` | :doc:`interactivity` | :doc:`tail-latency` | :doc:`qps`
