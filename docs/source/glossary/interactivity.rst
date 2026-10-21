.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: interactivity
   question: What is interactivity in inference benchmarks?
   aliases: tokens per second per user, tok/s/user
   related: inter-token-latency, decode-throughput, concurrency
   tier: 3
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/interactivity
   schema_type: DefinedTerm
   seo_title: What is interactivity in inference benchmarks? | LMCache
   meta_description: Interactivity is tokens delivered per second per user — roughly the inverse of inter-token latency.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-interactivity:

Interactivity
=============

.. meta::
   :description: Interactivity is tokens delivered per second per user — roughly the inverse of inter-token latency.
   :keywords: Interactivity, tokens per second per user, tok/s/user

**Interactivity is tokens delivered per second per user** — roughly the inverse of inter-token latency.

It is the same underlying phenomenon as ITL, expressed as a rate rather than a delay. Some benchmarks plot it instead of ITL because "higher is better" reads more intuitively than "lower is better." Be aware of which one a chart is showing before comparing across sources.

.. seealso::

   :doc:`inter-token-latency` | :doc:`decode-throughput` | :doc:`concurrency`
