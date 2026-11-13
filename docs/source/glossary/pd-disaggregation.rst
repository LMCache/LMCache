.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable; they are the
   contract between this file and the published page.

   slug: pd-disaggregation
   question: What is prefill-decode (P/D) disaggregation?
   aliases: P/D disaggregation, prefill decode disaggregation, disaggregated serving
   related: prefill, decode, cache-tiering, cache-aware-routing
   tier: 2
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/pd-disaggregation
   schema_type: DefinedTerm
   seo_title: What is prefill-decode (P/D) disaggregation? | LMCache
   meta_description: P/D disaggregation is running prefill and decode on separate GPU pools or machines instead of both phases on the same worker.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-pd-disaggregation:

Prefill–decode (P/D) disaggregation
===================================

.. meta::
   :description: P/D disaggregation is running prefill and decode on separate GPU pools or machines instead of both phases on the same worker.
   :keywords: Prefill–decode (P/D) disaggregation, P/D disaggregation, prefill decode disaggregation, disaggregated serving

**P/D disaggregation is running prefill and decode on separate GPU pools or machines instead of both phases on the same worker.**

Prefill is compute-bound and decode is memory-bandwidth-bound, so giving each phase its own pool lets each one run closer to its own hardware limit instead of competing for the same GPUs.

The cost is that the KV cache computed during prefill must be transferred to wherever decode runs. Cache transfer speed therefore becomes part of the latency of every request, instead of an occasional fallback to a remote storage tier.

.. seealso::

   :doc:`prefill` | :doc:`decode` | :doc:`cache-tiering` | :doc:`cache-aware-routing`
