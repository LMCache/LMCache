.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
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

The reason: prefill is compute-bound and decode is memory-bandwidth-bound, so dedicating separate pools lets each run closer to its own hardware ceiling rather than sharing one pool awkwardly.

The cost: the KV cache computed during prefill has to physically move to wherever decode happens. That turns cache transfer speed from an occasional remote-tier fallback into a first-class part of the latency story.

A subtlety most explainers miss — caching does not shrink the attention computation itself. It removes the redundant recompute around it.

.. seealso::

   :doc:`prefill` | :doc:`decode` | :doc:`cache-tiering` | :doc:`cache-aware-routing`
