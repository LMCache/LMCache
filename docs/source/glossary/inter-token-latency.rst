.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: inter-token-latency
   question: What is inter-token latency (ITL) or time per output token (TPOT)?
   aliases: ITL, TPOT, time per output token, inter token latency
   related: decode, interactivity, ttft, decode-throughput
   tier: 3
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/inter-token-latency
   schema_type: DefinedTerm
   seo_title: What is inter-token latency (ITL) or time per output token (TPOT)? | LMCache
   meta_description: ITL is the delay between each subsequent output token once generation has started.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-inter-token-latency:

Inter-token latency (ITL / TPOT)
================================

.. meta::
   :description: ITL is the delay between each subsequent output token once generation has started.
   :keywords: Inter-token latency (ITL / TPOT), ITL, TPOT, time per output token, inter token latency

**ITL is the delay between each subsequent output token once generation has started.** Time per output token (TPOT) is the same measurement under a different name.

Where TTFT governs how long a user waits before anything appears, ITL governs how choppy the response feels once it starts streaming.

.. seealso::

   :doc:`decode` | :doc:`interactivity` | :doc:`ttft` | :doc:`decode-throughput`
