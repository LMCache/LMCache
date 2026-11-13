.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable; they are the
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
   meta_description: ITL is the time between two consecutive output tokens once generation has started.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-inter-token-latency:

Inter-token latency (ITL / TPOT)
================================

.. meta::
   :description: ITL is the time between two consecutive output tokens once generation has started.
   :keywords: Inter-token latency (ITL / TPOT), ITL, TPOT, time per output token, inter token latency

**ITL is the time between two consecutive output tokens once generation has started.** Time per output token (TPOT) is the same quantity averaged over one request, and many tools use the two names interchangeably. vLLM reports them as separate metrics, however: ITL as the distribution of every gap between tokens, and TPOT as one average per request.

Where TTFT governs how long a user waits before anything appears, ITL governs how smoothly the response streams once it starts.

.. seealso::

   :doc:`decode` | :doc:`interactivity` | :doc:`ttft` | :doc:`decode-throughput`
