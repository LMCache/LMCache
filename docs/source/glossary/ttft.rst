.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: ttft
   question: What is time to first token (TTFT)?
   aliases: TTFT, time to first token
   related: prefill, tail-latency, inter-token-latency, cache-hit-rate
   tier: 3
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/ttft
   schema_type: DefinedTerm
   seo_title: What is time to first token (TTFT)? | LMCache
   meta_description: TTFT is the delay between a request arriving and the model producing its first output token.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-ttft:

Time to first token (TTFT)
==========================

.. meta::
   :description: TTFT is the delay between a request arriving and the model producing its first output token.
   :keywords: Time to first token (TTFT), TTFT, time to first token

**TTFT is the delay between a request arriving and the model producing its first output token.** It is dominated by prefill, and it is the metric KV-cache reuse targets most directly — a hit skips re-processing context the system has already seen.

One caveat that changes how the number should be read: TTFT is measured from request *arrival*, so it includes time spent queued before the server started work at all. A benchmark that does not report the load level relative to server capacity can hide a large queueing contribution inside what looks like a pure compute number.

.. seealso::

   :doc:`prefill` | :doc:`tail-latency` | :doc:`inter-token-latency` | :doc:`cache-hit-rate`
