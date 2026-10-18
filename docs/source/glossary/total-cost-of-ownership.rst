.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: total-cost-of-ownership
   question: What is TCO for LLM inference?
   aliases: TCO, total cost of ownership, cost per GPU hour, cost per GB
   related: cache-tiering, qps, decode-throughput
   tier: 3
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/total-cost-of-ownership
   schema_type: DefinedTerm
   seo_title: What is TCO for LLM inference? | LMCache
   meta_description: TCO is the cost of running a deployment over its lifetime, not its sticker price.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-total-cost-of-ownership:

Total cost of ownership (TCO)
=============================

.. meta::
   :description: TCO is the cost of running a deployment over its lifetime, not its sticker price.
   :keywords: Total cost of ownership (TCO), TCO, total cost of ownership, cost per GPU hour, cost per GB

**TCO is the cost of running a deployment over its lifetime, not its sticker price.** In caching discussions it usually shows up as a comparison between two units: dollars per GB of tiered storage against dollars per GPU-hour.

That comparison is the bet a caching layer is fundamentally making — cheaper storage substituting for more expensive GPU memory and compute.

.. note::

   **Sourcing note:** public numbers in this space are overwhelmingly vendor-sourced, including ours. Any figure we publish should be labeled as such, and independent GPU and storage pricing is worth gathering before we put dollar amounts in front of readers.


---

.. seealso::

   :doc:`cache-tiering` | :doc:`qps` | :doc:`decode-throughput`
