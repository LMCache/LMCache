.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable; they are the
   contract between this file and the published page.

   slug: cache-aware-routing
   question: What is cache-aware routing?
   aliases: prefix-aware routing, cache locality routing
   related: prefix-caching, cache-hit-rate, load-balancer, pd-disaggregation
   tier: 2
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/cache-aware-routing
   schema_type: DefinedTerm
   seo_title: What is cache-aware routing? | LMCache
   meta_description: Cache-aware routing sends requests likely to share a prefix to the same replica, so the cached data they need is actually reachable.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-cache-aware-routing:

Cache-aware routing
===================

.. meta::
   :description: Cache-aware routing sends requests likely to share a prefix to the same replica, so the cached data they need is actually reachable.
   :keywords: Cache-aware routing, prefix-aware routing, cache locality routing

**Cache-aware routing sends requests likely to share a prefix to the same replica, so the cached data they need is actually reachable.**

In any deployment with more than one replica, the replica that receives a request determines whether a relevant cached prefix exists locally at all.
A request routed to a replica that does not hold the matching cache misses on that replica, however well its cache is configured, unless the deployment shares cache data between replicas (for example, through a shared storage tier or peer-to-peer transfer).

A standard Kubernetes Service picks a backend Pod at random by default and does not inspect request content, so it has no concept of cache locality.
Routing is therefore a separate lever on effective hit rate, and a multi-replica benchmark should state which routing policy it used.

.. seealso::

   :doc:`prefix-caching` | :doc:`cache-hit-rate` | :doc:`kubernetes` | :doc:`pd-disaggregation`
