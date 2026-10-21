.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: attention-kernel
   question: What is an attention kernel?
   aliases: kernel, FlashAttention, attention implementation
   related: attention, decode, inter-token-latency
   tier: 4
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/attention-kernel
   schema_type: DefinedTerm
   seo_title: What is an attention kernel? | LMCache
   meta_description: A kernel is a low-level GPU program implementing one specific computation as hardware-optimized code; the attention kernel is the code that actually...
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-attention-kernel:

Attention kernel
================

.. meta::
   :description: A kernel is a low-level GPU program implementing one specific computation as hardware-optimized code; the attention kernel is the code that actually...
   :keywords: Attention kernel, kernel, FlashAttention, attention implementation

**A kernel is a low-level GPU program implementing one specific computation as hardware-optimized code; the attention kernel is the code that actually runs attention's matrix multiplications.**

Two engines — or two versions of the same engine — running an identical model on identical hardware can differ in speed purely because of which attention kernel they use. Same math, different code, different speed.

This is why a benchmark comparing engine versions is partly a kernel-implementation comparison, not purely a comparison of whatever feature it claims to be testing.

.. seealso::

   :doc:`attention` | :doc:`decode` | :doc:`inter-token-latency`
