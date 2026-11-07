.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: inference
   question: What is inference?
   aliases: model inference, LLM inference
   related: token, prefill, decode, model-weights
   tier: 1
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/inference
   schema_type: DefinedTerm
   seo_title: What is inference? | LMCache
   meta_description: Inference is the process of running a trained model to produce an output — as opposed to training, which is the process of producing the model in the...
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-inference:

Inference
=========

.. meta::
   :description: Inference is the process of running a trained model to produce an output — as opposed to training, which is the process of producing the model in the...
   :keywords: Inference, model inference, LLM inference

**Inference is the process of running a trained model to produce an output** — as opposed to training, which is the process of producing the model in the first place.

The interface makes it look like words flow in and words flow out. Underneath, the hardware is moving and transforming large arrays of numbers: your prompt becomes numbers, those numbers pass through many layers of matrix multiplication, and the result is a set of probabilities for which token should come next. Those arrays occupy memory, usually GPU memory, and the intermediate results produced along the way have to live somewhere until the next operation needs them.

Almost everything expensive about running a language model in production is an inference problem, not a training problem — and most of the expense is about memory and data movement rather than raw arithmetic.

.. seealso::

   :doc:`token` | :doc:`prefill` | :doc:`decode` | :doc:`model-weights`
