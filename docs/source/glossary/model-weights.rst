.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: model-weights
   question: What are model weights?
   aliases: weights, parameters, model parameters
   related: inference, quantization, attention, tensor-parallelism
   tier: 1
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/model-weights
   schema_type: DefinedTerm
   seo_title: What are model weights? | LMCache
   meta_description: Model weights are the numbers a model learned during training and keeps fixed during inference.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-model-weights:

Model weights
=============

.. meta::
   :description: Model weights are the numbers a model learned during training and keeps fixed during inference.
   :keywords: Model weights, weights, parameters, model parameters

**Model weights are the numbers a model learned during training and keeps fixed during inference.** They are the model — an 8-billion-parameter model has 8 billion of them.

The important property for anything cache-related: weights do not grow with the prompt. A layer's projection matrices are the same size whether you send it four tokens or forty thousand. Everything that *does* grow with the prompt — the activations, the keys and values — is request-dependent data, and that is the part that has to be stored, moved, or thrown away.

That split, fixed weights versus request-dependent tensors, is the line the entire KV-caching problem sits on.

.. seealso::

   :doc:`inference` | :doc:`quantization` | :doc:`attention` | :doc:`tensor-parallelism`
