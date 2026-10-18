.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: tokenization
   question: What is tokenization?
   aliases: tokenizer, tokenizing
   related: token, embedding, inference
   tier: 1
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/tokenization
   schema_type: DefinedTerm
   seo_title: What is tokenization? | LMCache
   meta_description: Tokenization is the step that splits raw text into tokens the model recognizes.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-tokenization:

Tokenization
============

.. meta::
   :description: Tokenization is the step that splits raw text into tokens the model recognizes.
   :keywords: Tokenization, tokenizer, tokenizing

**Tokenization is the step that splits raw text into tokens the model recognizes.**

It happens before any model computation. The tokenizer is specific to the model — the same sentence produces a different number of tokens under different tokenizers, which is why token counts from one vendor's model do not transfer cleanly to another's.

Tokenization is also where surprises come from in cost and context accounting: unusual formatting, non-English text, and long strings of code or identifiers can consume far more tokens than their character count suggests.

.. seealso::

   :doc:`token` | :doc:`embedding` | :doc:`inference`
