.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: token
   question: What is a token?
   aliases: tokens
   related: tokenization, embedding, context-length, ttft
   tier: 1
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/token
   schema_type: DefinedTerm
   seo_title: What is a token? | LMCache
   meta_description: A token is the unit of text a model actually operates on — a whole word, a piece of a word, punctuation, whitespace, or another recurring fragment,...
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-token:

Token
=====

.. meta::
   :description: A token is the unit of text a model actually operates on — a whole word, a piece of a word, punctuation, whitespace, or another recurring fragment,...
   :keywords: Token, tokens

**A token is the unit of text a model actually operates on** — a whole word, a piece of a word, punctuation, whitespace, or another recurring fragment, drawn from the model's fixed vocabulary.

A token is roughly ¾ of a word, or about four characters. Common words are usually a single token; rarer ones get split into pieces. Each token has an integer ID, which the model uses to look up its starting representation.

Tokens are the unit that nearly every other number in this glossary is denominated in: context length is measured in tokens, throughput is measured in tokens per second, KV-cache size is measured in bytes per token, and pricing is usually per million tokens.

.. seealso::

   :doc:`tokenization` | :doc:`embedding` | :doc:`context-length` | :doc:`ttft`
