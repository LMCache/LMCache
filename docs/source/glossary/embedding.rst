.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: embedding
   question: What is an embedding?
   aliases: embeddings, embedding vector, vector
   related: token, attention, tokenization
   tier: 1
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/embedding
   schema_type: DefinedTerm
   seo_title: What is an embedding? | LMCache
   meta_description: An embedding is the vector — a list of numbers — that represents a token inside the model.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-embedding:

Embedding
=========

.. meta::
   :description: An embedding is the vector — a list of numbers — that represents a token inside the model.
   :keywords: Embedding, embeddings, embedding vector, vector

**An embedding is the vector — a list of numbers — that represents a token inside the model.** A model cannot multiply words, so every token is first converted into a fixed-length list of learned values it can operate on.

The mapping is built so that related tokens land near each other in that space. "Cat" and "dog" end up close together; "cat" and "car" do not, despite looking similar as text.

The initial embedding is only a starting point. As the vector passes through the model's layers, its representation shifts based on surrounding context — the token "bank" should contribute differently in "river bank" than in "bank account." The mechanism that lets one token take on information from others is :doc:`attention <attention>`.

.. seealso::

   :doc:`token` | :doc:`attention` | :doc:`tokenization`
