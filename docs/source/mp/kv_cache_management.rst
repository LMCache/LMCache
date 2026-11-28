.. _mp_kv_cache_management:

KV Cache Management
===================

.. note::

   The multiprocess (MP) cache-management surface (operations such as
   ``move``, ``pin``, and ``compress``) is **coming soon**.

   Until it lands, the legacy management commands remain available from
   `LMCache_inprocess <https://github.com/LMCache/LMCache_inprocess>`_.

Recomputed missing blocks
-------------------------

The single-GPU full-attention vLLM and SGLang adapters track failed delivery
ranges independently of the normal stored-prefix cursor. After the engine
commits recomputation, those ranges become eligible for ordinary asynchronous
``STORE``. Healthy suffix chunks are not resubmitted solely because an earlier
chunk failed.

This bookkeeping correction adds no storage policy or retry loop. GPU source
ownership follows the existing transfer pins; L1 admission, deduplication and
L2 offload use the same path as any newly computed block. A successful ordinary
``STORE`` acknowledgment covers L1 admission, not completion of a CMX PUT.
External write completion can be verified with existing L2 tracing.

A request-scoped delivery failure does not prove external-object loss. The
adapter conservatively removes its already-stored assumption for the failed
range; lower cache layers retain their normal miss and deduplication behavior.
