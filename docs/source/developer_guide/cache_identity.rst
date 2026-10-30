Cache identity tags
===================

``ObjectKeyIdentity.tags`` adds opaque string dimensions to cache identity.
Tags are copied into an immutable, name-sorted mapping, so insertion order and
later caller mutation do not affect equality, hashing, or storage names.

.. code-block:: python

   identity = ObjectKeyIdentity(
       chunk_hash=chunk_hash,
       kv_rank=kv_rank,
       model_name=model_name,
       tags={"adapter_revision": "abc123"},
   )

L2 adapters use ``lmcache.v1.distributed.key_codec`` for canonical persistent
names. Untagged keys keep their existing names; tagged keys append a reversible
versioned token. Tagged filesystem names must fit within 255 UTF-8 bytes.

``ObjectKey`` keeps its flat constructor and accepts ``tags=`` for callers that
do not construct ``ObjectKeyIdentity`` directly. ``EncodedObjectKey`` and the
internal identity protobuf preserve tags across their existing projections.
Multiprocess clients choose tags with string-valued
``lmcache.tag.<name>`` entries in ``IPCCacheServerKey.request_configs``. The
server applies those tags to every expanded object key and keeps pending
transfers and session release checks isolated by the same mapping. Control APIs
are handled by a separate integration layer.
