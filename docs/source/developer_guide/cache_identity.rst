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
versioned token. The codec does not impose backend-specific name-length limits;
storage backends report unsupported names through their normal I/O results.

``ObjectKey`` keeps its flat constructor and accepts ``tags=`` for callers that
do not construct ``ObjectKeyIdentity`` directly. ``EncodedObjectKey`` and the
internal identity protobuf preserve tags across their existing projections.
Multiprocess clients choose tags with string-valued
``lmcache.tag.<name>`` entries in ``IPCCacheServerKey.request_configs``. The
server applies those tags to every expanded object key and keeps pending
transfers isolated by the same mapping. Invalid matching tags reject only their
RPC instead of being dropped or leaving the client waiting for a timeout.
Control APIs are handled by a separate integration layer.
