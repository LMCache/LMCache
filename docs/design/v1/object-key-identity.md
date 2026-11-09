# Typed object identity

`ObjectKeyIdentity` owns the immutable matching fields of `ObjectKey`:
chunk hash, model, packed KV rank, object group and cache salt. Read-only key
properties retain existing field access without storing a second copy.

The flat constructor stays available:

```python
key = ObjectKey(b"chunk", "model", 0, cache_salt="tenant")
same = ObjectKey(identity=key.identity)
```

Dataclass subclasses declare and construct the inherited `identity` field;
they do not inherit the base class's custom flat constructor.

Construction validates the model, salt, group and tags and freezes the tag
mapping. A prompt expansion holds all of those fixed while varying the chunk
hash and KV rank, so it builds one validated identity per object group and
derives the rest with `with_chunk(chunk_hash, kv_rank)`, which adopts the
validated fields as they are. Identities and frozen tag mappings cache their
hash: an object key is hashed on every dict lookup that reaches it, many times
per request, and it never changes.

Runtime `dataclasses.replace(key, object_group_id=...)` remains supported.
For statically checked code, replace the declared identity field explicitly:

```python
other = replace(key, identity=replace(key.identity, object_group_id=1))
```

IPC requests and flat HTTP `EncodedObjectKey` fields keep their existing shape.
Internal P2P and memory-orchestrator messages carry a nested identity and require
both services to upgrade together. Their shared structural codec validates the
same model, salt and object-group invariants as local construction.

Storage formatting remains adapter-owned, outside the identity type. The identity refactor alone does not add request extensions or
change persistent names.
