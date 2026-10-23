# Typed object identity

`ObjectKeyIdentity` owns the immutable matching fields of `ObjectKey`:
chunk hash, model, packed KV rank, object group and cache salt. Consumers read
these fields through `key.identity`, for example `key.identity.chunk_hash`.

`ObjectKey` uses the dataclass-generated constructor:

```python
from dataclasses import replace

from lmcache.v1.cache_identity import ObjectKeyIdentity
from lmcache.v1.distributed.api import ObjectKey

key = ObjectKey(identity=ObjectKeyIdentity(
    chunk_hash=b"chunk", model_name="model", kv_rank=0, cache_salt="tenant",
))
other = replace(key, identity=replace(key.identity, object_group_id=1))
```

IPC requests and flat HTTP `EncodedObjectKey` fields keep their existing shape.
Internal P2P and memory-orchestrator messages carry a nested identity and require
both services to upgrade together. Their shared structural codec validates the
same model, salt and object-group invariants as local construction.

Storage formatting remains adapter-owned, outside the identity type. The
identity refactor alone does not add request extensions or change persistent names.
