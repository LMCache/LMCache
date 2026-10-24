# L1 residency server module

`lmcache/v1/multiprocess/l1_residency.py` answers one question over
`server_module_call`: how much committed L1 do these chunks hold, and how
much does their whole model hold? A client that knows which chunks another
cache holds (for example, a vLLM scheduler that knows which chunks its GPU
prefix cache holds) divides the two to get the share of L1 that the other
cache duplicates.

## Definitions

- **L1 object**: one exact `ObjectKey` (model, cache salt, kv rank, object
  group, chunk hash). Each kv rank's object counts on its own.
- **Committed**: resident in an L1 manager's object table and not
  `is_temporary`, so its write finished; staging and temporary objects never
  count. This is `L1CommittedUsage` in `lmcache/v1/distributed/internal_api.py`,
  returned by `L1Manager.get_committed_usage`.
- **model_\***: committed objects and bytes of the model, over every kv
  rank, object group and cache salt. **matched_\***: those among the queried keys.

## Request and reply

```
L1ResidencyQuery(model_name, world_size, chunk_hashes_by_salt)
  -> keys: per salt, per chunk hash, one ObjectKey per kv rank of
     world_size, object group 0 (as a lookup of that model would build)
  -> StorageManager.get_l1_committed_usage(keys, model_name)
       = sum of L1Manager.get_committed_usage over every L1 manager
L1ResidencyReply(probe_hash, matched_objects, matched_bytes,
                 model_objects, model_bytes)
```

`probe_hash` is the server's `TokenHasher` hash of a fixed probe chunk
(`compute_probe_hash`). A client that hashes the same probe differently
(another `hash_algorithm`, or another `PYTHONHASHSEED` for a non-blake3 one)
would match nothing, so it can detect that instead of reporting 0.

`get_committed_usage` takes the L1 manager lock once, looks the keys up and
scans the object table for the model count. It takes no read lock, updates
no recency, and emits no event or listener call, so measuring does not
change eviction.

The module reads L1 directly instead of following the observability event
bus: the bus tail-drops events once its 10,000-entry queue fills, so an
index built from it could drift.

## Limits

- Only `separate_object_groups=False` (the default) is supported: a chunk is
  then one object of group 0 per kv rank. The module raises `ValueError` at
  startup when the server separates object groups.
- With several L1 managers, each is read under its own lock in turn, so the
  sum is consistent per manager, not across managers; a key held by two of
  them counts twice, in both numerator and denominator.
- Each query scans each L1 manager's whole object table under its lock,
  like `report_status`: the cost grows with every resident object, of every
  model on the server, not only the queried model's committed ones.

## Usage

The module is opt-in and lives outside the built-in `modules` package, so a
server that does not load it pays nothing:

```bash
--server-module '{"module_path": "lmcache.v1.multiprocess.l1_residency"}'
```
