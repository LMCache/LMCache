# `full_attention_only` store policy

Store policy for hybrid-attention models served with
`--separate-object-groups`. Full-attention object groups are written through
to L2 exactly as with `default`, and keep their clean copy in L1.
Sliding-window object groups are never written to L2 by the store path: their
chunks stay L1-resident, and the only way one reaches L2 is the commit path
([../../multiprocess/commit_policy.md](../../multiprocess/commit_policy.md)),
which copies a finished turn's final window there.

## Why

In a hybrid-attention model the KV of a sliding-window layer is reusable only
within the window: resuming a sequence at token offset `p` needs the
full-attention KV of every chunk in `[0, p)` but the sliding-window KV of the
last `w` chunks only. Every older sliding-window chunk is dead weight for
prefix reuse, and those chunks are the large ones (an order of magnitude more
bytes per chunk than a full-attention group). Writing them through to L2
spends most of the L2 bandwidth and space on KV that is never read back.

## Where the classification comes from

The policy does not carry a static list of sliding-window group ids; that
would break multi-model serving, where two models disagree on which group is
which. It asks `ObjectGroupClassifier`
(`lmcache/v1/distributed/object_group_classifier.py`), a reference-counted,
thread-safe map from `model_name` to the `AttnWindowDesc` the workers
registered:

```
worker registers KV cache
  -> LayoutDescRegistry.register(model_name, world_size, ..., attn_desc)
     -> StorageManager.object_group_classifier.register(model_name, attn_desc)
```

`StorageManager` owns the classifier and hands it to the policy through
`create_store_policy(name, classifier)`. `MPCacheServerContext` passes that
same classifier to the `LayoutDescRegistry` it builds, which is why the
`distributed` layer never imports `lmcache.v1.multiprocess`.

Registration reaches the classifier inside `register_kv_cache` /
`register_kv_cache_engine_driven_context`, i.e. strictly before the worker can
issue a store for that model, so no chunk is classified against a missing
layout in normal operation.

Keying by `model_name` alone is deliberate: `ObjectKey` has no `world_size`
field, and per-group attention windows do not depend on tensor parallelism.
`AttnWindowDesc.world_size` is therefore ignored when descriptors are
compared. Re-registering one model name with a different `num_chunks_in_sw`
makes the classifier raise `ValueError`, but `LayoutDescRegistry` catches it,
logs an error and lets the KV-cache registration succeed: the model keeps the
layout registered first, and a store whose group layout no longer matches is
classified against that first layout. One server cannot serve two
incompatible layouts under a single name, but a conflict does not fail the
worker.

The last `unregister` for a model drops the entry; keys of that model then
classify as `UNKNOWN`.

## Decisions

| Key class | L2 store | L1 after store |
|---|---|---|
| `FULL_ATTENTION` | every adapter | kept |
| `SLIDING_WINDOW` | none | kept |
| `UNKNOWN` | every adapter | kept |

`UNKNOWN` (model not registered, or `object_group_id` outside the registered
descriptor's `[0, num_object_groups)` range) is treated as full attention on
purpose: an extra L2 write costs write bandwidth, a wrongly skipped write can
cost the only copy of a live window and therefore a full prefill. Each such
key is logged at debug level.

`select_l1_deletions` returns `[]` for every class: the clean full-attention
copy stays in L1 so later reads hit L1 and eviction can discard it for free.

Key order within each adapter's list follows the order of the incoming batch.

## Eviction

A sliding-window chunk this policy kept out of L2 has no L2 copy, so when L1
evicts it, it is gone. That is the intended outcome for chunks behind a
window: nothing reads them again. For the window a follow-up turn will need,
the commit path copies it to L2 at the turn boundary, before L1 pressure can
reach it; a window that is evicted before it was committed costs one prefill.

## Validation

`validate_server_config` (`lmcache/v1/multiprocess/config.py`, called at the
top of `run_cache_server`) rejects `--l2-store-policy full_attention_only`
without `--separate-object-groups`: without the split, every layer shares one
full-attention object group, nothing is ever classified as sliding-window, and
the policy silently degenerates to `default`.
