# Sparse null block policies

## Summary

The block ID that means "this group has no data here" becomes a per-group
property (`EngineGroupInfo.null_block_id`) instead of the hardcoded vLLM null
block `0`. This lets a connector register *sparse* KV groups — groups whose
chunks may legitimately hold nothing — alongside ordinary dense ones.

Backwards compatible: existing registrations omit the field, decode to `0`, and
behave exactly as before.

## Motivation

A recurrent-state group (Mamba/GDN checkpoints, state images) only materializes
real data at a checkpoint boundary. Every earlier chunk maps to a null marker.
Two problems follow from assuming that marker is always `0`.

**1. Zero is not universally null.** A group whose address space starts at a
real block 0 cannot use `0` as its absent marker, and a group using a negative
sentinel (`-1`) must never let that sentinel reach a GPU index buffer. The old
code assumed `0` meant null everywhere (`all_null_chunk_masks`) and assumed
every staged block ID was a valid index.

**2. Sparse and dense groups cannot share an object.** Object keys are content
hashes and object presence is the retrieve-side hit signal. If a sparse group
shares an object group with a dense group, a retrieve cannot distinguish "the
sparse group stored nothing at this chunk" from "this object is missing".

## The policy

`EngineGroupInfo.null_block_id: int | None = 0`

| Value | Meaning |
|---|---|
| `0` (default) | Historical vLLM reserved null block. Wire-compatible: older payloads omit the field and decode to this. |
| `None` | No null block exists; every non-negative block ID, including zero, is real data. |
| other (e.g. `-1`) | That value is the absent marker for this group. |

Over gRPC this is a `oneof null_block_policy` (`null_block_id` /
`no_null_block`) so "omitted" stays distinguishable from "explicitly zero".

`all_null_chunk_masks` consumes the per-group markers to decide which chunks are
entirely absent for an object group. Those chunks are never reserved and never
committed — storing them would hash-commit garbage that a later prefix hit would
serve.

## Forced object-group separation

`KVLayerGroupsManager` treats `separate_object_groups=False` as a *preference*.
When any kernel group declares a non-default `null_block_id`, separation is
forced and logged at INFO. This is a soundness requirement, not a tuning knob —
see problem 2 above.

This is deliberately not an error: a connector that needs sparse groups would
otherwise have to set a flag that has exactly one valid value. It *is* logged,
because object-group ids feed the object key namespace, so the forced split
changes which keys a deployment writes.

The bucket key is `(extra_tag, recurrent, sw_chunks, null_block_id)`.

## Staging

`downsample_and_stage_block_ids` accepts optional `skipped_chunks` and
`skip_first_n_tokens`. Slots the caller will not transfer are staged as `0`, so
a negative sentinel never reaches a GPU index buffer.

Zeroing is only needed by groups that can carry such a sentinel. `store` and
`retrieve` therefore pass the masking arguments only when some kernel group
declares a non-default `null_block_id`; a dense deployment takes a slice-only
fast path and produces byte-identical staging to before this feature.

`kept_blocks_per_chunk()` is the single source for the per-chunk stride into the
staged buffer. Any future caller that indexes into that buffer must use it
rather than recomputing the geometry, or the index can drift from what staging
actually wrote.

## Testing

- `tests/v1/test_kv_layer_groups_manager.py` — bucket keys, forced separation.
- `tests/v1/multiprocess/test_group_view_null_block.py` — wire defaults.
- `tests/v1/multiprocess/test_lmcache_driven_transfer_skip.py` — null masks,
  staging fast path and masked path.
- `tests/v1/multiprocess/test_native_state_lookup.py` — sparse lookup.
- `tests/v1/multiprocess/test_native_state_alias_gpu.py` — real two-process
  device transfer; requires a GPU and is skipped otherwise.
