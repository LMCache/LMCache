# Sparse null block policy

## Summary

The MP server has one global block ID that means "no data is present here":

```bash
lmcache server --null-block-id 0
```

The default is `0`, which preserves the existing vLLM behavior. Connectors
whose block zero is real data can select another sentinel. ATOM native-state
transfer uses:

```bash
lmcache server --null-block-id -1 --separate-object-groups
```

The setting is server-wide rather than part of `EngineGroupInfo`, so it does
not change engine registration or gRPC protocol metadata.

## Motivation

A recurrent-state group only materializes data at a checkpoint boundary. Its
earlier chunks contain an absent marker. ATOM uses `-1` for those missing STATE
chunks, while PAGE block `0` is ordinary data:

```text
chunk:      0   1   2   3
PAGE ids:   0   1   2   3     all PAGE chunks are present
STATE ids: -1  -1  -1   7     only chunk 3 has a checkpoint
```

With `--null-block-id -1`, LMCache keeps PAGE block `0`, skips the first three
STATE objects, and stores the STATE object at chunk 3. Python `None` is not a
sentinel value here; the configuration is always an integer.

PAGE and STATE should be separate object groups. A merged object is stored
only where both are present, discarding otherwise valid PAGE data at missing
STATE checkpoints. Separation keeps complete PAGE objects independently
storable. ATOM therefore uses `--separate-object-groups` explicitly.

## Store and retrieve behavior

`incomplete_chunk_masks` compares retained block IDs against the configured
global sentinel. A chunk is skipped for an object group when any required block
in any of that object's kernel groups equals the sentinel. Skipped objects are
neither reserved nor committed.

Presence is checked after per-chunk window downsampling, using the block IDs
that will actually be transferred. For example, a recurrent group with a
one-block window and two blocks per chunk must skip `[7, 0]` when `0` is the
null marker: only the null tail block is transferred. A real block outside
that window does not make the endpoint checkpoint valid. This also applies
when PAGE and STATE are separate object groups. In a merged object, an absent
STATE slice also prevents storing its attention pages; separation lets complete
attention objects remain independently storable.

LMCache already skips these copies in the transfer layer. On store, skipped
objects become `None` entries and the D2H loop does not launch a kernel for
them. On retrieve, the existing sliding-window/object-prefix logic starts from
the first object that is actually present, and `skip_first_n_tokens` is applied
by the transfer helper. Block IDs are therefore staged with the existing
slice-only behavior; a null ID may be present in the staged tensor, but it is
outside every launched kernel's block-ID range and is never dereferenced.

With an unallocated sentinel such as `-1`, the incomplete-object mask prevents
the store copy from dereferencing an absent slice. ATOM's native STATE image
is indivisible: every STATE ordinal is absent in the same prefix chunks and
present at the same checkpoint endpoint. An object containing independently
sparse kernel groups is stored only where every retained slice is present;
separation avoids discarding complete objects because another group is missing
data.

## Testing

- `tests/v1/multiprocess/test_config.py` covers the default and CLI override.
- `tests/v1/multiprocess/test_lmcache_driven_transfer_skip.py` covers global
  null masks and reuse of the existing sparse-copy skip path.
- `tests/v1/multiprocess/test_native_state_lookup.py` covers sparse PAGE/STATE
  lookup with explicit object-group separation.
- `tests/v1/multiprocess/test_native_state_alias_gpu.py` covers a real
  two-process device transfer and is skipped when no GPU is available.
