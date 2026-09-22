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

PAGE and STATE must be separate object groups. Object presence is the lookup
signal, so putting dense PAGE data and sparse STATE data in one object would
make a PAGE hit incorrectly imply that STATE is present too. ATOM therefore
uses `--separate-object-groups` explicitly.

## Store and retrieve behavior

`all_null_chunk_masks` compares every block ID against the configured global
sentinel. A chunk is skipped for an object group only when all block IDs in all
of that object's kernel groups equal the sentinel. Skipped objects are neither
reserved nor committed.

LMCache already skips these copies in the transfer layer. On store, skipped
objects become `None` entries and the D2H loop does not launch a kernel for
them. On retrieve, the existing sliding-window/object-prefix logic starts from
the first object that is actually present, and `skip_first_n_tokens` is applied
by the transfer helper. Block IDs are therefore staged with the existing
slice-only behavior; a null ID may be present in the staged tensor, but it is
outside every launched kernel's block-ID range and is never dereferenced.

For a non-default sentinel, all block IDs represented by one object and chunk
must agree on presence. If an object mixes the null marker with live block IDs,
the server fails that store before staging or reserving memory. This protects
the transfer kernel from an invalid index while preserving the legacy vLLM
zero-null behavior. Supporting independently sparse kernel groups inside one
object requires a separate object-group contract or per-kernel transfer masks.

## Testing

- `tests/v1/multiprocess/test_config.py` covers the default and CLI override.
- `tests/v1/multiprocess/test_lmcache_driven_transfer_skip.py` covers global
  null masks and reuse of the existing sparse-copy skip path.
- `tests/v1/multiprocess/test_native_state_lookup.py` covers sparse PAGE/STATE
  lookup with explicit object-group separation.
- `tests/v1/multiprocess/test_native_state_alias_gpu.py` covers a real
  two-process device transfer and is skipped when no GPU is available.
