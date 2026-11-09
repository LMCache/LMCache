# `bitmap_ops`

Bitmap operators for computing a **cross-object-group prefix-cache hit**. A
hybrid model splits one request across several object groups (full attention,
sliding window) with different rules: full attention can serve a prefix
of length `L` only if chunks `[0, L)` are present; a sliding window of `w` chunks
needs only the last `min(w, L)`. Given each group's per-chunk presence, these
operators produce the longest length **every** group can serve and the concrete
chunks each group must keep.

## Operators

The pipeline is three composable operators (so the selection logic can evolve
without rewriting the primitives):

| Operator | Purpose |
|---|---|
| `fold` | Presence (`chunk x group x kv_rank`) → servable bitmap (bit `j` set iff every group can serve a length-`j+1` prefix). |
| `highest_set_bit` | Highest set bit of a bitmap, or `-1` if none — on `fold`'s output, the hit length minus one (hit length = result + 1, so `-1` → 0). |
| `unfold` | Hit length → per-group retain mask over the ranked layout. |

Supporting / convenience:

| Function | Purpose |
|---|---|
| `fold_grouped` / `unfold_grouped` / `fold_unfold_grouped` | The same operators over the **grouped** layout: a list of presence bitmaps (one per object, each of length `num_chunks`) paired 1:1 with a list of window sizes. No ordering of the rows is assumed; the caller resolves each row's window (a kv-rank shard is simply another row with its group's window). |
| `unfold_range` | Chunk range one group needs for a given hit length. |
| `merge_bitmaps` | Bitwise-OR several presence bitmaps (e.g. L1 ∪ L2). |

A chunk counts as present for a group only when **all** its `kv_rank` shards are
present, and `unfold` sets all ranks of each retained `(group, chunk)`. With a
single full-attention group the result is plain longest-contiguous-prefix
matching.

## Optional rows

`fold_unfold_grouped(rows, windows, optional)` accepts a per-row `optional`
flag for sliding windows the engine can recompute itself (vLLM bounded prefix
replay, e.g. DeepSeek V4.1). Without it, a window evicted from L1 would cut
the hit to zero even though the global KV is fully present.

- The hit length is folded from the required rows alone.
- The optional rows are retained together or not at all: each keeps its
  window at the hit only if every optional row has its whole window present
  there. A partial window is useless to the engine.
- A full-attention row cannot be optional, and at least one row must be
  required.

Example, 4 chunks, row 0 full attention, row 1 an optional 1-chunk window:

| row 1 present | hit | row 1 retained |
|---|---|---|
| chunk 3 | 4 | chunk 3 |
| chunk 2 only | 4 | nothing (the engine recomputes the window) |

The lookup locks an optional object group only when its rows were retained
(`Session.prefetch_locked_gids` in `lmcache/v1/multiprocess/session.py`), and
the retrieve skips an unlocked optional group.

## Performance

`fold` / `unfold` and their grouped variants delegate to native C++
(`csrc/lmcache_native/fold.cpp`, exported as `lmcache_native.fold` / `unfold` /
`fold_grouped` / `unfold_grouped`) and `highest_set_bit` to
`Bitmap.highest_set_bit()`. They scan the packed `Bitmap` buffer directly —
no Python per-bit loop and no `Bitmap`↔tensor conversion. `_fold_python` /
`_unfold_python` are reference implementations used only as test oracles. See
`benchmarks/microbenchmark/bitmap_ops_benchmark.py`
(`python benchmarks/microbenchmark/bitmap_ops_benchmark.py`):

| Case (full pipeline) | Python | native | speedup |
|---|---|---|---|
| DeepSeek 1M @256, 8 groups, world_size=8 (262k keys), all present | ~158 ms | ~0.6 ms | ~260× |
| same, 50% prefix present (realistic) | ~75 ms | ~0.35 ms | ~215× |
| world_size=1 (32k keys) | ~46 ms | ~0.17 ms | ~275× |
| stress: 4M keys | ~1300 ms | ~5 ms | ~255× |

`unfold` writes the retained keys back as contiguous spans via
`Bitmap::set_range` (whole-byte fills) rather than per-bit sets, so even the
all-present worst case stays sub-millisecond at the DeepSeek scale. The
remaining cost is the presence scan in `fold`.
