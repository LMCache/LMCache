# `bitmap_ops`: cross-object-group prefix-cache hit computation

## Problem

A hybrid model splits one request across several **object groups** (full
attention, sliding window, mamba, ...). Each group is stored as its own
`MemoryObj` and has its own token-dependency rule:

- **full attention** can serve a prefix of length `L` only if *every* chunk
  `[0, L)` is present;
- a **sliding window of `w` chunks** can serve length `L` if the last
  `min(w, L)` chunks are present (mamba is the `w == 1` case).

A model-wide prefix-cache hit of length `L` is valid only if **all** groups can
serve `L` under their own rule. The presence of each chunk is known per group
(from L1 residency and the L2 lookup), so the job is to turn the per-group
presence bitmaps into one hit length plus the concrete chunks to keep.

## Design: fold → right-most-1 → unfold

1. **fold** (`fold`) — per group, compute the prefix lengths it can serve and
   intersect across groups into a servable-lengths bitmap. A single pass tracks
   the run of consecutive present chunks, so the step is
   `O(num_groups * num_chunks)`.
2. **right-most-1** (`find_rightmost_one`) — the highest set bit of the servable
   bitmap is the hit length `i*` (a length is a hit only if every group can
   serve it).
3. **unfold** (`unfold`) — expand `i*` back into the chunks each group must
   retain (`[0, i*)` for full attention, `[i*-w, i*)` for a window; see
   `unfold_range`). The union over groups is the **retain mask** used to load,
   lock, and transfer.

When every group is full attention this collapses to the leading-ones count of
the AND of the per-group presences — i.e. plain longest-contiguous-prefix
matching. The fold is a strict generalization of that, so non-hybrid models keep
their existing behavior.

`fold_unfold_ranked` is the same pipeline over the `group x chunk x kv_rank`
lookup key layout: a chunk counts as present only when **all** kv_rank shards are
present, and the retain mask re-expands over every shard. This replaces the old
`// world_size` arithmetic with an explicit rank reduction.

## Where it runs, and why

LMCache prefetch is a two-stage **predict-then-execute** path: lookup (presence)
→ plan → load → finalize (reconcile rare failures). The fold lives in
`PrefetchPolicy.compute_retained` and is driven by the prefetch controller, not
inside `ObjectKey` or the L2 adapters, because:

- The decision is inherently **cross-group**: it needs the presence bitmaps of
  all groups at once, which only the controller has assembled.
- The controller folds over the **merged L1 ∪ L2 presence**, reports the full
  retained set (including chunks already resident in L1), and loads only the
  retained chunks that L2 has and L1 does not. On a reservation or load failure
  it **re-folds**, so a missing chunk shrinks the prefix consistently with the
  rule instead of leaving a hole.
- The controller therefore **owns L1 read-lock release** on every exit path:
  read-locks taken at submit time for chunks the fold did not keep are released
  by the controller, matching how it already releases L2 lookup locks.

## Window plumbing

Sliding-window sizes (`w`, in chunks) are **not** embedded in `ObjectKey`. They
are published per object group through the layout registry at `REGISTER` time
and carried out-of-band on the prefetch request (`group_windows`). This keeps
`ObjectKey` stable and lets the manager-blind lookup path run the fold. `-1` (or
any value `<= 0`) marks a full-attention group.

## Public API

The pipeline is exposed as **three composable operators** so the selection
logic can evolve without rewriting the primitives:

| Operator | Purpose |
|---|---|
| `fold` | Presence (`group x chunk x kv_rank`) → servable-prefix-lengths bitmap (bit `L` = every group can serve length `L`). |
| `find_rightmost_one` | Highest set bit of a bitmap — applied to `fold`'s output, the model-wide hit length. |
| `unfold` | Hit length → per-group retain mask over the ranked layout. |

Convenience / supporting:

| Function | Purpose |
|---|---|
| `fold_unfold_ranked` | Composes `fold` → `find_rightmost_one` → `unfold`. |
| `fold_unfold` | `fold_unfold_ranked` for the single-rank (`group x chunk`) layout. |
| `unfold_range` | Chunk range one group needs for a given hit length. |
| `merge_bitmaps` | Bitwise-OR several presence bitmaps (e.g. L1 ∪ L2). |
| `select_retained` | Non-windowed `TrimPolicy` selection. |

`TrimPolicy.PREFIX` selects the single longest servable prefix (right-most 1);
any other policy keeps every set bit (gaps included).

## Performance

`fold` and `unfold` delegate to **native C++** (`csrc/storage_manager/fold.cpp`,
exported as `native_storage_ops.fold` / `unfold`) and `find_rightmost_one` to
`Bitmap.find_rightmost_one()`. They scan the packed `Bitmap` buffer directly —
no Python per-bit loop and no `Bitmap`↔tensor conversion. Pure-Python fallbacks
(`_fold_python` / `_unfold_python`) are used only if the extension lacks the
ops, and serve as the equivalence oracle in tests. See `benchmark.py`
(`python -m lmcache.v1.distributed.bitmap_ops.benchmark`):

| Case (full pipeline) | Python | native | speedup |
|---|---|---|---|
| DeepSeek 1M @256, 8 groups, world_size=8 (262k keys), all present | ~158 ms | ~0.6 ms | ~260× |
| same, 50% prefix present (realistic) | ~75 ms | ~0.35 ms | ~215× |
| world_size=1 (32k keys) | ~46 ms | ~0.17 ms | ~275× |
| stress: 4M keys | ~1300 ms | ~5 ms | ~255× |

`unfold` writes the retained keys back as contiguous spans via
`Bitmap::set_range` (whole-byte fills) rather than per-bit sets, so even the
all-present worst case stays sub-millisecond at the DeepSeek scale. The
remaining cost is the presence scan in `fold`; a word-level rank-reduction
(all-ranks test over a contiguous span) is the next lever if it's ever needed.
