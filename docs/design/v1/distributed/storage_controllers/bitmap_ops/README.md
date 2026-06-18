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

1. **fold** — per group, compute the set of prefix lengths it can serve. A
   single backward/forward pass tracks the run of consecutive present chunks, so
   the whole step is `O(num_groups * num_chunks)` (`fold_unfold`).
2. **intersect + right-most-1** — a length is a model-wide hit only if every
   group can serve it; the longest surviving length is the hit length `i*`.
3. **unfold** — expand `i*` back into the chunks each group must retain
   (`unfold_range`): `[0, i*)` for full attention, `[i*-w, i*)` for a window.
   The union over groups is the **retain mask** used to load, lock, and transfer.

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

| Function | Purpose |
|---|---|
| `fold_unfold` | Fold/unfold over a `group x chunk` presence bitmap. |
| `fold_unfold_ranked` | Same, over the `group x chunk x kv_rank` lookup layout. |
| `unfold_range` | Chunk range one group needs for a given hit length. |
| `merge_bitmaps` | Bitwise-OR several presence bitmaps (e.g. L1 ∪ L2). |
| `select_retained` | Non-windowed `TrimPolicy` selection. |

`TrimPolicy.PREFIX` selects the single longest servable prefix (right-most 1);
any other policy keeps every set bit (gaps included).
