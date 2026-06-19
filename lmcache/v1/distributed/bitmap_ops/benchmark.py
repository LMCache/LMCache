# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark for the ranked fold (:func:`fold_unfold_ranked`).

Compares the pure-Python reference against the vectorized torch path across
request sizes, including a DeepSeek-scale hybrid case (1M tokens, 8 object
groups mixing full attention and sliding window). Run with::

    python -m lmcache.v1.distributed.bitmap_ops.benchmark

Findings it surfaces:

* The torch *compute* is ~constant (sub-millisecond) regardless of size; the
  pure-Python scan is linear in ``num_groups * num_chunks * num_ranks``.
* End-to-end, the torch path is dominated by the ``Bitmap``<->tensor conversion
  (``get_indices_list`` + ``batched_set``); a native dense ``Bitmap`` export
  would remove that and recover the compute-only numbers.
* Below a few thousand keys the Python path wins (torch's per-call tensor
  overhead), which is why :func:`fold_unfold_ranked` dispatches by size.
"""

# Standard
from collections.abc import Sequence
import time

# Third Party
import torch

# First Party
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.bitmap_ops.fold import (
    _fold_unfold_ranked_python,
    _fold_unfold_ranked_torch,
)


def _best_ms(fn, reps: int) -> float:
    """Best wall-clock time of ``fn`` over ``reps`` runs, in milliseconds."""
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best * 1e3


def bench_case(
    label: str,
    num_chunks: int,
    num_ranks: int,
    group_windows: Sequence[int],
    present_fraction: float,
    reps: int = 5,
) -> None:
    """Print Python vs torch timings for one (size, fill) configuration."""
    num_keys = len(group_windows) * num_chunks * num_ranks
    if present_fraction >= 1.0:
        found = Bitmap(num_keys, num_keys)
    else:
        found = Bitmap(num_keys, int(num_keys * present_fraction))

    py_ms = _best_ms(
        lambda: _fold_unfold_ranked_python(
            found, num_chunks, num_ranks, group_windows
        ),
        reps,
    )
    torch_ms = _best_ms(
        lambda: _fold_unfold_ranked_torch(found, num_chunks, num_ranks, group_windows),
        reps,
    )
    speedup = py_ms / torch_ms if torch_ms else float("inf")
    print(
        f"{label:<34}keys={num_keys:>9}  python={py_ms:>9.2f}ms  "
        f"torch={torch_ms:>8.2f}ms  speedup={speedup:>6.1f}x"
    )


def main() -> None:
    """Run the benchmark grid."""
    torch.set_num_threads(8)
    # 8 groups, mix of full attention and sliding window (DeepSeek-like hybrid).
    dpsk_windows = (-1, -1, -1, -1, 4, 4, 8, 1)

    print("== DeepSeek 1M tokens @ chunk_size=256 (num_chunks=4096), all present ==")
    bench_case("dpsk, world_size=1", 4096, 1, dpsk_windows, 1.0)
    bench_case("dpsk, world_size=8", 4096, 8, dpsk_windows, 1.0)

    print("\n== same, 50% prefix present (realistic) ==")
    bench_case("dpsk, world_size=8", 4096, 8, dpsk_windows, 0.5)

    print("\n== small request (should favor python; dispatch picks it) ==")
    bench_case("4K tokens @256 (16 chunks)", 16, 8, dpsk_windows, 1.0)

    print("\n== stress: chunk_size=16 -> 62500 chunks (4M keys) ==")
    bench_case("stress, world_size=8", 62500, 8, dpsk_windows, 1.0)


if __name__ == "__main__":
    main()
