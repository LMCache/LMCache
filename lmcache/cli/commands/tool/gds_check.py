# SPDX-License-Identifier: Apache-2.0
"""``lmcache tool gds-check`` sub-subcommand wiring.

Implementation lives in :mod:`lmcache.tools.gds_check`. This file
only defines flags and dispatches to it. Imports of heavy deps
(torch, kvikio) are deferred to execution time so ``lmcache -h``
stays fast even when GDS deps aren't installed.
"""

# Standard
import argparse
import os


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register ``lmcache tool gds-check``.

    Args:
        subparsers: The subparsers action from the ``lmcache tool``
            parser.
    """
    parser = subparsers.add_parser(
        "gds-check",
        help="Probe + benchmark the GDS L1 backend on this host.",
        description=(
            "Inspect host readiness for the GDS L1 backend (fstype, "
            "nvidia-fs, kvikio compat mode, cuFile alignment), verify "
            "a byte-for-byte round-trip through the same code path the "
            "MP server uses, and report store/retrieve throughput. Use "
            "this to compare hardware before enabling GDS L1 in "
            "production."
        ),
    )
    parser.add_argument(
        "--gds-path",
        default=os.path.expanduser("~/.lmcache_gds_check"),
        help="Directory the test will use as the GDS L1 disk root. "
        "Wiped at the start of verify/bench. Default: "
        "~/.lmcache_gds_check.",
    )
    parser.add_argument(
        "--small-num-chunks",
        type=int,
        default=64,
        help="Number of small chunks. Phase 1 of the bench measures "
        "per-call overhead, so the default is many small chunks. "
        "Default 64.",
    )
    parser.add_argument(
        "--small-chunk-mib",
        type=int,
        default=2,
        help="Per-chunk size in MiB for the small-chunks phase. "
        "Default 2 MiB.",
    )
    parser.add_argument(
        "--large-num-chunks",
        type=int,
        default=8,
        help="Number of large chunks. Phase 2 of the bench measures "
        "sustained bandwidth, so the default is few large chunks. "
        "Default 8.",
    )
    parser.add_argument(
        "--large-chunk-mib",
        type=int,
        default=256,
        help="Per-chunk size in MiB for the large-chunks phase. "
        "Default 256 MiB. Crank to 2048 (= 2 GiB) for a hard "
        "throughput ceiling test — note that uses "
        "large-chunk-mib × 1 of VRAM scratch.",
    )
    parser.add_argument(
        "--no-gds",
        action="store_true",
        help="Force the POSIX (mmap + cudaMemcpy) fallback path. Use "
        "this to compare the fallback against the kvikio cuFile path "
        "on the same host.",
    )
    parser.add_argument(
        "--use-direct-io",
        action="store_true",
        help="Open the test files with O_DIRECT. Required for true GDS "
        "DMA on ext4 — without it, libcufile routes the I/O through "
        "its compat path even when nvidia-fs is loaded. Combine this "
        "with the ``Ops: Read/Write`` counters in "
        "``/proc/driver/nvidia-fs/stats`` to confirm DMA is actually "
        "firing.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the round-trip correctness check.",
    )
    parser.add_argument(
        "--skip-bench",
        action="store_true",
        help="Skip the throughput benchmark. Useful for a quick host-info-only run.",
    )
    parser.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    """Dispatch into the prober implementation."""
    # First Party
    # Lazy import — keeps `lmcache -h` fast on hosts without torch/kvikio.
    from lmcache.tools.gds_check.prober import run_gds_check

    run_gds_check(
        gds_path=args.gds_path,
        small_num_chunks=args.small_num_chunks,
        small_chunk_bytes=args.small_chunk_mib * 1024 * 1024,
        large_num_chunks=args.large_num_chunks,
        large_chunk_bytes=args.large_chunk_mib * 1024 * 1024,
        use_gds=not args.no_gds,
        use_direct_io=args.use_direct_io,
        skip_verify=args.skip_verify,
        skip_bench=args.skip_bench,
    )
