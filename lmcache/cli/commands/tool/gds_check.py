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
        "--num-chunks",
        type=int,
        default=256,
        help="Number of chunks to write+read in the bench phase. Default 256.",
    )
    parser.add_argument(
        "--chunk-mib",
        type=int,
        default=2,
        help="Per-chunk size in MiB. Must be a 4 KiB multiple "
        "(any integer MiB works). Default 2.",
    )
    parser.add_argument(
        "--no-gds",
        action="store_true",
        help="Force the POSIX (mmap + cudaMemcpy) fallback path. Use "
        "this to compare the fallback against the kvikio cuFile path "
        "on the same host.",
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
        num_chunks=args.num_chunks,
        chunk_bytes=args.chunk_mib * 1024 * 1024,
        use_gds=not args.no_gds,
        skip_verify=args.skip_verify,
        skip_bench=args.skip_bench,
    )
