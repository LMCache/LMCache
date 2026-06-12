# SPDX-License-Identifier: Apache-2.0
"""``lmcache trace info`` — print a summary of a trace file."""

# Future
from __future__ import annotations

# Standard
from collections import Counter
from typing import TYPE_CHECKING
import argparse

if TYPE_CHECKING:
    # First Party
    from lmcache.cli.commands.base import BaseCommand


def register_info_parser(
    subparsers: argparse._SubParsersAction,
    dispatch_func,
) -> argparse.ArgumentParser:
    """Register the ``lmcache trace info`` subcommand parser.

    Args:
        subparsers: The ``trace`` subparsers action.
        dispatch_func: Function to bind via ``set_defaults(func=...)``.

    Returns:
        The created ``ArgumentParser``.
    """
    parser = subparsers.add_parser(
        "info",
        help="Print a summary of a trace file.",
    )
    parser.add_argument(
        "trace_path",
        metavar="FILE",
        help="Path to a .lct trace file.",
    )
    parser.set_defaults(func=dispatch_func)
    return parser


def run_trace_info(cmd: "BaseCommand", args: argparse.Namespace) -> None:
    """Read a trace file and print a one-screen summary.

    Args:
        cmd: The parent command instance (unused but kept for interface
            consistency with other subcommands).
        args: Parsed CLI arguments containing ``trace_path``.
    """
    # Deferred import — guarded by _require_full_install() in the
    # dispatcher before this function is called.
    # First Party
    from lmcache.v1.mp_observability.trace.reader import TraceReader

    with TraceReader(args.trace_path) as r:
        header = r.header
        counts: Counter[str] = Counter()
        max_mono = 0.0
        for record in r.records():
            counts[record.qualname] += 1
            if record.t_mono > max_mono:
                max_mono = record.t_mono

    print(f"Trace file: {args.trace_path}")
    print(f"  level:                {header.level}")
    print(f"  format_version:       {header.format_version}")
    print(f"  trace_schema_version: {header.trace_schema_version}")
    print(f"  duration:             {max_mono:.3f}s")
    print(f"  sm_config_digest:     {header.sm_config_digest or '(none)'}")
    print(f"  total_records:        {sum(counts.values())}")
    if counts:
        print("  ops:")
        for qn in sorted(counts):
            print(f"    {qn}: {counts[qn]}")
    else:
        print("  ops: (none)")
