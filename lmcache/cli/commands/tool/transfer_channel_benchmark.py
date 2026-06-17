# SPDX-License-Identifier: Apache-2.0
"""``lmcache tool transfer-channel-benchmark`` sub-subcommand wiring.

Argument definitions and execution logic live in the benchmark module:

* :func:`~lmcache.tools.transfer_channel_benchmark.benchmark.add_benchmark_arguments`
* :func:`~lmcache.tools.transfer_channel_benchmark.benchmark.run_benchmark`
"""

# Standard
import argparse
import sys


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``transfer-channel-benchmark`` sub-subcommand.

    Only the torch-free ``config`` module is imported here, so registering this
    tool (which happens on every ``lmcache`` invocation) does not require torch
    or the distributed runtime. Those are imported in ``execute``.

    Args:
        subparsers: The subparsers action from the ``lmcache tool`` parser.
    """
    # First Party
    from lmcache.tools.transfer_channel_benchmark.config import (
        add_benchmark_arguments,
    )

    parser = subparsers.add_parser(
        "transfer-channel-benchmark",
        help="Benchmark transfer channel read throughput (server/client).",
        description=(
            "Throughput benchmark for the LMCache transfer channel. Run one "
            "process with --role server and another with --role client."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_benchmark_arguments(parser)
    parser.set_defaults(func=execute)


def execute(args: argparse.Namespace) -> None:
    """Run the benchmark and exit non-zero on failure.

    Args:
        args: Parsed CLI arguments.
    """
    # First Party
    from lmcache.tools.transfer_channel_benchmark.benchmark import run_benchmark

    if not run_benchmark(args):
        sys.exit(1)
