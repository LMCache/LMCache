# SPDX-License-Identifier: Apache-2.0
"""``python -m lmcache.tools.transfer_channel_benchmark`` entry point."""

# Standard
import argparse
import sys

# First Party
from lmcache.tools.transfer_channel_benchmark.benchmark import (
    add_benchmark_arguments,
    run_benchmark,
)


def main() -> int:
    """Parse arguments and run the transfer channel benchmark.

    Returns:
        Process exit code (0 on success, 1 on failure).
    """
    parser = argparse.ArgumentParser(
        prog="python -m lmcache.tools.transfer_channel_benchmark",
        description="Throughput benchmark for the LMCache transfer channel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_benchmark_arguments(parser)
    args = parser.parse_args()
    return 0 if run_benchmark(args) else 1


if __name__ == "__main__":
    sys.exit(main())
