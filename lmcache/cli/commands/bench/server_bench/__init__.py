# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench server`` subpackage.

Exposes :class:`ServerBenchCommand` for auto-discovery by
:class:`~lmcache.cli.commands.base.CompositeCommand`.
"""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.logging import init_logger

logger = init_logger(__name__)


class ServerBenchCommand(BaseCommand):
    """End-to-end test for LMCache MP cache server."""

    def name(self) -> str:
        return "server"

    def help(self) -> str:
        return "End-to-end test for LMCache MP cache server (GPU mode)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register benchmark arguments when runtime dependencies are available."""
        try:
            # First Party
            from lmcache.cli.commands.bench.server_bench.command import (
                add_server_arguments,
            )

            add_server_arguments(parser)
        except ImportError as exc:
            logger.warning(
                "Unable to load 'lmcache bench server' arguments (%s); "
                "install the full package with: pip install lmcache",
                exc,
            )

    def execute(self, args: argparse.Namespace) -> None:
        """Load and execute the benchmark, reporting runtime import failures."""
        # Standard
        import copy
        import sys

        try:
            # First Party
            from lmcache.cli.commands.bench.server_bench.command import (
                run_server_bench,
            )
        except ImportError as exc:
            print(
                "ERROR: Unable to load `lmcache bench server`. "
                "This command requires the full LMCache package.\n"
                "  Install with: pip install lmcache\n"
                f"  Original import error: {exc}",
                file=sys.stderr,
            )
            sys.exit(2)

        # A comma-separated --flamegraph-mode drives the load once per mode
        # (the modes cannot share a recording window), each rendered to its
        # own default path.
        modes = [args.flamegraph_mode]
        if getattr(args, "flamegraph", "off") == "on" and "," in args.flamegraph_mode:
            modes = list(
                dict.fromkeys(
                    m.strip() for m in args.flamegraph_mode.split(",") if m.strip()
                )
            )
            if args.flamegraph_output:
                print(
                    "Error: --flamegraph-output takes a single path; with "
                    "several modes each graph uses its own default path.",
                    file=sys.stderr,
                )
                sys.exit(2)

        for mode in modes:
            run_args = copy.copy(args)
            run_args.flamegraph_mode = mode
            run_server_bench(self, run_args)
