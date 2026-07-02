# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench server`` subpackage.

Exposes :class:`ServerBenchCommand` for auto-discovery by
:class:`~lmcache.cli.commands.base.CompositeCommand`.
"""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand


class ServerBenchCommand(BaseCommand):
    """End-to-end test for LMCache MP cache server."""

    def name(self) -> str:
        return "server"

    def help(self) -> str:
        return "End-to-end test for LMCache MP cache server (GPU mode)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # First Party
        from lmcache.cli.commands.bench.server_bench.command import (
            add_server_arguments,
        )

        add_server_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        # First Party
        from lmcache.cli.commands.bench.server_bench.command import (
            run_server_bench,
        )

        run_server_bench(self, args)
