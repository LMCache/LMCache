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

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        """Register with slim-install stub fallback.

        On a slim install (missing torch/zmq), registers a stub parser
        with a helpful message instead of the full argument set.
        """
        # First Party
        from lmcache.cli.commands.bench.server_bench.helpers import (
            _IMPORT_ERROR,
        )

        if _IMPORT_ERROR is not None:
            # Slim install — register a stub parser only.
            stub = subparsers.add_parser(
                self.name(),
                help="(requires full lmcache install)",
                description=(
                    "End-to-end sanity test for the LMCache MP cache server. "
                    "Requires the full `lmcache` package; not available in "
                    "the `lmcache-cli` install."
                ),
            )
            stub.set_defaults(func=self.execute)
            return

        # Full install — use standard BaseCommand.register() flow
        super().register(subparsers)

    def execute(self, args: argparse.Namespace) -> None:
        # First Party
        from lmcache.cli.commands.bench.server_bench.command import (
            run_server_bench,
        )

        run_server_bench(self, args)
