# SPDX-License-Identifier: Apache-2.0
"""``lmcache kvcache`` command — KV cache state management.

Sub-subcommands are auto-discovered from modules in this package.
To add a new subcommand, create a module defining a concrete
:class:`BaseCommand` subclass — no edits to this file are required.
"""

# Standard
from collections.abc import Sequence
from functools import partial
import argparse

# First Party
from lmcache.cli.commands.base import CompositeCommand, add_output_args


class KVCacheCommand(CompositeCommand):
    """CLI command for KV cache state management on the LMCache server."""

    def name(self) -> str:
        return "kvcache"

    def help(self) -> str:
        return "Manage KV cache state."

    def register_path(
        self, subparsers: argparse._SubParsersAction, route: Sequence[str]
    ) -> None:
        """Accept the original group-level output flags for ``clear`` scripts.

        Args:
            subparsers: Parent parser's subcommands.
            route: Invocation tokens used for directed registration.
        """
        if (
            len(route) < 2
            or route[0] != self.name()
            or not route[1].startswith("-")
            or route[1] in ("-h", "--help")
        ):
            super().register_path(subparsers, route)
            return

        group = subparsers.add_parser(self.name(), help=self.help())
        add_output_args(group)
        # A leaf option overrides the group value only when explicitly supplied.
        children = group.add_subparsers(
            required=True,
            parser_class=partial(
                argparse.ArgumentParser, argument_default=argparse.SUPPRESS
            ),
        )
        self._discover_subcommands()["clear"].register(children)
