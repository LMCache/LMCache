# SPDX-License-Identifier: Apache-2.0
"""``lmcache query`` command — single-shot inference request interface.

Subcommands:

* ``engine`` — send one request to an OpenAI-compatible HTTP API
* ``kvcache`` — query KV-cache endpoints (not implemented yet)
"""

# Standard
import argparse
import sys

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.commands.query.engine_command import (
    register_engine_parser,
    run_query_engine,
)
from lmcache.cli.commands.query.kvcache_command import (
    register_kvcache_parser,
    run_query_kvcache,
)


class QueryCommand(BaseCommand):
    """CLI command that sends one request to a serving engine."""

    def name(self) -> str:
        return "query"

    def help(self) -> str:
        return "Run one inference request and report metrics."

    def add_arguments(self, _parser: argparse.ArgumentParser) -> None:
        pass  # args registered in register() via subparsers

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        """Register ``lmcache query`` and all query sub-subcommands.

        Args:
            subparsers: The subparsers action from the root parser.
        """
        parser = subparsers.add_parser(
            self.name(),
            help=self.help(),
            description=(
                "Run one OpenAI-compatible inference request and report metrics."
            ),
        )
        inner = parser.add_subparsers(
            dest="query_target",
            required=True,
            metavar="{engine,kvcache}",
        )
        register_engine_parser(inner, self.execute)
        register_kvcache_parser(inner, self.execute)

    def execute(self, args: argparse.Namespace) -> None:
        """Dispatch to the appropriate query subcommand handler.

        Args:
            args: Parsed CLI arguments containing ``query_target``.
        """
        handlers = {
            "engine": lambda a: run_query_engine(self, a),
            "kvcache": lambda a: run_query_kvcache(self, a),
        }
        handler = handlers.get(args.query_target)
        if handler is None:
            print(
                f"Unknown query target: {args.query_target}",
                file=sys.stderr,
            )
            sys.exit(1)
        handler(args)
