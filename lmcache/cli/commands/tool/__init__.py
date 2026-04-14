# SPDX-License-Identifier: Apache-2.0
"""``lmcache tool`` command — offline analysis utilities.

To add a new tool:

1. Create ``lmcache/cli/commands/tool/<your_tool>.py`` with a ``register``
   function that adds a sub-subcommand to the supplied ``subparsers``.
2. Import it here and call ``your_tool.register(inner)`` inside
   :meth:`ToolCommand.register`.
"""

# Standard
import argparse
import sys

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.commands.tool import cache_simulator


class ToolCommand(BaseCommand):
    """CLI command for offline analysis tools bundled with LMCache."""

    def name(self) -> str:
        """Return the subcommand name."""
        return "tool"

    def help(self) -> str:
        """Return short help text shown by ``lmcache -h``."""
        return "Run offline analysis tools."

    def add_arguments(self, _parser: argparse.ArgumentParser) -> None:
        """No top-level arguments; all args are registered in register()."""

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        """Register ``lmcache tool`` and all tool sub-subcommands.

        Args:
            subparsers: The subparsers action from the root parser.
        """
        parser = subparsers.add_parser(
            self.name(),
            help=self.help(),
            description="Run offline analysis tools bundled with LMCache.",
        )
        inner = parser.add_subparsers(
            dest="tool_name",
            required=True,
            metavar="{cache-simulator}",
        )
        cache_simulator.register(inner)

    def execute(self, args: argparse.Namespace) -> None:
        """Dispatch is handled per-tool via parser.set_defaults(func=...).

        This method is never called directly; each tool's register() binds
        its own execute function as the dispatch target.

        Args:
            args: Parsed CLI arguments.
        """
        print(f"Unknown tool: {getattr(args, 'tool_name', '?')}", file=sys.stderr)
        sys.exit(1)
