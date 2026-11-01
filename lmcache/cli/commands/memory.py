# SPDX-License-Identifier: Apache-2.0
"""``lmcache memory`` — launch the memory orchestrator for one shared region."""

# Standard
import argparse
import sys

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.logging import init_logger

logger = init_logger(__name__)


class MemoryCommand(BaseCommand):
    """CLI command that launches the memory orchestrator (gRPC)."""

    def name(self) -> str:
        """Return the subcommand name."""
        return "memory"

    def help(self) -> str:
        """Return the one-line help shown by ``lmcache -h``."""
        return (
            "Launch the memory orchestrator: the allocation and readiness "
            "authority for one shared memory region."
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add the orchestrator flags, shared with
        ``python -m lmcache.v1.memory_orchestrator``.

        Skips registration when the orchestrator dependencies (gRPC and its
        generated bindings) are not installed; ``execute`` then prints an
        actionable error.

        Args:
            parser: The ``ArgumentParser`` for this subcommand.
        """
        try:
            # First Party
            from lmcache.v1.memory_orchestrator.server import add_orchestrator_args
        except ImportError as e:
            logger.warning(
                "Memory orchestrator dependencies are missing (%s); install the "
                "full lmcache package to use 'memory'.",
                e,
            )
            return
        add_orchestrator_args(parser)

    def execute(self, args: argparse.Namespace) -> None:
        """Build the orchestrator config and serve until SIGTERM or SIGINT.

        Args:
            args: Parsed CLI arguments.

        Raises:
            SystemExit: With the orchestrator's exit code (2 when another
                orchestrator may own the region), or 1 when the orchestrator
                dependencies are not installed.
            ValueError: When the capacity is smaller than the alignment.
        """
        try:
            # First Party
            from lmcache.v1.memory_orchestrator.server import (
                parse_args_to_orchestrator_config,
                serve,
            )
        except ImportError:
            print(
                "The 'lmcache memory' command requires the full lmcache "
                "installation.\nInstall with: pip install lmcache",
                file=sys.stderr,
            )
            sys.exit(1)

        sys.exit(serve(parse_args_to_orchestrator_config(args)))
