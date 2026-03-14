# SPDX-License-Identifier: Apache-2.0
"""Abstract base class and shared helpers for CLI subcommands."""

# Standard
import abc
import argparse


class BaseCommand(abc.ABC):
    """Abstract base class that all CLI subcommands must inherit from.

    Subclasses must implement :meth:`name`, :meth:`help`,
    :meth:`add_arguments`, and :meth:`handler`.  The :meth:`register`
    method wires everything together automatically.

    Example::

        class PingCommand(BaseCommand):
            def name(self) -> str:
                return "ping"

            def help(self) -> str:
                return "Ping the KV cache server."

            def add_arguments(self, parser: argparse.ArgumentParser) -> None:
                parser.add_argument("--url", required=True)

            def handler(self, args: argparse.Namespace) -> None:
                ...
    """

    @abc.abstractmethod
    def name(self) -> str:
        """Return the subcommand name (e.g. ``"mock"``)."""

    @abc.abstractmethod
    def help(self) -> str:
        """Return short help text shown by ``lmcache -h``."""

    @abc.abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add command-specific arguments to *parser*.

        Args:
            parser: The ``ArgumentParser`` for this subcommand.
        """

    @abc.abstractmethod
    def handler(self, args: argparse.Namespace) -> None:
        """Execute the subcommand.

        Args:
            args: Parsed CLI arguments.
        """

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        """Register this command with the CLI argument parser.

        This method is not typically overridden.  It calls
        :meth:`name`, :meth:`help`, and :meth:`add_arguments`, then
        binds :meth:`handler` as the dispatch target.

        Args:
            subparsers: The subparsers action from the root parser.
        """
        parser = subparsers.add_parser(self.name(), help=self.help())
        self.add_arguments(parser)
        parser.set_defaults(func=self.handler)


def add_output_arg(parser: argparse.ArgumentParser) -> None:
    """Add the common ``--output`` flag for JSON metrics export.

    Args:
        parser: The ``ArgumentParser`` to add the flag to.
    """
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Save metrics to a JSON file at PATH.",
    )
