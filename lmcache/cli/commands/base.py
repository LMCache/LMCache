# SPDX-License-Identifier: Apache-2.0
"""Abstract base class and shared helpers for CLI subcommands."""

# Standard
import abc
import argparse

# First Party
from lmcache.cli.metrics import (
    FileHandler,
    Metrics,
    StreamHandler,
    TerminalFormatter,
    get_formatter,
)


class BaseCommand(abc.ABC):
    """Abstract base class that all CLI subcommands must inherit from.

    Subclasses must implement :meth:`name`, :meth:`help`,
    :meth:`add_arguments`, and :meth:`execute`.  The :meth:`register`
    method wires everything together automatically.

    Example::

        class PingCommand(BaseCommand):
            def name(self) -> str:
                return "ping"

            def help(self) -> str:
                return "Ping the KV cache server."

            def add_arguments(self, parser: argparse.ArgumentParser) -> None:
                parser.add_argument("--url", required=True)
                add_output_args(parser)

            def execute(self, args: argparse.Namespace) -> None:
                metrics = self.create_metrics("Ping Result", args)
                metrics.add("status", "Status", "OK")
                metrics.emit()
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
    def execute(self, args: argparse.Namespace) -> None:
        """Execute the subcommand.

        Called by the CLI dispatcher in ``main.py`` via
        ``args.func(args)`` after argument parsing.  The
        :meth:`register` method binds this as the dispatch target
        using ``parser.set_defaults(func=self.execute)``.

        Args:
            args: Parsed CLI arguments.
        """

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        """Register this command with the CLI argument parser.

        This method is not typically overridden.  It calls
        :meth:`name`, :meth:`help`, and :meth:`add_arguments`, then
        binds :meth:`execute` as the dispatch target.

        Args:
            subparsers: The subparsers action from the root parser.
        """
        parser = subparsers.add_parser(self.name(), help=self.help())
        self.add_arguments(parser)
        parser.set_defaults(func=self.execute)

    def create_metrics(
        self,
        title: str,
        args: argparse.Namespace,
        width: int = 48,
    ) -> Metrics:
        """Create a :class:`Metrics` with default handlers pre-registered.

        Handlers are configured from ``args.format`` and ``args.output``:

        * A :class:`StreamHandler` writing to stdout. The formatter is
          determined by ``--format`` (default: ``terminal``).
        * A :class:`FileHandler` if ``--output`` is set (uses the same
          formatter chosen by ``--format``).

        Args:
            title: Report title.
            args: Parsed CLI arguments (inspects ``format`` and ``output``).
            width: Character width for terminal rendering (only used by
                formatters that support it, e.g. ``TerminalFormatter``).

        Returns:
            A ready-to-use ``Metrics`` instance.
        """
        metrics = Metrics(title=title)

        # Resolve stdout formatter from --format
        fmt_name = getattr(args, "format", None) or "terminal"
        stdout_formatter = get_formatter(fmt_name)
        if isinstance(stdout_formatter, TerminalFormatter):
            stdout_formatter = TerminalFormatter(width=width)
        metrics.add_handler(StreamHandler(stdout_formatter))

        # File handler if --output is set
        output = getattr(args, "output", None)
        if output:
            file_formatter = get_formatter(fmt_name)
            if isinstance(file_formatter, TerminalFormatter):
                file_formatter = TerminalFormatter(width=width)
            metrics.add_handler(FileHandler(output, file_formatter))

        return metrics


def add_output_args(parser: argparse.ArgumentParser) -> None:
    """Add the common ``--format`` and ``--output`` flags.

    Args:
        parser: The ``ArgumentParser`` to add the flags to.
    """
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        metavar="FORMAT",
        help=("Stdout output format (default: terminal). Available: terminal, json."),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Save metrics to a file at PATH (format chosen by --format).",
    )
