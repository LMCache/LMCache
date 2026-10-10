# SPDX-License-Identifier: Apache-2.0
"""Abstract base class and shared helpers for CLI subcommands."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
import abc
import argparse
import sys

# First Party
from lmcache.cli.metrics import (
    FileHandler,
    Metrics,
    StreamHandler,
    get_formatter,
)
from lmcache.logging import init_logger

logger = init_logger(__name__)


def _route_head(tokens: Sequence[str]) -> tuple[str | None, list[str]]:
    """Split the current level's route head off the remaining tokens.

    Args:
        tokens: Invocation tokens remaining for this level.

    Returns:
        ``(head, rest)``, or ``(None, [])`` when *tokens* is empty or
        starts with an option token — selection stops there.
    """
    if not tokens:
        return None, []
    token = tokens[0]
    if token.startswith("-"):
        return None, []
    return token, list(tokens[1:])


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
        _add_output_args(parser)
        parser.set_defaults(func=self.execute)

    def create_metrics(
        self,
        title: str,
        args: argparse.Namespace,
        width: int = 48,
    ) -> Metrics:
        """Create a :class:`Metrics` with default handlers pre-registered.

        Handlers are configured from ``args.format``, ``args.output``,
        and ``args.quiet``:

        * A :class:`StreamHandler` writing to stdout, unless ``--quiet``
          is set. The formatter is determined by ``--format``
          (default: ``terminal``).
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

        quiet = getattr(args, "quiet", False)
        fmt_name = getattr(args, "format", None) or "terminal"
        if not quiet:
            metrics.add_handler(StreamHandler(get_formatter(fmt_name, width=width)))

        output = getattr(args, "output", None)
        if output:
            metrics.add_handler(
                FileHandler(output, get_formatter(fmt_name, width=width))
            )

        return metrics

    def register_path(
        self,
        subparsers: argparse._SubParsersAction,
        route: Sequence[str],
    ) -> None:
        """Directed registration of a leaf: full :meth:`register` when this
        command heads *route*, name/help summary otherwise.

        Args:
            subparsers: The subparsers action of the parent parser.
            route: Invocation tokens for this level (see :func:`_route_head`).
        """
        head, _ = _route_head(route)
        if head == self.name():
            self.register(subparsers)
        else:
            self._register_summary(subparsers)

    def _register_summary(self, subparsers: argparse._SubParsersAction) -> None:
        """Bind only this command's name and short help: complete choice
        listings, no ``add_arguments``, no ``func``, no deferred imports.
        """
        subparsers.add_parser(self.name(), help=self.help())


class CompositeCommand(BaseCommand):
    """Base class for commands that contain auto-discovered sub-subcommands.

    Subclasses only need to implement :meth:`name` and :meth:`help`.
    Sub-subcommands are discovered automatically by scanning the package
    for concrete :class:`BaseCommand` subclasses.

    Example::

        class QueryCommand(CompositeCommand):
            def name(self) -> str:
                return "query"

            def help(self) -> str:
                return "Run one inference request and report metrics."
    """

    def add_arguments(self, _parser: argparse.ArgumentParser) -> None:
        """No top-level arguments; all args are registered by subcommands."""

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        """Register this command and auto-discover all sub-subcommands.

        Scans the package where this class is defined for concrete
        :class:`BaseCommand` subclasses and registers each one fully as a
        nested subcommand.

        Args:
            subparsers: The subparsers action from the root parser.
        """
        inner = self._create_parser(subparsers)
        self._subcmds = self._discover_subcommands()
        for inst in self._subcmds.values():
            inst.register(inner)

    def execute(self, args: argparse.Namespace) -> None:
        """Dispatch to the appropriate sub-subcommand.

        Args:
            args: Parsed CLI arguments.
        """
        target = getattr(args, f"{self.name()}_target", None)
        subcmd = self._subcmds.get(target) if target else None
        if subcmd is None:
            print(
                f"Unknown {self.name()} target: {target}",
                file=sys.stderr,
            )
            sys.exit(1)
        subcmd.execute(args)

    def register_path(
        self,
        subparsers: argparse._SubParsersAction,
        route: Sequence[str],
    ) -> None:
        """Directed registration of a group: descend only along *route*;
        unselected groups are summarized without discovering children.

        Args:
            subparsers: The subparsers action of the parent parser.
            route: Invocation tokens for this level (see :func:`_route_head`).
        """
        head, rest = _route_head(route)
        if head != self.name():
            self._register_summary(subparsers)
            return
        inner = self._create_parser(subparsers)
        self._subcmds = self._discover_subcommands()
        for inst in self._subcmds.values():
            inst.register_path(inner, rest)

    def _create_parser(
        self, subparsers: argparse._SubParsersAction
    ) -> argparse._SubParsersAction:
        """Build this group's parser; return its children's subparsers action."""
        parser = subparsers.add_parser(
            self.name(),
            help=self.help(),
            description=self.help(),
        )
        return parser.add_subparsers(
            dest=f"{self.name()}_target",
            required=True,
        )

    def _discover_subcommands(self) -> dict[str, BaseCommand]:
        """Return ``name -> instance`` of this group's children, in discovery
        order; imports child wrappers only, never runs ``add_arguments``.
        """
        # Deferred import to avoid circular dependency
        # First Party
        from lmcache.v1.utils.subclass_discovery import discover_subclasses

        package = self.__class__.__module__

        def _raise(module_name: str, exc: Exception) -> None:
            raise exc

        subcmds: dict[str, BaseCommand] = {}
        for cls in discover_subclasses(
            package,
            BaseCommand,  # type: ignore[type-abstract]
            module_filter=lambda name: not name.startswith("_"),
            require_defined_in_module=True,
            on_import_error=_raise,
        ):
            # Skip the composite command class itself
            if cls is self.__class__:
                continue
            inst = cls()
            subcmds[inst.name()] = inst

        logger.debug(
            "CompositeCommand[%s] discovered subcommands: %s",
            self.name(),
            list(subcmds.keys()),
        )
        return subcmds


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    """Add the common ``--format`` and ``--output`` flags.

    Called automatically by :meth:`BaseCommand.register` — subcommands
    do not need to call this themselves.

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
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress stdout output. Exit code only.",
    )
