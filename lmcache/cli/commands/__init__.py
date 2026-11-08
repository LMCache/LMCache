# SPDX-License-Identifier: Apache-2.0
"""CLI subcommand package.

Importing this package imports no concrete command wrapper modules;
``lmcache.cli.main.main(argv)`` directs registration along the
invocation route via :func:`register_commands`; semantics and
limitations: section 1 of ``docs/design/cli/framework-and-metrics.md``.

To add a new top-level command, simply create a new module (or sub-package
with an ``__init__.py``) under this package that defines a concrete
:class:`BaseCommand` subclass.  It will be discovered and registered
automatically — no edits to this file are required.
"""

# Standard
from collections.abc import Sequence
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand, CompositeCommand
from lmcache.v1.utils.subclass_discovery import discover_subclasses

ALL_COMMANDS: list[BaseCommand]


def _discover_commands() -> list[BaseCommand]:
    """Scan direct submodules of this package and collect all concrete
    :class:`BaseCommand` subclasses, returning one instance per class.

    Import errors are intentionally re-raised: a broken CLI command
    module should fail loudly rather than silently disappear from the
    CLI.
    """

    def _raise(module_name: str, exc: Exception) -> None:
        raise exc

    return [
        cls()
        for cls in discover_subclasses(
            __name__,
            BaseCommand,  # type: ignore[type-abstract]
            module_filter=lambda name: name != "base",
            on_import_error=_raise,
        )
    ]


def register_commands(
    subparsers: argparse._SubParsersAction,
    route: Sequence[str] = (),
) -> None:
    """Directed registration: *route* names the one fully-registered path;
    every other command binds a name/help summary.

    Args:
        subparsers: The root parser's subparsers action.
        route: Invocation tokens after the program name; ambient
            ``sys.argv`` is not read.
    """
    for cmd in _discover_commands():
        cmd.register_path(subparsers, route)


def __getattr__(name: str) -> list[BaseCommand]:
    """Resolve the transitional lazy ``ALL_COMMANDS`` shim (PEP 562).

    Args:
        name: Requested module attribute name.

    Returns:
        The full command registry, cached after the first access.

    Raises:
        AttributeError: If *name* is not a known module attribute.
    """
    if name == "ALL_COMMANDS":
        commands = _discover_commands()
        globals()["ALL_COMMANDS"] = commands
        return commands
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ALL_COMMANDS", "BaseCommand", "CompositeCommand"]
