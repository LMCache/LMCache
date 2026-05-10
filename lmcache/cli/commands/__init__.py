# SPDX-License-Identifier: Apache-2.0
"""CLI subcommand package.

To add a new top-level command, simply create a new module (or sub-package
with an ``__init__.py``) under this package that defines a concrete
:class:`BaseCommand` subclass.  It will be discovered and registered
automatically — no edits to this file are required.
"""

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.v1.utils.subclass_discovery import discover_subclasses


def _discover_commands() -> list[BaseCommand]:
    """Scan direct submodules of this package and collect all concrete
    :class:`BaseCommand` subclasses, returning one instance per class.
    """
    return [
        cls()
        for cls in discover_subclasses(
            __name__,
            BaseCommand,  # type: ignore[type-abstract]
            module_filter=lambda name: name != "base",
        )
    ]


ALL_COMMANDS: list[BaseCommand] = _discover_commands()

__all__ = ["ALL_COMMANDS", "BaseCommand"]
