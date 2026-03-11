# SPDX-License-Identifier: Apache-2.0

"""
LMCache multiprocess server CLI entry-point.

Usage::

    python -m lmcache.v1.multiprocess.cli <command> [options]

Sub-commands are discovered automatically from the
``commands/`` sub-package.  To add a new command, create a
new module under ``commands/`` and define a module-level
``register_command(subparsers)`` function.  No existing
files need to be modified.
"""

# Standard
import argparse
import importlib
import pkgutil
import sys

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


def _discover_commands(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Walk ``cli.commands`` and call each module's registrar."""
    # First Party
    from lmcache.v1.multiprocess.cli import commands as cmd_pkg

    for finder, name, _ in pkgutil.iter_modules(cmd_pkg.__path__):
        module = importlib.import_module(
            ".%s" % name,
            package=cmd_pkg.__name__,
        )
        registrar = getattr(module, "register_command", None)
        if registrar is None:
            logger.warning(
                "Command module %s has no register_command, skipped",
                name,
            )
            continue
        registrar(subparsers)


def main(argv: list[str] | None = None) -> None:
    """Build the top-level parser and dispatch."""
    parser = argparse.ArgumentParser(
        prog="lmcache-server",
        description="LMCache multiprocess server CLI",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )
    _discover_commands(subparsers)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Each command stores its handler via set_defaults(func=...)
    args.func(args)


if __name__ == "__main__":
    main()
