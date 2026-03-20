# SPDX-License-Identifier: Apache-2.0
"""CLI subcommand package.

To add a new command:

1. Create a module with a :class:`BaseCommand` subclass.
2. Add one import + one entry to :data:`ALL_COMMANDS` below.
"""

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.commands.mock import MockCommand

ALL_COMMANDS: list[BaseCommand] = [
    MockCommand(),
]

__all__ = ["ALL_COMMANDS", "BaseCommand"]
