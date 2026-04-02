# SPDX-License-Identifier: Apache-2.0
"""CLI subcommand package.

To add a new command:

1. Create a module with a :class:`BaseCommand` subclass.
2. Add one import + one entry to :data:`ALL_COMMANDS` below.
"""

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.commands.bench import BenchCommand
from lmcache.cli.commands.describe import DescribeCommand
from lmcache.cli.commands.kvcache import KVCacheCommand
from lmcache.cli.commands.mock import MockCommand
from lmcache.cli.commands.ping import PingCommand
from lmcache.cli.commands.query import QueryCommand
from lmcache.cli.commands.server import ServerCommand

ALL_COMMANDS: list[BaseCommand] = [
    MockCommand(),
    KVCacheCommand(),
    DescribeCommand(),
    PingCommand(),
    QueryCommand(),
    ServerCommand(),
    BenchCommand(),
]

__all__ = ["ALL_COMMANDS", "BaseCommand"]
