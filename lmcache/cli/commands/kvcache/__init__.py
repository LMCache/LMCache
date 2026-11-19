# SPDX-License-Identifier: Apache-2.0
"""``lmcache kvcache`` command — KV cache state management.

Sub-subcommands are auto-discovered from modules in this package.
To add a new subcommand, create a module defining a concrete
:class:`BaseCommand` subclass — no edits to this file are required.
"""

# First Party
from lmcache.cli.commands.base import CompositeCommand


class KVCacheCommand(CompositeCommand):
    """CLI command for KV cache state management on the LMCache server."""

    def name(self) -> str:
        return "kvcache"

    def help(self) -> str:
        return "Manage KV cache state."
