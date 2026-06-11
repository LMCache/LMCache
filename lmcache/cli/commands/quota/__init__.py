# SPDX-License-Identifier: Apache-2.0
"""``lmcache quota`` command — per-salt quota management.

Subcommands:

* ``set SALT --limit-gb N``  — create or update a quota
* ``get SALT``               — show quota and current usage for a salt
* ``list``                   — list all registered quotas
* ``delete SALT``            — remove a quota entry
"""

# Standard
import argparse
import sys

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.commands.quota.delete_command import (
    register_delete_parser,
    run_quota_delete,
)
from lmcache.cli.commands.quota.get_command import (
    register_get_parser,
    run_quota_get,
)
from lmcache.cli.commands.quota.list_command import (
    register_list_parser,
    run_quota_list,
)
from lmcache.cli.commands.quota.set_command import (
    register_set_parser,
    run_quota_set,
)
from lmcache.logging import init_logger

logger = init_logger(__name__)


class QuotaCommand(BaseCommand):
    """CLI command for per-salt quota management on LMCache server."""

    def name(self) -> str:
        return "quota"

    def help(self) -> str:
        return "Manage per-salt cache quotas."

    def add_arguments(self, _parser: argparse.ArgumentParser) -> None:
        pass  # args registered in register() via subparsers

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        """Register ``lmcache quota`` and all quota sub-subcommands.

        Args:
            subparsers: The subparsers action from the root parser.
        """
        parser = subparsers.add_parser(
            self.name(),
            help=self.help(),
            description="Manage per-salt cache quotas on the LMCache server.",
        )
        inner = parser.add_subparsers(
            dest="quota_action",
            required=True,
            metavar="{set,get,list,delete}",
        )
        register_set_parser(inner, self.execute)
        register_get_parser(inner, self.execute)
        register_list_parser(inner, self.execute)
        register_delete_parser(inner, self.execute)

    def execute(self, args: argparse.Namespace) -> None:
        """Dispatch to the appropriate quota subcommand handler.

        Args:
            args: Parsed CLI arguments containing ``quota_action``.
        """
        handlers = {
            "set": lambda a: run_quota_set(self, a),
            "get": lambda a: run_quota_get(self, a),
            "list": lambda a: run_quota_list(self, a),
            "delete": lambda a: run_quota_delete(self, a),
        }
        handler = handlers.get(args.quota_action)
        if handler is None:
            print(
                f"Unknown quota action: {args.quota_action}",
                file=sys.stderr,
            )
            sys.exit(1)
        handler(args)
