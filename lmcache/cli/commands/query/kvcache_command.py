# SPDX-License-Identifier: Apache-2.0
"""``lmcache query kvcache`` — query KV-cache endpoints (placeholder)."""

# Standard
from typing import TYPE_CHECKING
import argparse

# First Party
from lmcache.cli.commands.base import _add_output_args

if TYPE_CHECKING:
    # First Party
    from lmcache.cli.commands.base import BaseCommand


def register_kvcache_parser(
    subparsers: argparse._SubParsersAction,
    dispatch_func,
) -> argparse.ArgumentParser:
    """Register the ``lmcache query kvcache`` subcommand parser.

    Args:
        subparsers: The ``query`` subparsers action.
        dispatch_func: Function to bind via ``set_defaults(func=...)``.

    Returns:
        The created ``ArgumentParser``.
    """
    parser = subparsers.add_parser(
        "kvcache",
        help="Query KV-cache endpoints (not implemented yet).",
    )
    _add_output_args(parser)
    parser.set_defaults(func=dispatch_func)
    return parser


def run_query_kvcache(cmd: "BaseCommand", args: argparse.Namespace) -> None:
    """Execute the ``lmcache query kvcache`` subcommand.

    Args:
        cmd: The parent command instance (for metrics creation).
        args: Parsed CLI arguments.
    """
    # TODO: implement kvcache query logic
    pass
