# SPDX-License-Identifier: Apache-2.0
"""``lmcache quota delete`` — remove a quota for a cache_salt."""

# Standard
from typing import TYPE_CHECKING
import argparse

# First Party
from lmcache.cli.commands.base import _add_output_args
from lmcache.cli.commands.quota.helpers import (
    escape_salt,
    http_request,
    normalize_url,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.cli.commands.base import BaseCommand


def register_delete_parser(
    subparsers: argparse._SubParsersAction,
    dispatch_func,
) -> argparse.ArgumentParser:
    """Register the ``lmcache quota delete`` subcommand parser.

    Args:
        subparsers: The ``quota`` subparsers action.
        dispatch_func: Function to bind via ``set_defaults(func=...)``.

    Returns:
        The created ``ArgumentParser``.
    """
    parser = subparsers.add_parser(
        "delete",
        help="Remove a quota for a cache_salt.",
        description=(
            "Delete the quota entry for a given cache_salt. Any bytes "
            "still cached under this salt become over-budget on the "
            "next eviction cycle and will be evicted."
        ),
    )
    parser.add_argument(
        "salt",
        type=str,
        help=(
            "The cache_salt identifier. Use '_default' for anonymous "
            "(un-salted) traffic."
        ),
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8080",
        help="LMCache HTTP server URL (default: http://localhost:8080).",
    )
    _add_output_args(parser)
    parser.set_defaults(func=dispatch_func)
    return parser


def run_quota_delete(cmd: "BaseCommand", args: argparse.Namespace) -> None:
    """Execute the ``lmcache quota delete`` subcommand.

    Args:
        cmd: The parent command instance (for metrics creation).
        args: Parsed CLI arguments.
    """
    base_url = normalize_url(args.url)
    salt = escape_salt(args.salt)

    result = http_request("DELETE", f"{base_url}/quota/{salt}")

    metrics = cmd.create_metrics("Quota Delete", args)
    metrics.add("cache_salt", "Cache salt", result.get("cache_salt", salt))
    metrics.add("status", "Status", result.get("status", "unknown"))
    metrics.emit()
