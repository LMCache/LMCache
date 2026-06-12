# SPDX-License-Identifier: Apache-2.0
"""``lmcache quota get`` — read the quota and usage for a cache_salt."""

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


def register_get_parser(
    subparsers: argparse._SubParsersAction,
    dispatch_func,
) -> argparse.ArgumentParser:
    """Register the ``lmcache quota get`` subcommand parser.

    Args:
        subparsers: The ``quota`` subparsers action.
        dispatch_func: Function to bind via ``set_defaults(func=...)``.

    Returns:
        The created ``ArgumentParser``.
    """
    parser = subparsers.add_parser(
        "get",
        help="Show the quota and current usage for a cache_salt.",
        description=(
            "Query the current quota limit and live usage for a "
            "specific cache_salt on the LMCache server."
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


def run_quota_get(cmd: "BaseCommand", args: argparse.Namespace) -> None:
    """Execute the ``lmcache quota get`` subcommand.

    Args:
        cmd: The parent command instance (for metrics creation).
        args: Parsed CLI arguments.
    """
    base_url = normalize_url(args.url)
    salt = escape_salt(args.salt)

    result = http_request("GET", f"{base_url}/quota/{salt}")

    metrics = cmd.create_metrics("Quota Info", args)
    metrics.add("cache_salt", "Cache salt", result.get("cache_salt", salt))
    metrics.add("limit_gb", "Limit (GB)", result.get("limit_gb"))
    metrics.add(
        "current_usage_gb", "Current usage (GB)", result.get("current_usage_gb")
    )
    metrics.add("exists", "Exists", result.get("exists"))
    metrics.emit()
