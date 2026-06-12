# SPDX-License-Identifier: Apache-2.0
"""``lmcache quota set`` — create or update a quota for a cache_salt."""

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


def register_set_parser(
    subparsers: argparse._SubParsersAction,
    dispatch_func,
) -> argparse.ArgumentParser:
    """Register the ``lmcache quota set`` subcommand parser.

    Args:
        subparsers: The ``quota`` subparsers action.
        dispatch_func: Function to bind via ``set_defaults(func=...)``.

    Returns:
        The created ``ArgumentParser``.
    """
    parser = subparsers.add_parser(
        "set",
        help="Create or update a quota for a cache_salt.",
        description=(
            "Set a per-salt quota (in GB) on the LMCache server. "
            "The quota is soft: exceeding it triggers eviction on the "
            "next cycle rather than rejecting writes."
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
        "--limit-gb",
        type=float,
        required=True,
        metavar="GB",
        help="Quota limit in gigabytes (non-negative).",
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


def run_quota_set(cmd: "BaseCommand", args: argparse.Namespace) -> None:
    """Execute the ``lmcache quota set`` subcommand.

    Args:
        cmd: The parent command instance (for metrics creation).
        args: Parsed CLI arguments.
    """
    base_url = normalize_url(args.url)
    salt = escape_salt(args.salt)
    limit_gb = args.limit_gb

    result = http_request(
        "PUT",
        f"{base_url}/quota/{salt}",
        data={"limit_gb": limit_gb},
    )

    metrics = cmd.create_metrics("Quota Set", args)
    metrics.add("cache_salt", "Cache salt", result.get("cache_salt", salt))
    metrics.add("limit_gb", "Limit (GB)", result.get("limit_gb", limit_gb))
    metrics.add("status", "Status", result.get("status", "ok"))
    metrics.emit()
