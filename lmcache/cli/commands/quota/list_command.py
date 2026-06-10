# SPDX-License-Identifier: Apache-2.0
"""``lmcache quota list`` — list all registered quotas and their usage."""

# Standard
from typing import TYPE_CHECKING
import argparse

# First Party
from lmcache.cli.commands.base import _add_output_args
from lmcache.cli.commands.quota.helpers import http_request, normalize_url

if TYPE_CHECKING:
    # First Party
    from lmcache.cli.commands.base import BaseCommand


def register_list_parser(
    subparsers: argparse._SubParsersAction,
    dispatch_func,
) -> argparse.ArgumentParser:
    """Register the ``lmcache quota list`` subcommand parser.

    Args:
        subparsers: The ``quota`` subparsers action.
        dispatch_func: Function to bind via ``set_defaults(func=...)``.

    Returns:
        The created ``ArgumentParser``.
    """
    parser = subparsers.add_parser(
        "list",
        help="List all registered quotas and their usage.",
        description=(
            "Retrieve all per-salt quotas from the LMCache server "
            "along with their current usage."
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


def run_quota_list(cmd: "BaseCommand", args: argparse.Namespace) -> None:
    """Execute the ``lmcache quota list`` subcommand.

    Args:
        cmd: The parent command instance (for metrics creation).
        args: Parsed CLI arguments.
    """
    base_url = normalize_url(args.url)

    result = http_request("GET", f"{base_url}/quota")
    users = result.get("users", {})

    metrics = cmd.create_metrics("Quota List", args, width=55)

    if not users:
        metrics.add("info", "Info", "No quotas configured")
        metrics.emit()
        return

    for idx, (salt, info) in enumerate(users.items()):
        section_key = f"quota_{idx}"
        metrics.add_list_section("quotas", section_key, f"Salt: {salt}")
        sec = metrics[section_key]
        sec.add("cache_salt", "Cache salt", salt)
        sec.add("limit_gb", "Limit (GB)", info.get("limit_gb"))
        sec.add("current_usage_gb", "Current usage (GB)", info.get("current_usage_gb"))

    metrics.emit()
