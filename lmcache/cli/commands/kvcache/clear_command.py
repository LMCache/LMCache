# SPDX-License-Identifier: Apache-2.0
"""``lmcache kvcache clear`` — clear all cached KV data in L1 (CPU)."""

# Standard
import argparse
import sys

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.http import CLIHTTPError, fetch_json, normalize_url


class ClearCommand(BaseCommand):
    """Clear all cached KV data in L1 (CPU) on the MP HTTP server."""

    def name(self) -> str:
        return "clear"

    def help(self) -> str:
        return "Clear all cached KV data in L1 (CPU)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--url",
            required=True,
            help="Target MP HTTP endpoint (e.g. http://localhost:8080).",
        )

    def execute(self, args: argparse.Namespace) -> None:
        url = f"{normalize_url(args.url)}/cache/clear"
        try:
            fetch_json(url, method="POST", data={"tier": "l1"})
        except CLIHTTPError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc

        metrics = self.create_metrics("KV Cache Clear", args)
        metrics.add("status", "Status", "OK")
        metrics.emit()
