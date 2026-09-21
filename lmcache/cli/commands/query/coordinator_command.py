# SPDX-License-Identifier: Apache-2.0
"""``lmcache query coordinator`` — read the MP coordinator's HTTP APIs."""

# Standard
import argparse
import sys

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.commands.query._coordinator import (
    APIS,
    get_json,
    get_text,
    normalize_url,
)
from lmcache.logging import init_logger

logger = init_logger(__name__)

_DEFAULT_URL = "http://127.0.0.1:9300"


class CoordinatorQueryCommand(BaseCommand):
    """Query one of the coordinator's read-only HTTP APIs."""

    def name(self) -> str:
        return "coordinator"

    def help(self) -> str:
        return "Query the MP coordinator's read-only HTTP APIs."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--api",
            required=True,
            choices=sorted(APIS),
            metavar="NAME",
            help="Which API to read. "
            + "; ".join(f"{name}: {api.summary}" for name, api in sorted(APIS.items())),
        )
        parser.add_argument(
            "--url",
            default=_DEFAULT_URL,
            help=f"Coordinator base URL (default: {_DEFAULT_URL}).",
        )
        parser.add_argument(
            "--instance",
            default=None,
            help="Instance id. Narrows --api usage to one server; required "
            "by --api prefetch.",
        )
        parser.add_argument(
            "--cache-salt",
            default=None,
            help="Cache salt. Narrows --api quota to one tenant; pass an "
            "empty string for un-salted traffic.",
        )
        parser.add_argument(
            "--request-id",
            default=None,
            help="Prefetch request id, required by --api prefetch.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=20,
            help="Rows to request for --api keys (default: 20).",
        )

    def execute(self, args: argparse.Namespace) -> None:
        """Fetch the selected API and emit it.

        Args:
            args: Parsed CLI arguments.
        """
        api = APIS[args.api]

        missing = [
            f"--{name.replace('_', '-')}"
            for name in api.requires
            if getattr(args, name, None) is None
        ]
        if missing:
            logger.error("--api %s requires %s", args.api, " and ".join(missing))
            sys.exit(2)

        url = normalize_url(args.url) + api.path(args)

        if api.raw:
            # Prometheus text, not JSON: pass it through untouched so it can
            # be piped to promtool or grep without the report wrapper.
            sys.stdout.write(get_text(url))
            return

        metrics = self.create_metrics(f"Coordinator: {args.api}", args)
        api.render(get_json(url), metrics)
        metrics.emit()
