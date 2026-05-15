# SPDX-License-Identifier: Apache-2.0
"""``lmcache conf`` — fetch the active server configuration as JSON.

Usage::

    lmcache conf --url http://localhost:8080
    lmcache conf --file /tmp/lmcache-config-8080.json
"""

# Standard
from pathlib import Path
import argparse
import json
import sys
import urllib.error
import urllib.request

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.commands.describe import normalize_url

DEFAULT_URL = "http://localhost:8080"


def _fetch_from_url(url: str, timeout: int = 10) -> str:
    """GET the ``/conf`` endpoint and return the response body.

    Args:
        url: Fully-qualified ``/conf`` URL.
        timeout: Request timeout in seconds.

    Returns:
        Response body as a string.

    Raises:
        urllib.error.URLError: If the request cannot be made.
        urllib.error.HTTPError: If the server responds with an error
            status.
    """
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _read_from_file(path: Path) -> str:
    """Read a previously persisted config-dump file.

    Args:
        path: Filesystem path to the JSON dump.

    Returns:
        File contents as a string.

    Raises:
        OSError: If the file cannot be read.
    """
    return path.read_text()


class ConfCommand(BaseCommand):
    """Fetch the active LMCache server configuration as JSON."""

    def name(self) -> str:
        """Return the subcommand name.

        Returns:
            The string ``"conf"``.
        """
        return "conf"

    def help(self) -> str:
        """Return short help text shown by ``lmcache -h``.

        Returns:
            Help string.
        """
        return "Fetch the LMCache server configuration as JSON."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register CLI arguments for the ``conf`` subcommand.

        Args:
            parser: The ``ArgumentParser`` for this subcommand.
        """
        source = parser.add_mutually_exclusive_group()
        source.add_argument(
            "--url",
            type=str,
            default=None,
            help=(
                f"LMCache HTTP server base URL (default: {DEFAULT_URL}). "
                f"Mutually exclusive with --file."
            ),
        )
        source.add_argument(
            "--file",
            type=str,
            default=None,
            help=(
                "Read configuration from a previously persisted JSON dump "
                "instead of querying a running server. Mutually exclusive "
                "with --url."
            ),
        )

    def execute(self, args: argparse.Namespace) -> None:
        """Fetch the config and print it as pretty-printed JSON to stdout.

        On error, the message is written to stderr and the process exits
        with status 1.

        Args:
            args: Parsed CLI arguments.
        """
        if args.file is not None:
            try:
                body = _read_from_file(Path(args.file))
            except OSError as exc:
                print(
                    f"Cannot read {args.file}: {exc}",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            base_url = normalize_url(args.url or DEFAULT_URL)
            endpoint = f"{base_url}/conf"
            try:
                body = _fetch_from_url(endpoint)
            except urllib.error.HTTPError as exc:
                print(
                    f"HTTP {exc.code}: {exc.reason}",
                    file=sys.stderr,
                )
                sys.exit(1)
            except (urllib.error.URLError, OSError) as exc:
                reason = getattr(exc, "reason", str(exc))
                print(
                    f"Cannot connect to {endpoint}: {reason}",
                    file=sys.stderr,
                )
                sys.exit(1)
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            print(body)
            return
        print(json.dumps(parsed, indent=2, sort_keys=True))
