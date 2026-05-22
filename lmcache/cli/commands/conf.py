# SPDX-License-Identifier: Apache-2.0
"""``lmcache conf`` — fetch the active server configuration as JSON.

Usage::

    lmcache conf [--url URL] [--file PATH]
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
CONF_ENDPOINT = "/conf"


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


def _write_to_file(path: Path, body: str) -> None:
    """Write a JSON configuration file.

    Args:
        path: Filesystem path where the JSON config should be written.
        body: File contents to write.

    Raises:
        OSError: If the file cannot be written.
    """
    path.write_text(body)


def _resolve_conf_endpoint(url: str | None) -> str:
    """Resolve a base URL or ``/conf`` endpoint URL to the endpoint URL.

    Args:
        url: Optional server base URL, port shorthand, or full ``/conf`` URL.

    Returns:
        Fully-qualified URL for the ``/conf`` endpoint.
    """
    base_url = normalize_url(url or DEFAULT_URL)
    if base_url.endswith(CONF_ENDPOINT):
        return base_url
    return f"{base_url}{CONF_ENDPOINT}"


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
        parser.add_argument(
            "--url",
            type=str,
            default=None,
            help=(
                f"LMCache HTTP server base URL. Default: {DEFAULT_URL}."
            ),
        )
        parser.add_argument(
            "--file",
            type=str,
            default=None,
            help=(
                "Write the fetched configuration JSON to this file."
            ),
        )

    def execute(self, args: argparse.Namespace) -> None:
        """Fetch the config and print it as pretty-printed JSON to stdout.

        On error, the message is written to stderr and the process exits
        with status 1.

        Args:
            args: Parsed CLI arguments.
        """
        endpoint = _resolve_conf_endpoint(args.url)
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
        pretty = json.dumps(parsed, indent=2, sort_keys=True)
        if args.file is not None:
            try:
                _write_to_file(Path(args.file), pretty + "\n")
            except OSError as exc:
                print(
                    f"Cannot write {args.file}: {exc}",
                    file=sys.stderr,
                )
                sys.exit(1)
        print(pretty)
