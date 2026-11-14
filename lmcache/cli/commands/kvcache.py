# SPDX-License-Identifier: Apache-2.0
"""``lmcache kvcache`` — KV cache management.

Sub-commands:
    clear         Clear all cached KV data
    list          List one page of L2 cache objects
"""

# Standard
from typing import Any, Optional
import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

# First Party
from lmcache.cli.commands.base import BaseCommand, positive_int
from lmcache.cli.http import CLIHTTPError, fetch_json, normalize_url
from lmcache.logging import init_logger

logger = init_logger(__name__)


def _page_size(value: str) -> int:
    """Parse a page size within the HTTP endpoint's 1–5000 object limit.

    Args:
        value: Page-size text supplied on the command line.

    Returns:
        The requested number of objects per page.

    Raises:
        ValueError: If value is not integer text.
        argparse.ArgumentTypeError: If the size is outside 1–5000.
    """
    size = positive_int(value)
    if size > 5000:
        raise argparse.ArgumentTypeError("must be at most 5000")
    return size


def _http_request(
    method: str, url: str, data: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Send an HTTP request and return the parsed JSON response.

    Args:
        method: HTTP method (GET, POST, PUT, DELETE).
        url: Full URL to request.
        data: Optional JSON body to send.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        SystemExit: On connection error or non-2xx HTTP response.
    """
    body = None
    headers: dict[str, str] = {}
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            error_body = json.loads(e.read().decode())
            msg = error_body.get("message") or error_body.get("error") or str(e)
        except (json.JSONDecodeError, ValueError, OSError):
            msg = str(e)
        logger.error("Server error: %s", msg)
        sys.exit(1)
    except urllib.error.URLError as e:
        logger.error("Cannot reach %s — is the server running? (%s)", url, e.reason)
        sys.exit(1)


class KVCacheCommand(BaseCommand):
    """Manage KV cache state.

    This command provides sub-commands for managing KV cache data on the
    MP HTTP server. Supports clearing all L1 cache and listing one page of
    objects in an L2 adapter. Future sub-commands (pin, compress, info) are
    defined in the design doc.
    """

    def name(self) -> str:
        """Return the subcommand name.

        Returns:
            The string ``"kvcache"``.
        """
        return "kvcache"

    def help(self) -> str:
        """Return short help text.

        Returns:
            Help string shown by ``lmcache -h``.
        """
        return "Manage KV cache state."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add kvcache-specific sub-commands and their arguments.

        Args:
            parser: The ``ArgumentParser`` for this subcommand.
        """
        sub = parser.add_subparsers(dest="action")

        # clear
        p_clear = sub.add_parser("clear", help="Clear all cached KV data in L1 (CPU).")
        p_clear.add_argument(
            "--url",
            type=str,
            required=True,
            help="Target MP HTTP endpoint (e.g. http://localhost:8000).",
        )

        p_list = sub.add_parser("list", help="List one page of L2 cache objects.")
        p_list.add_argument(
            "--url",
            required=True,
            help="Target MP HTTP endpoint (e.g. http://localhost:8080).",
        )
        p_list.add_argument(
            "--adapter",
            help="L2 adapter type name (default: the server's primary adapter).",
        )
        p_list.add_argument("--model-name", help="Filter by exact model name.")
        p_list.add_argument(
            "--page-size",
            type=_page_size,
            default=500,
            help="Maximum objects per page, from 1 to 5000 (default: 500).",
        )
        p_list.add_argument(
            "--page-token",
            help="Opaque next_page_token from a previous page; pass it unchanged.",
        )

    def execute(self, args: argparse.Namespace) -> None:
        """Dispatch to the appropriate sub-command handler.

        Args:
            args: Parsed CLI arguments. Must contain ``action`` indicating
                which sub-command was invoked.
        """
        action = getattr(args, "action", None)
        if action is None:
            logger.error("No sub-command specified. Run: lmcache kvcache -h")
            sys.exit(1)

        dispatch = {
            "clear": self._clear,
            "list": self._list,
        }
        dispatch[action](args)

    def _clear(self, args: argparse.Namespace) -> None:
        """Clear all cached KV data via the MP HTTP server."""
        url = args.url.rstrip("/")

        # MP HTTP server endpoint: POST /cache/clear
        _http_request("POST", f"{url}/cache/clear", {"tier": "l1"})

        quiet = getattr(args, "quiet", False)
        if quiet:
            return

        metrics = self.create_metrics("KV Cache Clear", args)
        metrics.add("status", "Status", "OK")
        metrics.emit()

    def _list(self, args: argparse.Namespace) -> None:
        """Fetch one L2 page, preserving object keys and the continuation token."""
        params = {"tier": "l2", "page_size": str(args.page_size)}
        for name in ("adapter", "model_name", "page_token"):
            value = getattr(args, name)
            if value is not None:
                params[name] = value
        url = (
            f"{normalize_url(args.url)}/cache/objects?{urllib.parse.urlencode(params)}"
        )
        try:
            result = fetch_json(url)
        except CLIHTTPError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc

        metrics = self.create_metrics("L2 Cache Objects", args)
        metrics.add("adapter", "Adapter", result["adapter"])
        metrics.add("shared", "Shared", result["shared"])
        metrics.add("next_page_token", "Next page token", result["next_page_token"])
        entries = metrics.add_table(
            "entries", "Objects", ("key", "Key"), ("size_bytes", "Bytes")
        )
        for entry in result["entries"]:
            entries.add_row(key=entry["key"], size_bytes=entry["size_bytes"])
        metrics.emit()
