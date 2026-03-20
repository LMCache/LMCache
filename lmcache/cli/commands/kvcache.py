# SPDX-License-Identifier: Apache-2.0
"""``lmcache kvcache`` — per-request KV cache management.

Sub-commands:
    clear         Clear all cached KV data
    end-session   Clean up session state for a finished request
    pin           Pin a request's KV cache to L1/CPU (not yet implemented)
    compress      Compress a request's KV cache in-place (not yet implemented)
    info          Show per-request cache state (not yet implemented)
"""

# Standard
from typing import Any, Optional
import argparse
import json
import sys
import urllib.error
import urllib.request

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.logging import init_logger

logger = init_logger(__name__)


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
    MP HTTP server: clearing all cache, ending sessions, and (in the
    future) pinning, compressing, and inspecting per-request cache state.
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
        p_clear = sub.add_parser("clear", help="Clear all cached KV data.")
        p_clear.add_argument(
            "--url",
            type=str,
            required=True,
            help="Target MP HTTP endpoint (e.g. http://localhost:8000).",
        )

        # end-session
        p_end = sub.add_parser(
            "end-session",
            help="Clean up session state for a finished request.",
        )
        p_end.add_argument(
            "--url",
            type=str,
            required=True,
            help="Target MP HTTP endpoint (e.g. http://localhost:8000).",
        )
        p_end.add_argument(
            "--request-id",
            type=str,
            required=True,
            help="Request ID whose session should be removed.",
        )

        # pin (stub)
        p_pin = sub.add_parser(
            "pin",
            help="Pin a request's KV cache to L1/CPU (not yet implemented).",
        )
        p_pin.add_argument("--url", type=str, required=True)
        p_pin.add_argument("--request-id", type=str, required=True)
        p_pin.add_argument("--start", type=int, default=None)
        p_pin.add_argument("--end", type=int, default=None)

        # compress (stub)
        p_compress = sub.add_parser(
            "compress",
            help="Compress a request's KV cache in-place (not yet implemented).",
        )
        p_compress.add_argument("--url", type=str, required=True)
        p_compress.add_argument("--request-id", type=str, required=True)
        p_compress.add_argument("--method", type=str, required=True)
        p_compress.add_argument("--start", type=int, default=None)
        p_compress.add_argument("--end", type=int, default=None)

        # info (stub)
        p_info = sub.add_parser(
            "info",
            help="Show per-request cache state (not yet implemented).",
        )
        p_info.add_argument("--url", type=str, required=True)
        p_info.add_argument("--request-id", type=str, required=True)
        p_info.add_argument("--start", type=int, default=None)
        p_info.add_argument("--end", type=int, default=None)

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
            "end-session": self._end_session,
            "pin": self._not_implemented,
            "compress": self._not_implemented,
            "info": self._not_implemented,
        }
        dispatch[action](args)

    def _clear(self, args: argparse.Namespace) -> None:
        """Clear all cached KV data via the MP HTTP server."""
        url = args.url.rstrip("/")

        # MP HTTP server endpoint: POST /api/clear-cache
        _http_request("POST", f"{url}/api/clear-cache")

        quiet = getattr(args, "quiet", False)
        if quiet:
            return

        metrics = self.create_metrics("KV Cache Clear", args)
        metrics.add("status", "Status", "OK")
        metrics.emit()

    def _end_session(self, args: argparse.Namespace) -> None:
        """End a request's session via the MP HTTP server."""
        url = args.url.rstrip("/")
        request_id = args.request_id

        # This endpoint does not exist yet on the MP HTTP server.
        # See docs/design/cli/kvcache-command.md for the design.
        # Expected: POST /api/end-session  {"request_id": "..."}
        result = _http_request(
            "POST",
            f"{url}/api/end-session",
            data={"request_id": request_id},
        )

        quiet = getattr(args, "quiet", False)
        if quiet:
            return

        metrics = self.create_metrics(f"KV Cache End Session ({request_id})", args)
        metrics.add("status", "Status", result.get("status", "OK"))
        metrics.emit()

    def _not_implemented(self, args: argparse.Namespace) -> None:
        """Print an error for sub-commands that are not yet implemented."""
        action = args.action
        logger.error(
            "'lmcache kvcache %s' is not yet implemented. "
            "See docs/design/cli/kvcache-command.md for the design.",
            action,
        )
        sys.exit(1)
