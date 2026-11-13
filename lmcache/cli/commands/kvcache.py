# SPDX-License-Identifier: Apache-2.0
"""``lmcache kvcache`` — KV cache management.

Sub-commands:
    clear         Clear all cached KV data
    delete        Delete specific cached objects by key
"""

# Standard
from pathlib import Path
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
    MP HTTP server. Currently supports clearing all L1 cache. Future
    sub-commands (pin, compress, info) are defined in the design doc.
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

        # delete
        p_delete = sub.add_parser(
            "delete",
            help="Delete specific cached objects by key (L1, L2, or both).",
        )
        p_delete.add_argument(
            "--url",
            type=str,
            required=True,
            help="Target MP HTTP endpoint (e.g. http://localhost:8000).",
        )
        source = p_delete.add_mutually_exclusive_group(required=True)
        source.add_argument(
            "--key",
            type=str,
            action="append",
            dest="keys",
            metavar="JSON",
            help="Object key as JSON, repeatable. Fields: chunk_hash_hex, "
            "model_name, kv_rank and optionally object_group_id, cache_salt, "
            "tags.",
        )
        source.add_argument(
            "--keys-file",
            type=str,
            metavar="PATH",
            help="JSON file holding an array of object keys, or an object with "
            'a "keys" array.',
        )
        p_delete.add_argument(
            "--tier",
            type=str,
            choices=("l1", "l2", "all"),
            default="l1",
            help="Tier to delete from. Default is l1.",
        )
        p_delete.add_argument(
            "--adapter",
            type=str,
            default=None,
            help="L2 adapter name; ignored when --tier is l1. Defaults to the "
            "primary adapter.",
        )
        p_delete.add_argument(
            "--force",
            action="store_true",
            help="Delete L1 keys even while they are read- or write-locked.",
        )
        p_delete.add_argument(
            "-y",
            "--yes",
            action="store_true",
            help="Skip the confirmation prompt. Required when stdin is not a terminal.",
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
            "delete": self._delete,
        }
        dispatch[action](args)

    @staticmethod
    def _parse_key(raw: str) -> dict[str, Any]:
        """Parse one ``--key`` JSON argument.

        Raises:
            SystemExit: If the value is not a JSON object.
        """
        try:
            key = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("--key is not valid JSON: %s (%s)", raw, exc)
            sys.exit(2)
        if not isinstance(key, dict):
            logger.error("--key must be a JSON object, got %s", type(key).__name__)
            sys.exit(2)
        return key

    @classmethod
    def _load_keys(cls, args: argparse.Namespace) -> list[dict[str, Any]]:
        """Collect the object keys to delete from ``--key`` or ``--keys-file``.

        Raises:
            SystemExit: If no keys are supplied or the file cannot be read or
                parsed.
        """
        raw_keys = getattr(args, "keys", None)
        if raw_keys:
            return [cls._parse_key(raw) for raw in raw_keys]

        path = Path(args.keys_file)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            logger.error("Cannot read --keys-file %s: %s", path, exc)
            sys.exit(2)
        except json.JSONDecodeError as exc:
            logger.error("--keys-file %s is not valid JSON: %s", path, exc)
            sys.exit(2)

        if isinstance(payload, dict):
            payload = payload.get("keys")
        if not isinstance(payload, list) or not payload:
            logger.error('--keys-file must hold a non-empty array, or {"keys": [...]}')
            sys.exit(2)
        for key in payload:
            if not isinstance(key, dict):
                logger.error("Every key in --keys-file must be a JSON object")
                sys.exit(2)
        return payload

    @staticmethod
    def _confirm(count: int, tier: str) -> None:
        """Ask before deleting, unless ``--yes`` was passed and stdin is a TTY.

        Raises:
            SystemExit: If the caller declined, or stdin is not interactive.
        """
        if not sys.stdin.isatty():
            logger.error("Refusing to delete %d object(s) without --yes", count)
            sys.exit(2)
        answer = input(f"Delete {count} object(s) from {tier}? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            logger.error("Aborted")
            sys.exit(1)

    def _delete(self, args: argparse.Namespace) -> None:
        """Delete specific cached objects via ``DELETE /cache/objects``."""
        url = args.url.rstrip("/")
        keys = self._load_keys(args)
        tier = args.tier

        if not getattr(args, "yes", False):
            self._confirm(len(keys), tier)

        body: dict[str, Any] = {"keys": keys, "tier": tier, "force": args.force}
        if args.adapter is not None and tier != "l1":
            body["adapter"] = args.adapter

        result = _http_request("DELETE", f"{url}/cache/objects", body)

        # The server answers 200 with ``ok: false`` when an adapter rejected
        # part of the batch, so the exit status cannot be inferred from the
        # HTTP response alone. A caller running `delete ... && next-step`
        # must not treat that as success.
        succeeded = bool(result.get("ok"))

        if not getattr(args, "quiet", False):
            metrics = self.create_metrics("KV Cache Delete", args)
            metrics.add("status", "Status", "OK" if succeeded else "FAIL")
            metrics.add("deleted", "Deleted objects", result.get("deleted", 0))
            metrics.add("skipped", "Skipped objects", result.get("skipped", 0))
            if result.get("error"):
                metrics.add("error", "Error", result["error"])
            metrics.emit()

        if not succeeded:
            sys.exit(1)

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
