# SPDX-License-Identifier: Apache-2.0
"""``lmcache kvcache list`` — list one page of L2 cache objects."""

# Standard
from typing import Any
import argparse
import sys
import urllib.parse

# First Party
from lmcache.cli.commands.base import BaseCommand, positive_int
from lmcache.cli.http import CLIHTTPError, fetch_json, normalize_url

# Object-key fields rendered as table columns, in display order. Every
# column doubles as a JSON field, so the JSON rows carry the full wire key.
_KEY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("chunk_hash_hex", "Chunk hash"),
    ("model_name", "Model"),
    ("kv_rank", "Rank"),
    ("object_group_id", "Group"),
    ("cache_salt", "Salt"),
    ("tags", "Tags"),
)

# Required wire-key fields with their protocol types, and optional fields
# (per the /cache/objects contract) with their default values.
_REQUIRED_KEY_FIELDS: tuple[tuple[str, type], ...] = (
    ("chunk_hash_hex", str),
    ("model_name", str),
    ("kv_rank", int),
)
_OPTIONAL_KEY_FIELDS: tuple[tuple[str, type, Any], ...] = (
    ("object_group_id", int, 0),
    ("cache_salt", str, ""),
    ("tags", dict, {}),
)


def _invalid_entry(index: int, message: str) -> CLIHTTPError:
    """Build the format error for the page entry at *index*."""
    return CLIHTTPError(f"Invalid cache object at index {index}: {message}")


def _flatten_entry(index: int, entry: dict[str, Any]) -> dict[str, Any]:
    """Flatten one ``{key, size_bytes}`` wire object into a table/JSON row.

    ``object_group_id`` (``0``), ``cache_salt`` (``""``), and ``tags``
    (``{}``) are optional on the wire for backward compatibility with
    older servers and are filled with their protocol defaults when absent
    or ``null``. Every other field is required with its protocol type;
    anything missing, ``null``, or wrongly typed is reported as a format
    error instead of being passed off as a value.

    Args:
        index: Position of the entry in the page, for error messages.
        entry: One wire entry with ``key`` and ``size_bytes``.

    Returns:
        The flattened row, ready for ``Section.add_row``.

    Raises:
        CLIHTTPError: If the entry, its key, a required field, or
            ``size_bytes`` is missing or has the wrong type.
    """
    if not isinstance(entry, dict):
        raise _invalid_entry(index, "entry is not an object")
    key = entry.get("key")
    if not isinstance(key, dict):
        raise _invalid_entry(index, "'key' is not an object")

    row: dict[str, Any] = {}
    for field, expected in _REQUIRED_KEY_FIELDS:
        value = key.get(field)
        if not isinstance(value, expected):
            raise _invalid_entry(
                index, f"required key field {field!r} must be {expected.__name__}"
            )
        row[field] = value
    for field, expected, default in _OPTIONAL_KEY_FIELDS:
        value = key.get(field)
        if value is None:
            # Fresh mutable default per row (only ``tags`` is a dict).
            value = {} if field == "tags" else default
        elif not isinstance(value, expected):
            raise _invalid_entry(
                index, f"optional key field {field!r} must be {expected.__name__}"
            )
        row[field] = value
    size_bytes = entry.get("size_bytes")
    if not isinstance(size_bytes, int):
        raise _invalid_entry(index, "'size_bytes' must be an integer")
    row["size_bytes"] = size_bytes
    return row


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


class ListCommand(BaseCommand):
    """List one page of object keys and sizes from an L2 storage adapter."""

    def name(self) -> str:
        return "list"

    def help(self) -> str:
        return "List one page of L2 cache objects."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--url",
            required=True,
            help="Target MP HTTP endpoint (e.g. http://localhost:8080).",
        )
        parser.add_argument(
            "--adapter",
            help="L2 adapter type name (default: the server's primary adapter).",
        )
        parser.add_argument("--model-name", help="Filter by exact model name.")
        parser.add_argument(
            "--page-size",
            type=_page_size,
            default=500,
            help="Maximum objects per page, from 1 to 5000 (default: 500).",
        )
        parser.add_argument(
            "--page-token",
            help="Opaque next_page_token from a previous page; pass it unchanged.",
        )

    def execute(self, args: argparse.Namespace) -> None:
        params: dict[str, str] = {"tier": "l2", "page_size": str(args.page_size)}
        for name in ("adapter", "model_name", "page_token"):
            value = getattr(args, name)
            if value is not None:
                params[name] = value
        url = (
            f"{normalize_url(args.url)}/cache/objects?{urllib.parse.urlencode(params)}"
        )
        try:
            result = fetch_json(url)
            metrics = self.create_metrics("L2 Cache Objects", args)
            metrics.add("adapter", "Adapter", result["adapter"])
            metrics.add("shared", "Shared", result["shared"])
            metrics.add("next_page_token", "Next page token", result["next_page_token"])
            entries = metrics.add_table(
                "entries", "Objects", *_KEY_COLUMNS, ("size_bytes", "Bytes")
            )
            page_entries = result.get("entries")
            if not isinstance(page_entries, list):
                raise CLIHTTPError("Invalid response: 'entries' is not a list")
            for index, entry in enumerate(page_entries):
                entries.add_row(**_flatten_entry(index, entry))
            metrics.emit()
        except CLIHTTPError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc
