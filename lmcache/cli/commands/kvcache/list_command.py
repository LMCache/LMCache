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


def _is_int(value: Any) -> bool:
    """Return whether *value* is an integer and not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def _invalid_entry(index: int, message: str) -> CLIHTTPError:
    """Build the format error for the page entry at *index*."""
    return CLIHTTPError(f"Invalid cache object at index {index}: {message}")


def _validated_page(result: Any) -> dict[str, Any]:
    """Validate the shape of a ``/cache/objects`` page before use.

    Args:
        result: The decoded response body.

    Returns:
        The same object, once every documented field checks out:
        ``adapter`` a string, ``shared`` a boolean, ``entries`` a list,
        and ``next_page_token`` present as a string or ``null``.

    Raises:
        CLIHTTPError: If the body is not an object or any documented
            field is missing or has the wrong type.
    """
    if not isinstance(result, dict):
        raise CLIHTTPError("Invalid response: body is not an object")
    if not isinstance(result.get("adapter"), str):
        raise CLIHTTPError("Invalid response: 'adapter' must be a string")
    if not isinstance(result.get("shared"), bool):
        raise CLIHTTPError("Invalid response: 'shared' must be a boolean")
    if not isinstance(result.get("entries"), list):
        raise CLIHTTPError("Invalid response: 'entries' is not a list")
    if "next_page_token" not in result:
        raise CLIHTTPError("Invalid response: 'next_page_token' is missing")
    token = result["next_page_token"]
    if token is not None and not isinstance(token, str):
        raise CLIHTTPError(
            "Invalid response: 'next_page_token' must be a string or null"
        )
    return result


def _flatten_entry(index: int, entry: dict[str, Any]) -> dict[str, Any]:
    """Flatten one ``{key, size_bytes}`` wire object into a table/JSON row.

    ``object_group_id`` (``0``), ``cache_salt`` (``""``), and ``tags``
    (``{}``) are optional on the wire for backward compatibility with
    older servers and are filled with their protocol defaults when absent
    or ``null``. Every other field is required with its protocol type;
    anything missing, ``null``, wrongly typed, or a boolean in an integer
    slot is reported as a format error instead of being passed off as a
    value. Tag names must be non-empty strings and tag values strings,
    per the identity-tag contract.

    Args:
        index: Position of the entry in the page, for error messages.
        entry: One wire entry with ``key`` and ``size_bytes``.

    Returns:
        The flattened row, ready for ``Section.add_row``.

    Raises:
        CLIHTTPError: If the entry, its key, or any field is missing,
            mistyped, or violates the tag contract.
    """
    if not isinstance(entry, dict):
        raise _invalid_entry(index, "entry is not an object")
    key = entry.get("key")
    if not isinstance(key, dict):
        raise _invalid_entry(index, "'key' is not an object")

    chunk_hash_hex = key.get("chunk_hash_hex")
    if not isinstance(chunk_hash_hex, str):
        raise _invalid_entry(index, "required key field 'chunk_hash_hex' must be str")
    model_name = key.get("model_name")
    if not isinstance(model_name, str):
        raise _invalid_entry(index, "required key field 'model_name' must be str")
    kv_rank = key.get("kv_rank")
    if not _is_int(kv_rank):
        raise _invalid_entry(index, "required key field 'kv_rank' must be int")

    object_group_id = key.get("object_group_id")
    if object_group_id is None:
        object_group_id = 0
    elif not _is_int(object_group_id):
        raise _invalid_entry(index, "optional key field 'object_group_id' must be int")
    cache_salt = key.get("cache_salt")
    if cache_salt is None:
        cache_salt = ""
    elif not isinstance(cache_salt, str):
        raise _invalid_entry(index, "optional key field 'cache_salt' must be str")
    tags = key.get("tags")
    if tags is None:
        tags = {}
    elif not isinstance(tags, dict):
        raise _invalid_entry(index, "optional key field 'tags' must be an object")
    for tag_name, tag_value in tags.items():
        if not isinstance(tag_name, str) or not tag_name:
            raise _invalid_entry(index, "tag names must be non-empty strings")
        if not isinstance(tag_value, str):
            raise _invalid_entry(index, "tag values must be strings")

    size_bytes = entry.get("size_bytes")
    if not _is_int(size_bytes):
        raise _invalid_entry(index, "'size_bytes' must be an integer")

    return {
        "chunk_hash_hex": chunk_hash_hex,
        "model_name": model_name,
        "kv_rank": kv_rank,
        "object_group_id": object_group_id,
        "cache_salt": cache_salt,
        "tags": tags,
        "size_bytes": size_bytes,
    }


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
            result = _validated_page(fetch_json(url))
            metrics = self.create_metrics("L2 Cache Objects", args)
            metrics.add("adapter", "Adapter", result["adapter"])
            metrics.add("shared", "Shared", result["shared"])
            metrics.add("next_page_token", "Next page token", result["next_page_token"])
            entries = metrics.add_table(
                "entries", "Objects", *_KEY_COLUMNS, ("size_bytes", "Bytes")
            )
            for index, entry in enumerate(result["entries"]):
                entries.add_row(**_flatten_entry(index, entry))
            metrics.emit()
        except CLIHTTPError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc
