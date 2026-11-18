#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail if a vLLM nightly wheel index is missing or has not been updated recently.

vLLM publishes nightly wheels per CUDA variant under
``https://wheels.vllm.ai/nightly/<cuda_tag>/``. When upstream stops building a
variant (``cu130`` stopped on 2026-11-10 when vLLM main moved to ``cu132``),
the index keeps serving its last wheel, so ``pip install vllm`` from it still
succeeds and an image silently ships an old vLLM. This check turns that, and an
index that was never published or has been removed, into a build failure.

The age is the ``Last-Modified`` header of ``<index>/vllm/``, which the vLLM
wheel uploader rewrites whenever it publishes a new nightly.

Usage: ``check_vllm_nightly_age.py <index_url> <max_age_days>``
(``max_age_days`` of 0 disables the check). Exits 0 when the index is usable,
1 when it is missing, undated or too old, and 2 on invalid arguments. Network
errors other than HTTP 404 propagate as tracebacks.
"""

# Future
from __future__ import annotations

# Standard
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import argparse
import sys
import urllib.error
import urllib.request

_GUIDANCE = (
    "See https://wheels.vllm.ai/nightly/ for the variants vLLM publishes now. "
    "Pin a vLLM release with VLLM_VERSION=<version>, switch CUDA_VERSION, or "
    "pass VLLM_NIGHTLY_MAX_AGE_DAYS=0 to skip this check."
)


class MissingLastModifiedError(Exception):
    """The index page was served without a ``Last-Modified`` header."""


def index_last_modified(index_url: str) -> datetime:
    """Return when the ``vllm`` project page of a wheel index last changed.

    Args:
        index_url: Base URL of a PEP 503 index, e.g.
            ``https://wheels.vllm.ai/nightly/cu130``.

    Returns:
        The page's ``Last-Modified`` time, timezone-aware (UTC).

    Raises:
        urllib.error.HTTPError: The server answered with an error status,
            e.g. 404 when vLLM does not publish this index.
        urllib.error.URLError: The server cannot be reached.
        MissingLastModifiedError: The page has no ``Last-Modified`` header.
    """
    url = index_url.rstrip("/") + "/vllm/"
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        header = response.headers.get("Last-Modified")
    if header is None:
        raise MissingLastModifiedError(url)
    return parsedate_to_datetime(header)


def main(argv: list[str]) -> int:
    """Check the index age and report the result.

    Args:
        argv: ``[index_url, max_age_days]``; ``max_age_days`` is a
            non-negative integer and 0 skips the check without a request.

    Returns:
        0 if the index is fresh enough or the check is disabled; 1 if the
        index returns 404, has no ``Last-Modified`` or is older than
        ``max_age_days``. Each failure prints the reason and the ways to fix
        the build to stderr.

    Raises:
        SystemExit: With code 2 when the arguments are invalid.
    """
    args = _parse_args(argv)
    if args.max_age_days == 0:
        print(f"vLLM nightly age check disabled for {args.index_url}")
        return 0
    try:
        last_modified = index_last_modified(args.index_url)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        return _fail(f"{args.index_url} is not published (HTTP 404).")
    except MissingLastModifiedError:
        return _fail(f"{args.index_url}/vllm/ has no Last-Modified header.")
    age_days = (datetime.now(timezone.utc) - last_modified).days
    print(f"vLLM nightly index {args.index_url} last updated {last_modified:%Y-%m-%d}")
    if age_days > args.max_age_days:
        return _fail(
            f"{args.index_url} is {age_days} days old (limit "
            f"{args.max_age_days}); vLLM has likely stopped publishing "
            "nightlies for this CUDA variant."
        )
    return 0


def _non_negative_int(value: str) -> int:
    """Parse a non-negative integer for argparse."""
    number = int(value)
    if number < 0:
        raise ValueError(value)
    return number


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse ``argv``; exits with code 2 on invalid input."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("index_url", help="e.g. https://wheels.vllm.ai/nightly/cu130")
    parser.add_argument(
        "max_age_days",
        type=_non_negative_int,
        help="maximum index age in days; 0 disables the check",
    )
    return parser.parse_args(argv)


def _fail(reason: str) -> int:
    """Print ``reason`` with the fix guidance to stderr and return 1."""
    print(f"ERROR: {reason} {_GUIDANCE}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
