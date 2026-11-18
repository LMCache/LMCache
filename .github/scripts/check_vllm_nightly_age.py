#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail if a vLLM nightly wheel index has not been updated recently.

vLLM publishes nightly wheels per CUDA variant under
``https://wheels.vllm.ai/nightly/<cuda_tag>/``. When upstream stops building a
variant (``cu130`` stopped on 2026-11-10 when vLLM main moved to ``cu132``),
the index keeps serving its last wheel, so ``pip install vllm`` from it still
succeeds and an image silently ships an old vLLM. This check turns that into a
build failure.

The age is the ``Last-Modified`` header of ``<index>/vllm/``, which the vLLM
wheel uploader rewrites whenever it publishes a new nightly.

Usage: ``check_vllm_nightly_age.py <index_url> <max_age_days>``
(``max_age_days`` of 0 disables the check). Exits 1 when the index is older.
"""

# Future
from __future__ import annotations

# Standard
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import sys
import urllib.request


def index_last_modified(index_url: str) -> datetime:
    """Return when the ``vllm`` project page of a wheel index last changed.

    Args:
        index_url: Base URL of a PEP 503 index, e.g.
            ``https://wheels.vllm.ai/nightly/cu130``.

    Returns:
        The page's ``Last-Modified`` time, timezone-aware (UTC).

    Raises:
        urllib.error.URLError: The page cannot be fetched.
        KeyError: The server sent no ``Last-Modified`` header.
    """
    url = index_url.rstrip("/") + "/vllm/"
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        return parsedate_to_datetime(response.headers["Last-Modified"])


def main(argv: list[str]) -> int:
    """Check the index age and report the result.

    Args:
        argv: ``[index_url, max_age_days]``.

    Returns:
        0 if the index is fresh enough or the check is disabled, 1 otherwise.
    """
    index_url, max_age_days = argv[0], int(argv[1])
    if max_age_days <= 0:
        print(f"vLLM nightly age check disabled for {index_url}")
        return 0
    last_modified = index_last_modified(index_url)
    age_days = (datetime.now(timezone.utc) - last_modified).days
    print(f"vLLM nightly index {index_url} last updated {last_modified:%Y-%m-%d}")
    if age_days > max_age_days:
        print(
            f"ERROR: {index_url} is {age_days} days old (limit {max_age_days}). "
            "vLLM has likely stopped publishing nightlies for this CUDA "
            "variant; see https://wheels.vllm.ai/nightly/ for the current "
            "ones. Pin a vLLM release with VLLM_VERSION=<version>, switch "
            "CUDA_VERSION, or pass VLLM_NIGHTLY_MAX_AGE_DAYS=0 to accept "
            "the old wheel.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
