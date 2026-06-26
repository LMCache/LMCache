# SPDX-License-Identifier: Apache-2.0
"""Pure, dependency-free bucket routing for the S3 L2 adapter (multi-tenant isolation).

Kept import-free (no awscrt / lmcache) so the routing logic is unit-testable in isolation and
trivial to review. The adapter delegates here.
"""
from __future__ import annotations

# Standard
import base64
import json
import re

_SANITIZE = re.compile(r"[^a-z0-9-]+")


def sanitize_salt(salt: str) -> str:
    """cache_salt → an S3-bucket-name-safe fragment (lowercase [a-z0-9-], trimmed)."""
    return _SANITIZE.sub("-", (salt or "").lower()).strip("-")


def salt_of_key(key_str: str) -> str:
    """Extract the cache_salt from a stored object key.
    Key format: ``<model>@<rank>@<group>@<hash>[@<cache_salt>]`` ('@' barred in model & salt)."""
    parts = (key_str or "").split("@")
    return parts[4] if len(parts) >= 5 else ""


def resolve_bucket_host(
    key_str: str,
    *,
    base_endpoint: str,
    mode: str = "single",
    template: str = "{base}-{salt}",
) -> str:
    """Virtual-hosted S3 Host for a key.

    - mode 'single' (default = upstream behavior): always the configured base bucket.
    - mode 'per_cache_salt': a *salted* key routes to its own bucket named by ``template``
      (``{salt}`` = sanitized cache_salt, ``{base}`` = base bucket name). Unsalted keys, and any
      salt that sanitizes to empty, fall back to the base bucket.

    base_endpoint is virtual-hosted: ``<bucket>.s3.<region>.amazonaws.com``.
    """
    if mode != "per_cache_salt":
        return base_endpoint
    safe = sanitize_salt(salt_of_key(key_str))
    if not safe:
        return base_endpoint
    base_bucket, _, suffix = base_endpoint.partition(".")
    if not suffix:
        return base_endpoint
    bucket = template.format(salt=safe, base=base_bucket)
    return f"{bucket}.{suffix}"


# ---- per-bucket listing / eviction support ---------------------------------
def bucket_hosts(
    base_endpoint: str,
    seen_salts: set[str],
    mode: str = "single",
    template: str = "{base}-{salt}",
) -> list[str]:
    """Ordered list of bucket Hosts the adapter must list for eviction: the base bucket plus
    one per observed tenant salt (per_cache_salt mode). 'single' mode → just the base."""
    if mode != "per_cache_salt":
        return [base_endpoint]
    base_bucket, _, suffix = base_endpoint.partition(".")
    hosts = [base_endpoint]
    for salt in sorted(s for s in seen_salts if s):
        h = (template.format(salt=salt, base=base_bucket) + "." + suffix) if suffix else base_endpoint
        if h not in hosts:
            hosts.append(h)
    return hosts


def encode_cursor(bucket_idx: int, token: str | None) -> str:
    """Encode a (bucket_index, in-bucket continuation token) pair into one opaque cursor."""
    return base64.urlsafe_b64encode(json.dumps([bucket_idx, token]).encode()).decode()


def decode_cursor(cursor: str | None) -> tuple[int, str | None]:
    """Cross-bucket cursor → (bucket_index, in-bucket continuation token). None/garbage → (0, None)."""
    if not cursor:
        return 0, None
    try:
        i, tok = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
        return int(i), tok
    except Exception:
        return 0, None
