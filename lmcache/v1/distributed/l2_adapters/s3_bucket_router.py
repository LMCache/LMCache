# SPDX-License-Identifier: Apache-2.0
"""Pure bucket routing for the S3 L2 adapter (multi-tenant isolation).

Kept import-free (no awscrt / lmcache) so routing is unit-testable in
isolation and trivial to review. The adapter delegates here.

Tenant bucket names
-------------------
In ``per_cache_salt`` mode a salted key goes to the bucket
``template.format(base=<base bucket>, salt=<token>)``. The token is
``<readable>-<hash>``:

- ``<hash>`` is the first 12 hex digits of SHA-256 over the *original*
  ``cache_salt``. It is what keeps tenants apart: salts that differ only in
  case or punctuation (``tenant:a``, ``tenant/a``, ``Tenant:A``) map to
  different buckets.
- ``<readable>`` is the salt lowercased with every run of characters outside
  ``[a-z0-9-]`` replaced by ``-``, truncated so the whole name fits S3's
  63-character limit. It is only there to make bucket names recognizable; it
  can be empty, in which case the token is the hash alone.
"""

# Future
from __future__ import annotations

# Standard
from string import Formatter
import base64
import hashlib
import json
import re

DEFAULT_TEMPLATE = "{base}-{salt}"

#: Hex digits of SHA-256(cache_salt) kept in every tenant bucket name.
SALT_HASH_LEN = 12

MAX_BUCKET_NAME_LEN = 63

_SANITIZE = re.compile(r"[^a-z0-9-]+")
_BUCKET_NAME = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_IPV4 = re.compile(r"^\d+\.\d+\.\d+\.\d+$")
# Prefixes and suffixes S3 reserves for other bucket types.
_RESERVED_PREFIXES = ("xn--", "sthree-", "amzn-s3-demo-")
_RESERVED_SUFFIXES = ("-s3alias", "--ol-s3", "--x-s3", "--table-s3")
_TEMPLATE_FIELDS = frozenset({"base", "salt"})
# Host suffixes of the AWS S3 virtual-hosted endpoints, after "<bucket>.":
# s3.<region>.amazonaws.com, s3-<region>.amazonaws.com (legacy dash form),
# s3.dualstack.<region>.amazonaws.com, the global s3.amazonaws.com and
# s3-external-1.amazonaws.com (both us-east-1), and the .com.cn partition.
_AWS_S3_SUFFIX = re.compile(
    r"^s3(?:-external-1|\.dualstack)?"
    r"(?:[.-](?P<region>[a-z]{2}(?:-[a-z]+)+-\d+))?"
    r"\.amazonaws\.com(?:\.cn)?$"
)
#: Region the regionless S3 endpoints (s3.amazonaws.com) resolve to.
GLOBAL_ENDPOINT_REGION = "us-east-1"


class VirtualHostedEndpoint:
    """An AWS S3 endpoint of the form ``<bucket>.<s3 suffix>``."""

    def __init__(self, bucket: str, suffix: str, region: str | None):
        self.bucket = bucket
        self.suffix = suffix
        #: Region named by the host, or None for the regionless endpoints.
        self.region = region

    @property
    def effective_region(self) -> str:
        return self.region or GLOBAL_ENDPOINT_REGION


def parse_virtual_hosted_endpoint(endpoint: str) -> VirtualHostedEndpoint:
    """Parse the whole endpoint for ``per_cache_salt`` mode.

    Accepts only ``[s3://]<bucket>.<AWS S3 suffix>`` with a dot-free,
    lowercase bucket label: the tenant bucket replaces that label, so it must
    be the entire first label of the host.

    Raises:
        ValueError: a path-style endpoint (no bucket label, or a path), a
            port, credentials or a query, a dotted or illegal bucket name, or
            a host that is not an AWS S3 endpoint.
    """
    host = endpoint.removeprefix("s3://")
    if not host or any(c in host for c in "/?#@:"):
        raise ValueError(
            f"{endpoint!r} is not a bare virtual-hosted host: no scheme other "
            "than s3://, path, port, query or credentials is allowed"
        )
    if host != host.lower():
        raise ValueError(f"{endpoint!r} must be lowercase")
    if _AWS_S3_SUFFIX.match(host):
        raise ValueError(
            f"{endpoint!r} is a path-style endpoint with no bucket label; use "
            "'<bucket>.s3.<region>.amazonaws.com'"
        )
    pos = host.find(".")
    while pos != -1:
        m = _AWS_S3_SUFFIX.match(host[pos + 1 :])
        if m:
            bucket = host[:pos]
            if "." in bucket:
                raise ValueError(
                    f"bucket {bucket!r} in {endpoint!r} contains dots, which "
                    "per_cache_salt mode does not allow"
                )
            if not is_valid_bucket_name(bucket):
                raise ValueError(f"{bucket!r} in {endpoint!r} is not a bucket name")
            return VirtualHostedEndpoint(bucket, host[pos + 1 :], m.group("region"))
        pos = host.find(".", pos + 1)
    raise ValueError(
        f"{endpoint!r} is not an AWS S3 virtual-hosted endpoint "
        "('<bucket>.s3.<region>.amazonaws.com')"
    )


def sanitize_salt(salt: str) -> str:
    """cache_salt → readable bucket-name fragment (lowercase [a-z0-9-], trimmed).

    Lossy on purpose: it only labels a bucket. Isolation comes from
    :func:`salt_hash`.
    """
    return _SANITIZE.sub("-", (salt or "").lower()).strip("-")


def salt_hash(salt: str) -> str:
    """Stable digest of the original cache_salt (lowercase hex)."""
    return hashlib.sha256(salt.encode("utf-8")).hexdigest()[:SALT_HASH_LEN]


def salt_of_key(key_str: str) -> str:
    """Extract the cache_salt from a stored object key.
    Key format: ``<model>@<rank>@<group>@<hash>[@<cache_salt>[@<tags>]]``
    ('@' barred in model & salt).
    """
    parts = (key_str or "").split("@")
    return parts[4] if len(parts) >= 5 else ""


def is_valid_bucket_name(name: str) -> bool:
    """Whether ``name`` is a legal general-purpose S3 bucket name."""
    return (
        bool(_BUCKET_NAME.match(name))
        and ".." not in name
        and not _IPV4.match(name)
        and not name.startswith(_RESERVED_PREFIXES)
        and not name.endswith(_RESERVED_SUFFIXES)
    )


def validate_template(template: str, base_bucket: str) -> None:
    """Check that ``template`` yields legal names for every salt.

    Raises:
        ValueError: unknown or repeated placeholders, no ``{salt}``, no room
            for the salt hash within 63 characters, or an illegal name.
    """
    fields = [f for _, f, _, _ in Formatter().parse(template) if f is not None]
    unknown = set(fields) - _TEMPLATE_FIELDS
    if unknown:
        raise ValueError(f"unknown placeholder(s) {sorted(unknown)}")
    if fields.count("salt") != 1:
        raise ValueError("must contain {salt} exactly once")
    fixed = len(template.format(base=base_bucket, salt=""))
    if fixed + SALT_HASH_LEN > MAX_BUCKET_NAME_LEN:
        raise ValueError(
            f"leaves no room for the {SALT_HASH_LEN}-character salt hash "
            f"within {MAX_BUCKET_NAME_LEN} characters"
        )
    for sample in ("x", "tenant:a", "tenant.a", "T" * 128):
        name = tenant_bucket_name(sample, template, base_bucket)
        if not is_valid_bucket_name(name):
            raise ValueError(f"produces an invalid bucket name {name!r}")
        if "." in name:
            # The sanitizer never emits dots, so one can only come from the
            # template or the base bucket.
            raise ValueError(f"produces a dotted bucket name {name!r}")


def tenant_bucket_name(
    salt: str, template: str = DEFAULT_TEMPLATE, base_bucket: str = ""
) -> str:
    """Bucket name for a non-empty cache_salt (see the module docstring)."""
    digest = salt_hash(salt)
    fixed = len(template.format(base=base_bucket, salt=""))
    room = MAX_BUCKET_NAME_LEN - fixed - SALT_HASH_LEN - 1  # 1 = "-" separator
    readable = sanitize_salt(salt)[: max(room, 0)].strip("-")
    if readable:
        name = template.format(base=base_bucket, salt=f"{readable}-{digest}")
        if is_valid_bucket_name(name):
            return name
    # No readable part, or it formed a reserved/illegal name: hash only.
    return template.format(base=base_bucket, salt=digest)


def resolve_bucket_host(
    key_str: str,
    *,
    base_endpoint: str,
    mode: str = "single",
    template: str = DEFAULT_TEMPLATE,
) -> str:
    """Virtual-hosted S3 Host for a key.

    - mode 'single' (default = upstream behavior): always the configured base bucket.
    - mode 'per_cache_salt': a *salted* key routes to its own bucket
      (:func:`tenant_bucket_name`). Unsalted keys stay in the base bucket.

    base_endpoint is virtual-hosted: ``<bucket>.s3.<region>.amazonaws.com``.
    """
    if mode != "per_cache_salt":
        return base_endpoint
    salt = salt_of_key(key_str)
    if not salt:
        return base_endpoint
    base_bucket, _, suffix = base_endpoint.partition(".")
    if not suffix:
        return base_endpoint
    return f"{tenant_bucket_name(salt, template, base_bucket)}.{suffix}"


def bucket_of_host(host: str) -> str:
    """Bucket name of a virtual-hosted Host."""
    return host.split(".", 1)[0]


# ---- discovery -----------------------------------------------------------------
# What a {salt} token can look like: an optional readable part ending in "-",
# then the salt hash.
_TOKEN_RE = r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?-)?[0-9a-f]{%d}" % SALT_HASH_LEN


def _template_parts(template: str, base_bucket: str) -> tuple[str, str]:
    marker = "\x00"
    before, _, after = template.format(base=base_bucket, salt=marker).partition(marker)
    return before, after


def tenant_bucket_prefix(template: str, base_bucket: str) -> str:
    """Literal prefix every tenant bucket name starts with (may be empty)."""
    return _template_parts(template, base_bucket)[0]


def tenant_bucket_pattern(template: str, base_bucket: str) -> re.Pattern[str]:
    """Regex matching exactly the names :func:`tenant_bucket_name` produces."""
    before, after = _template_parts(template, base_bucket)
    return re.compile(f"^{re.escape(before)}{_TOKEN_RE}{re.escape(after)}$")


# ---- per-bucket listing support ---------------------------------------------
def bucket_hosts(base_endpoint: str, tenant_buckets: set[str]) -> list[str]:
    """Hosts to list, in cursor order: the base bucket, then tenant buckets
    sorted by name."""
    _, _, suffix = base_endpoint.partition(".")
    hosts = [base_endpoint]
    if not suffix:
        return hosts
    for bucket in sorted(tenant_buckets):
        host = f"{bucket}.{suffix}"
        if host != base_endpoint:
            hosts.append(host)
    return hosts


def encode_cursor(bucket: str, token: str | None) -> str:
    """Encode a bucket name and its continuation token as one opaque cursor."""
    return base64.urlsafe_b64encode(json.dumps([bucket, token]).encode()).decode()


def decode_cursor(cursor: str | None) -> tuple[str | None, str | None]:
    """Decode cursor → (bucket name, continuation token).

    ``(None, None)`` means "start from the first bucket" and is also what a
    malformed cursor decodes to.
    """
    if not cursor:
        return None, None
    try:
        bucket, tok = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
    except Exception:
        return None, None
    if not isinstance(bucket, str) or not (tok is None or isinstance(tok, str)):
        return None, None
    return bucket, tok
