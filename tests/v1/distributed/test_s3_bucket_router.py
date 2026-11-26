# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the per-cache_salt S3 bucket router (no deps)."""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters import s3_bucket_router as r

BASE = "kv-cache.s3.us-east-1.amazonaws.com"
SUFFIX = ".s3.us-east-1.amazonaws.com"


def _host(salt: str, template: str = "kv-cache-{salt}") -> str:
    return r.resolve_bucket_host(
        f"m@1@0@h@{salt}",
        base_endpoint=BASE,
        mode="per_cache_salt",
        template=template,
    )


def test_sanitize():
    assert r.sanitize_salt("tenant:a") == "tenant-a"
    assert r.sanitize_salt("ORG:Big_Co!") == "org-big-co"
    assert r.sanitize_salt("") == ""
    assert r.sanitize_salt("user:42/x") == "user-42-x"


def test_salt_of_key():
    assert r.salt_of_key("model@01@00@deadbeef@tenant:a") == "tenant:a"
    assert r.salt_of_key("model@01@00@deadbeef@tenant:a@tags") == "tenant:a"
    assert r.salt_of_key("model@01@00@deadbeef") == ""  # unsalted
    assert r.salt_of_key("model@01@00@deadbeef@@tags") == ""  # tagged, unsalted
    assert r.salt_of_key("") == ""


def test_single_mode_is_upstream():
    # default mode never changes the host (PR-safe default)
    assert r.resolve_bucket_host("m@1@0@h@tenant:a", base_endpoint=BASE) == BASE
    assert r.resolve_bucket_host("m@1@0@h", base_endpoint=BASE) == BASE


def test_per_salt_routes_to_own_bucket():
    digest = r.salt_hash("tenant:a")
    assert len(digest) == r.SALT_HASH_LEN
    assert _host("tenant:a") == f"kv-cache-tenant-a-{digest}{SUFFIX}"


@pytest.mark.parametrize(
    "a, b",
    [
        ("tenant:a", "tenant/a"),  # punctuation differs
        ("tenant:a", "Tenant:A"),  # case differs
        ("tenant:a", "tenant-a"),  # sanitizes to the same readable part
        ("a_b", "a.b"),
    ],
)
def test_salts_with_the_same_readable_part_get_distinct_buckets(a, b):
    """Regression: sanitizing alone collapsed these pairs into one bucket."""
    assert r.sanitize_salt(a) == r.sanitize_salt(b)
    assert _host(a) != _host(b)


def test_same_salt_is_stable():
    assert _host("tenant:a") == _host("tenant:a")


def test_per_salt_unsalted_uses_base():
    # unsalted KV must never leak into a tenant bucket
    assert (
        r.resolve_bucket_host(
            "m@1@0@h",
            base_endpoint=BASE,
            mode="per_cache_salt",
            template="kv-cache-{salt}",
        )
        == BASE
    )


def test_salt_with_no_readable_part_uses_hash_only():
    digest = r.salt_hash("!!!")
    assert _host("!!!") == f"kv-cache-{digest}{SUFFIX}"


def test_long_salt_is_truncated_to_a_legal_name():
    """cache_salt may be 128 characters; bucket names may not exceed 63."""
    salt = "tenant-" + "x" * 121
    name = r.bucket_of_host(_host(salt))
    assert len(name) <= r.MAX_BUCKET_NAME_LEN
    assert r.is_valid_bucket_name(name)
    assert name.endswith(r.salt_hash(salt))
    # Truncation never makes two long salts collide.
    other = r.bucket_of_host(_host(salt[:-1] + "y"))
    assert name != other


def test_reserved_prefix_falls_back_to_hash_only():
    name = r.tenant_bucket_name("xn--evil", "{salt}", "kv-cache")
    assert name == r.salt_hash("xn--evil")
    assert r.is_valid_bucket_name(name)


def test_base_template_placeholder():
    digest = r.salt_hash("tenant:9")
    assert _host("tenant:9", "{base}-{salt}") == f"kv-cache-tenant-9-{digest}{SUFFIX}"


@pytest.mark.parametrize(
    "template",
    [
        "{tenant}",  # unknown placeholder
        "kv-cache",  # no {salt}
        "{salt}-{salt}",  # {salt} twice
        "Upper-{salt}",  # illegal character
        "x" * 52 + "-{salt}",  # no room for the hash
    ],
)
def test_validate_template_rejects(template):
    with pytest.raises(ValueError):
        r.validate_template(template, "kv-cache")


def test_validate_template_accepts_defaults():
    r.validate_template(r.DEFAULT_TEMPLATE, "kv-cache")
    r.validate_template("kv-cache-{salt}", "kv-cache")


def test_bucket_hosts():
    assert r.bucket_hosts(BASE, set()) == [BASE]
    hosts = r.bucket_hosts(BASE, {"kv-cache-b-1", "kv-cache-a-2"})
    assert hosts == [BASE, f"kv-cache-a-2{SUFFIX}", f"kv-cache-b-1{SUFFIX}"]


def test_cursor_roundtrip():
    for bucket, tok in [("kv-cache", None), ("kv-cache-a-1", "abc=="), ("b", "x/y+z")]:
        assert r.decode_cursor(r.encode_cursor(bucket, tok)) == (bucket, tok)
    assert r.decode_cursor(None) == (None, None)
    assert r.decode_cursor("garbage!!") == (None, None)
    # Index-based cursors from an earlier revision restart the walk.
    assert r.decode_cursor(r.encode_cursor(2, None)) == (None, None)  # type: ignore[arg-type]


@pytest.mark.parametrize("template", ["{base}-{salt}", "{salt}", "kv-{salt}-x"])
def test_pattern_matches_exactly_the_generated_names(template):
    pattern = r.tenant_bucket_pattern(template, "kv-cache")
    prefix = r.tenant_bucket_prefix(template, "kv-cache")
    for salt in ("tenant:a", "!!!", "T" * 128, "xn--evil"):
        name = r.tenant_bucket_name(salt, template, "kv-cache")
        assert name.startswith(prefix)
        assert pattern.match(name), name
    # Unrelated buckets sharing the prefix are not tenant buckets.
    for other in ("kv-cache", prefix + "logs", prefix + "backup-2024"):
        assert not pattern.match(other), other
