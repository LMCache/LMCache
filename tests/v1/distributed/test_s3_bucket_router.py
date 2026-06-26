# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the per-cache_salt S3 bucket router (no awscrt/lmcache deps needed)."""
from lmcache.v1.distributed.l2_adapters import s3_bucket_router as r

BASE = "kv-cache.s3.us-east-1.amazonaws.com"


def test_sanitize():
    assert r.sanitize_salt("tenant:a") == "tenant-a"
    assert r.sanitize_salt("ORG:Big_Co!") == "org-big-co"
    assert r.sanitize_salt("") == ""
    assert r.sanitize_salt("user:42/x") == "user-42-x"


def test_salt_of_key():
    assert r.salt_of_key("model@01@00@deadbeef@tenant:a") == "tenant:a"
    assert r.salt_of_key("model@01@00@deadbeef") == ""          # unsalted
    assert r.salt_of_key("") == ""


def test_single_mode_is_upstream():
    # default mode never changes the host (PR-safe default)
    assert r.resolve_bucket_host("m@1@0@h@tenant:a", base_endpoint=BASE) == BASE
    assert r.resolve_bucket_host("m@1@0@h", base_endpoint=BASE) == BASE


def test_per_salt_routes_to_own_bucket():
    h = r.resolve_bucket_host("m@1@0@h@tenant:a", base_endpoint=BASE,
                              mode="per_cache_salt", template="kv-cache-{salt}")
    assert h == "kv-cache-tenant-a.s3.us-east-1.amazonaws.com"


def test_per_salt_distinct_tenants_distinct_buckets():
    a = r.resolve_bucket_host("m@1@0@h@tenant:a", base_endpoint=BASE, mode="per_cache_salt",
                              template="kv-cache-{salt}")
    b = r.resolve_bucket_host("m@1@0@h@tenant:b", base_endpoint=BASE, mode="per_cache_salt",
                              template="kv-cache-{salt}")
    assert a != b
    assert "tenant-a" in a and "tenant-b" in b


def test_per_salt_unsalted_uses_base():
    # unsalted KV must never leak into a tenant bucket
    assert r.resolve_bucket_host("m@1@0@h", base_endpoint=BASE, mode="per_cache_salt",
                                 template="kv-cache-{salt}") == BASE


def test_base_template_placeholder():
    h = r.resolve_bucket_host("m@1@0@h@tenant:9", base_endpoint=BASE, mode="per_cache_salt",
                              template="{base}-{salt}")
    assert h == "kv-cache-tenant-9.s3.us-east-1.amazonaws.com"


def test_bucket_hosts_single():
    assert r.bucket_hosts(BASE, {"tenant-1"}, mode="single") == [BASE]


def test_bucket_hosts_per_salt():
    hosts = r.bucket_hosts(BASE, {"tenant-a", "tenant-b"}, mode="per_cache_salt",
                           template="kv-cache-{salt}")
    assert hosts[0] == BASE
    assert any("tenant-a" in h for h in hosts) and any("tenant-b" in h for h in hosts)
    assert len(hosts) == 3


def test_cursor_roundtrip():
    for i, tok in [(0, None), (2, "abc=="), (5, "x/y+z")]:
        assert r.decode_cursor(r.encode_cursor(i, tok)) == (i, tok)
    assert r.decode_cursor(None) == (0, None)
    assert r.decode_cursor("garbage!!") == (0, None)
