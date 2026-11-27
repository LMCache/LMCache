# SPDX-License-Identifier: Apache-2.0
"""Adapter-level tests for ``s3_bucket_mode: per_cache_salt``.

``awscrt.s3.S3Request`` is replaced by an in-memory fake that routes each
request by its ``Host`` header to a per-bucket store, and the boto3 bucket
client by a fake that shares the same set of buckets. No network required.
"""

# Standard
from concurrent.futures import Future as ConcurrentFuture
from types import SimpleNamespace
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlsplit
import logging
import select
import threading
import xml.sax.saxutils as saxutils

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters import s3_bucket_router as router
from lmcache.v1.distributed.l2_adapters import s3_l2_adapter as s3mod
from lmcache.v1.distributed.l2_adapters.s3_l2_adapter import (
    S3L2Adapter,
    S3L2AdapterConfig,
    _object_key_to_string,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

# Test helpers
from tests.v1.distributed.utils import wait_until

BASE = "test-bucket"
TEMPLATE = router.DEFAULT_TEMPLATE
OWNER = "123456789012"


def endpoint(region: str = "us-east-1") -> str:
    return f"{BASE}.s3.{region}.amazonaws.com"


def bucket_for(salt: str) -> str:
    return router.tenant_bucket_name(salt, TEMPLATE, BASE)


# =============================================================================
# Fakes: one "world" of buckets shared by the data plane and the boto3 client
# =============================================================================


class _ClientError(Exception):
    """Shape of botocore's ClientError that the adapter reads."""

    def __init__(self, code: str):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _World:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.objects: dict[str, dict[str, bytes]] = {BASE: {}}  # owned + base
        self.foreign: set[str] = set()  # names owned by another account
        self.lifecycle: dict[str, Any] = {}
        self.created: list[str] = []
        self.requests: list[tuple[str, str, dict[str, str]]] = []  # method, bucket
        self.lifecycle_kwargs: list[dict] = []

    def add_bucket(self, name: str, **objects: bytes) -> None:
        with self.lock:
            self.objects.setdefault(name, {}).update(objects)

    def requests_to(self, bucket: str) -> list[str]:
        with self.lock:
            return [m for m, b, _ in self.requests if b == bucket]


WORLD = _World()


class _FakeMgmt:
    """boto3 S3 client subset used for tenant buckets."""

    def __init__(self, world: _World, region: str, page_size: int = 2):
        self.world = world
        self.region = region
        self.page_size = page_size
        self.hide_from_listing: set[str] = set()  # simulate ListBuckets lag
        members = {"Prefix": 1, "BucketRegion": 1, "ContinuationToken": 1}
        self.meta = SimpleNamespace(
            service_model=SimpleNamespace(
                operation_model=lambda _name: SimpleNamespace(
                    input_shape=SimpleNamespace(members=members)
                )
            )
        )

    def list_buckets(self, Prefix: str = "", ContinuationToken: str = "", **_kw):
        with self.world.lock:
            names = sorted(
                n
                for n in self.world.objects
                if n.startswith(Prefix) and n not in self.hide_from_listing
            )
        start = int(ContinuationToken or 0)
        page = names[start : start + self.page_size]
        resp: dict[str, Any] = {"Buckets": [{"Name": n} for n in page]}
        if start + self.page_size < len(names):
            resp["ContinuationToken"] = str(start + self.page_size)
        return resp

    def create_bucket(self, Bucket: str, CreateBucketConfiguration=None):
        with self.world.lock:
            if Bucket in self.world.foreign:
                raise _ClientError("BucketAlreadyExists")
            if Bucket in self.world.objects:
                if self.region == "us-east-1":
                    return {}  # legacy 200 for a bucket the caller owns
                raise _ClientError("BucketAlreadyOwnedByYou")
            self.world.objects[Bucket] = {}
            self.world.created.append(Bucket)
        return {}

    def put_bucket_lifecycle_configuration(self, Bucket, LifecycleConfiguration, **kw):
        with self.world.lock:
            self.world.lifecycle[Bucket] = LifecycleConfiguration
            self.world.lifecycle_kwargs.append(kw)


def _list_xml(objects: dict[str, bytes], prefix, max_keys, token) -> bytes:
    keys = sorted(k for k in objects if not prefix or k.startswith(prefix))
    if token is not None:
        keys = [k for k in keys if k > token]
    page, more = keys[:max_keys], len(keys) > max_keys
    parts = ['<?xml version="1.0"?><ListBucketResult>']
    for k in page:
        parts.append(
            f"<Contents><Key>{saxutils.escape(k)}</Key>"
            f"<Size>{len(objects[k])}</Size></Contents>"
        )
    if more:
        parts.append(f"<NextContinuationToken>{page[-1]}</NextContinuationToken>")
    parts.append("</ListBucketResult>")
    return "".join(parts).encode()


class _FakeS3Request:
    """Routes by Host to WORLD; settles finished_future like the CRT."""

    def __init__(
        self,
        *,
        request,
        on_body=None,
        on_done=None,
        on_headers=None,
        operation_name=None,
        **_kwargs,
    ):
        self.finished_future: ConcurrentFuture = ConcurrentFuture()
        headers = {k.lower(): v for k, v in request.headers}
        bucket = router.bucket_of_host(headers["host"])
        with WORLD.lock:
            WORLD.requests.append((request.method, bucket, headers))
            store = WORLD.objects.get(bucket)
        if store is None:  # NoSuchBucket
            if on_headers is not None:
                on_headers(404, [])
            self.finished_future.set_exception(RuntimeError("404 NoSuchBucket"))
            return
        method, path = request.method, request.path
        if method == "GET" and operation_name == "ListObjectsV2":
            qs = parse_qs(urlsplit(path).query)
            xml = _list_xml(
                store,
                qs.get("prefix", [None])[0],
                int(qs.get("max-keys", ["1000"])[0]),
                qs.get("continuation-token", [None])[0],
            )
            if on_headers is not None:
                on_headers(200, [])
            if on_body is not None:
                on_body(xml, 0)
            self.finished_future.set_result(None)
            return
        key = unquote(path.lstrip("/"))
        if method == "PUT":
            body = request.body_stream
            body.seek(0)
            with WORLD.lock:
                store[key] = bytes(body.read())
            self.finished_future.set_result(None)
        elif method in ("GET", "HEAD"):
            with WORLD.lock:
                data = store.get(key)
            if data is None:
                if on_headers is not None:
                    on_headers(404, [])
                self.finished_future.set_exception(RuntimeError("404"))
                return
            if on_headers is not None:
                on_headers(200, [("content-length", str(len(data)))])
            if method == "GET" and on_body is not None:
                on_body(data, 0)
            self.finished_future.set_result(None)
        elif method == "DELETE":
            with WORLD.lock:
                store.pop(key, None)
            if on_headers is not None:
                on_headers(204, [])
            self.finished_future.set_result(None)


class _FakeHttpRequest:
    def __init__(self, method, path, headers, body_stream=None):
        self.method = method
        self.path = path
        self.headers = headers
        self.body_stream = body_stream


MGMT: dict[str, _FakeMgmt] = {}


@pytest.fixture(autouse=True)
def fake_s3(monkeypatch):
    global WORLD
    WORLD = _World()
    MGMT.clear()
    monkeypatch.setattr(s3mod.s3, "S3Request", _FakeS3Request)
    monkeypatch.setattr(s3mod, "HttpRequest", _FakeHttpRequest)
    monkeypatch.setattr(
        s3mod,
        "_make_credentials_provider",
        lambda _config: s3mod.auth.AwsCredentialsProvider.new_static("k", "s"),
    )

    def new_mgmt_client(self):
        client = _FakeMgmt(WORLD, self._region)
        MGMT["client"] = client
        return client

    monkeypatch.setattr(S3L2Adapter, "_new_mgmt_client", new_mgmt_client)
    yield


# =============================================================================
# Helpers
# =============================================================================


def make_adapter(region: str = "us-east-1", **overrides) -> S3L2Adapter:
    config = S3L2AdapterConfig(
        s3_endpoint=f"s3://{endpoint(region)}",
        s3_region=region,
        s3_prefer_http2=False,
        s3_num_io_threads=1,
        s3_bucket_mode="per_cache_salt",
        **overrides,
    )
    return S3L2Adapter(config)


@pytest.fixture
def adapters():
    made: list[S3L2Adapter] = []

    def make(**kw) -> S3L2Adapter:
        a = make_adapter(**kw)
        made.append(a)
        return a

    yield make
    for a in made:
        a.close()


def key(chunk: int, salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk),
        model_name="m",
        kv_rank=0,
        cache_salt=salt,
    )


def obj(fill: float = 1.0, size: int = 8) -> TensorMemoryObj:
    raw = torch.full((size,), fill, dtype=torch.float32)
    meta = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.float32,
        address=0,
        phy_size=size * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw, meta, parent_allocator=None)


def _wait_fd(fd: int) -> None:
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    assert poll.poll(5000), "adapter did not signal"
    try:
        consume_fd(fd)
    except BlockingIOError:
        pass


def store(adapter: S3L2Adapter, keys: list[ObjectKey]) -> bool:
    task = adapter.submit_store_task(keys, [obj() for _ in keys])
    _wait_fd(adapter.get_store_event_fd())
    return adapter.pop_completed_store_tasks()[task].is_successful()


def lookup(adapter: S3L2Adapter, keys: list[ObjectKey]) -> list[bool]:
    task = adapter.submit_lookup_and_lock_task(keys, {})
    _wait_fd(adapter.get_lookup_and_lock_event_fd())
    bitmap = adapter.query_lookup_and_lock_result(task)
    assert bitmap is not None
    adapter.submit_unlock(keys)
    return [bitmap.test(i) for i in range(len(keys))]


def load(adapter: S3L2Adapter, keys: list[ObjectKey]) -> list[bool]:
    task = adapter.submit_load_task(keys, [obj(0.0) for _ in keys])
    _wait_fd(adapter.get_load_event_fd())
    bitmap = adapter.query_load_result(task)
    assert bitmap is not None
    return [bitmap.test(i) for i in range(len(keys))]


def settled(adapter: S3L2Adapter) -> bool:
    with adapter._lock:
        return not adapter._pending_buckets


def store_for_new_tenant(adapter: S3L2Adapter, keys: list[ObjectKey]) -> None:
    """First store fails while the bucket is provisioned; the next succeeds."""
    assert store(adapter, keys) is False
    assert wait_until(lambda: settled(adapter))
    assert store(adapter, keys) is True


def list_all(adapter: S3L2Adapter, page_size: int) -> list[ObjectKey]:
    out: list[ObjectKey] = []
    cursor: Optional[str] = None
    while True:
        page = adapter.list_l2_keys(page_size=page_size, cursor=cursor)
        out.extend(e.key.to_object_key() for e in page.entries)
        cursor = page.next_page_token
        if cursor is None:
            return out


def name(k: ObjectKey) -> str:
    return _object_key_to_string(k)


# =============================================================================
# Review point 1: lossy salt → bucket mapping
# =============================================================================


class TestDistinctSaltsDistinctBuckets:
    def test_salts_differing_only_in_punctuation_or_case_do_not_share(self, adapters):
        a = adapters()
        # "/" is not a legal cache_salt; these all sanitize to "tenant-a".
        salts = ["tenant:a", "tenant.a", "tenant_a", "Tenant:A"]
        for i, salt in enumerate(salts):
            store_for_new_tenant(a, [key(i, salt)])
        buckets = {bucket_for(s) for s in salts}
        assert len(buckets) == len(salts)
        for i, salt in enumerate(salts):
            assert set(WORLD.objects[bucket_for(salt)]) == {name(key(i, salt))}

    def test_unsalted_keys_stay_in_the_base_bucket(self, adapters):
        a = adapters()
        assert store(a, [key(1)]) is True
        assert set(WORLD.objects[BASE]) == {name(key(1))}
        assert WORLD.created == []


# =============================================================================
# Review point 2: tenant buckets after a restart; cursor stability
# =============================================================================


class TestDiscovery:
    def test_restart_lists_existing_tenant_buckets(self, adapters):
        salts = [f"tenant:{i}" for i in range(5)]  # > one ListBuckets page
        expected = set()
        for i, salt in enumerate(salts):
            k = key(i, salt)
            WORLD.add_bucket(bucket_for(salt), **{name(k): b"x"})
            expected.add(name(k))
        WORLD.add_bucket(f"{BASE}-logs", junk=b"x")  # shares the prefix only

        a = adapters()  # a fresh process: nothing used yet

        listed = {name(k) for k in list_all(a, page_size=3)}
        assert listed == expected
        assert f"{BASE}-logs" not in a._tenant_buckets
        # Discovered buckets are usable at once: a lookup hits on first try.
        assert lookup(a, [key(0, "tenant:0")]) == [True]
        assert WORLD.created == []

    def test_failed_startup_scan_is_retried_by_listing(self, adapters, monkeypatch):
        WORLD.add_bucket(bucket_for("t"), **{name(key(1, "t")): b"x"})
        real = s3mod._list_owned_buckets
        calls = {"n": 0}

        def flaky(*args):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("throttled")
            return real(*args)

        monkeypatch.setattr(s3mod, "_list_owned_buckets", flaky)
        a = adapters()
        assert a._tenant_buckets == set()
        assert {name(k) for k in list_all(a, page_size=10)} == {name(key(1, "t"))}


class TestCursorStability:
    def test_bucket_added_between_pages_does_not_shift_the_walk(self, adapters):
        # Pick three salts and order them by bucket name: early < middle < late.
        salts = sorted(("s1", "s2", "s3", "s4"), key=bucket_for)
        early, middle, late = salts[0], salts[1], salts[2]
        keys = {early: [key(i, early) for i in range(3)]}
        keys[late] = [key(10 + i, late) for i in range(3)]
        for salt, ks in keys.items():
            WORLD.add_bucket(bucket_for(salt), **{name(k): b"x" for k in ks})
        a = adapters()

        seen: list[ObjectKey] = []
        page = a.list_l2_keys(page_size=4)  # all of early + one of late
        seen.extend(e.key.to_object_key() for e in page.entries)
        assert page.next_page_token is not None
        # A tenant bucket that sorts before the cursor appears mid-walk.
        WORLD.add_bucket(bucket_for(middle), **{name(key(20, middle)): b"x"})
        a._mark_bucket_usable(bucket_for(middle))
        cursor = page.next_page_token
        while cursor is not None:
            page = a.list_l2_keys(page_size=4, cursor=cursor)
            seen.extend(e.key.to_object_key() for e in page.entries)
            cursor = page.next_page_token

        names_seen = [name(k) for k in seen]
        assert len(names_seen) == len(set(names_seen)), "a bucket was repeated"
        assert set(names_seen) == {name(k) for ks in keys.values() for k in ks}
        # The next full pass includes the new bucket.
        assert name(key(20, middle)) in {name(k) for k in list_all(a, 4)}

    def test_cursor_for_a_vanished_bucket_moves_on(self, adapters):
        salts = sorted(("s1", "s2"), key=bucket_for)
        for i, salt in enumerate(salts):
            WORLD.add_bucket(bucket_for(salt), **{name(key(i, salt)): b"x"})
        a = adapters()
        gone = router.encode_cursor(bucket_for(salts[0]) + "0", "stale-token")
        page = a.list_l2_keys(page_size=10, cursor=gone)
        assert [name(e.key.to_object_key()) for e in page.entries] == [
            name(key(1, salts[1]))
        ]


# =============================================================================
# Review point 3: a bucket name owned by another account
# =============================================================================


class TestForeignBucket:
    def test_foreign_bucket_is_never_read_or_written(self, adapters):
        WORLD.foreign.add(bucket_for("victim"))
        a = adapters()
        k = key(1, "victim")
        records: list[logging.LogRecord] = []
        handler = logging.Handler(logging.ERROR)
        handler.emit = records.append  # type: ignore[method-assign]
        s3mod.logger.addHandler(handler)
        try:
            assert store(a, [k]) is False
            assert wait_until(lambda: settled(a))
            assert bucket_for("victim") in a._foreign_buckets
            assert store(a, [k]) is False
            assert lookup(a, [k]) == [False]
            assert load(a, [k]) == [False]
            a.delete([k])
        finally:
            s3mod.logger.removeHandler(handler)
        assert WORLD.requests_to(bucket_for("victim")) == []
        assert bucket_for("victim") not in a._tenant_buckets
        owned_elsewhere = [r for r in records if "another AWS account" in r.msg]
        assert len(owned_elsewhere) == 1

    def test_already_owned_by_you_is_accepted(self, adapters):
        # Outside us-east-1, a concurrent creator makes create_bucket fail
        # with BucketAlreadyOwnedByYou; ListBuckets has not caught up yet.
        a = adapters(region="eu-west-1")
        WORLD.add_bucket(bucket_for("t"))
        MGMT["client"].hide_from_listing.add(bucket_for("t"))
        store_for_new_tenant(a, [key(1, "t")])
        assert bucket_for("t") not in WORLD.created
        assert bucket_for("t") not in WORLD.lifecycle  # not ours to change

    def test_expected_owner_header_on_tenant_requests_only(self, adapters):
        a = adapters(s3_expected_bucket_owner=OWNER)
        store_for_new_tenant(a, [key(1, "t")])
        assert store(a, [key(2)]) is True
        with WORLD.lock:
            by_bucket = {b: h for m, b, h in WORLD.requests if m == "PUT"}
        assert by_bucket[bucket_for("t")]["x-amz-expected-bucket-owner"] == OWNER
        assert "x-amz-expected-bucket-owner" not in by_bucket[BASE]
        assert WORLD.lifecycle_kwargs == [{"ExpectedBucketOwner": OWNER}]


# =============================================================================
# Review point 4: lifecycle only on a bucket this process created
# =============================================================================


class TestLifecycle:
    def test_new_bucket_gets_the_configured_ttl(self, adapters):
        a = adapters(s3_bucket_ttl_days=7)
        store_for_new_tenant(a, [key(1, "t")])
        assert WORLD.created == [bucket_for("t")]
        rules = WORLD.lifecycle[bucket_for("t")]["Rules"]
        assert rules[0]["Expiration"] == {"Days": 7}

    def test_existing_bucket_keeps_its_rules_in_us_east_1(self, adapters):
        a = adapters()
        # Created out of band after startup, so not discovered; create_bucket
        # would answer 200 here (us-east-1), so ownership must be checked first.
        WORLD.add_bucket(bucket_for("t"))
        WORLD.lifecycle[bucket_for("t")] = {"Rules": ["managed elsewhere"]}
        store_for_new_tenant(a, [key(1, "t")])
        assert WORLD.created == []
        assert WORLD.lifecycle[bucket_for("t")] == {"Rules": ["managed elsewhere"]}

    def test_ttl_zero_sets_no_rule(self, adapters):
        a = adapters(s3_bucket_ttl_days=0)
        store_for_new_tenant(a, [key(1, "t")])
        assert WORLD.created == [bucket_for("t")]
        assert WORLD.lifecycle == {}


# =============================================================================
# Reads never create; auto-create can be turned off
# =============================================================================


class TestProvisioningPath:
    def test_reads_never_create_a_bucket(self, adapters):
        a = adapters()
        k = key(1, "t")
        assert lookup(a, [k]) == [False]
        assert load(a, [k]) == [False]
        a.delete([k])
        assert wait_until(lambda: settled(a))
        assert WORLD.created == []
        assert bucket_for("t") not in WORLD.objects
        # A store may create it.
        store_for_new_tenant(a, [k])
        assert WORLD.created == [bucket_for("t")]
        assert lookup(a, [k]) == [True]

    def test_auto_create_off_never_creates(self, adapters):
        a = adapters(s3_bucket_auto_create=False)
        assert store(a, [key(1, "t")]) is False
        assert wait_until(lambda: settled(a))
        assert WORLD.created == []
        # A bucket created out of band is picked up (after the retry delay).
        WORLD.add_bucket(bucket_for("t"))
        a._bucket_retry_at.clear()
        store_for_new_tenant(a, [key(1, "t")])
        assert WORLD.created == []

    def test_deleted_tenant_bucket_reads_as_a_miss(self, adapters):
        WORLD.add_bucket(bucket_for("t"), **{name(key(1, "t")): b"x"})
        a = adapters()
        with WORLD.lock:
            del WORLD.objects[bucket_for("t")]
        assert lookup(a, [key(1, "t")]) == [False]


# =============================================================================
# Config
# =============================================================================


class TestConfig:
    def test_from_dict_reads_new_fields(self):
        cfg = S3L2AdapterConfig.from_dict(
            {
                "s3_endpoint": endpoint(),
                "s3_region": "us-east-1",
                "s3_bucket_mode": "per_cache_salt",
                "s3_bucket_template": "kv-{salt}",
                "s3_bucket_auto_create": False,
                "s3_bucket_ttl_days": 0,
                "s3_expected_bucket_owner": OWNER,
            }
        )
        assert cfg.s3_bucket_template == "kv-{salt}"
        assert cfg.s3_bucket_auto_create is False
        assert cfg.s3_bucket_ttl_days == 0
        assert cfg.s3_expected_bucket_owner == OWNER

    def test_defaults(self):
        cfg = S3L2AdapterConfig.from_dict(
            {"s3_endpoint": endpoint(), "s3_region": "us-east-1"}
        )
        assert cfg.s3_bucket_mode == "single"
        assert cfg.s3_bucket_auto_create is True
        assert cfg.s3_bucket_ttl_days == 30
        assert cfg.s3_expected_bucket_owner is None

    @pytest.mark.parametrize(
        "field, value",
        [
            ("s3_bucket_ttl_days", -1),
            ("s3_bucket_ttl_days", "30"),
            ("s3_bucket_auto_create", "yes"),
            ("s3_expected_bucket_owner", "12345"),
            ("s3_bucket_template", "{tenant}"),
            ("s3_bucket_template", "x" * 60 + "{salt}"),
            ("s3_bucket_mode", "per_tenant"),
        ],
    )
    def test_rejects(self, field, value):
        d = {
            "s3_endpoint": endpoint(),
            "s3_region": "us-east-1",
            "s3_bucket_mode": "per_cache_salt",
            field: value,
        }
        with pytest.raises(ValueError):
            S3L2AdapterConfig.from_dict(d)

    def test_per_salt_needs_a_virtual_hosted_endpoint(self):
        """A bucket-only endpoint would silently put every tenant in one
        bucket, so it is refused."""
        with pytest.raises(ValueError, match="virtual-hosted"):
            S3L2AdapterConfig.from_dict(
                {
                    "s3_endpoint": "s3://my-bucket",
                    "s3_region": "us-east-1",
                    "s3_bucket_mode": "per_cache_salt",
                }
            )
        # Single mode keeps accepting it.
        S3L2AdapterConfig.from_dict(
            {"s3_endpoint": "s3://my-bucket", "s3_region": "us-east-1"}
        )

    @pytest.mark.parametrize(
        "endpoint, region, match",
        [
            ("s3://s3.us-west-2.amazonaws.com", "us-west-2", "path-style"),
            ("s3.us-west-2.amazonaws.com/kv", "us-west-2", "path"),
            ("s3://my.kv.s3.us-west-2.amazonaws.com", "us-west-2", "dots"),
            ("s3://kv-cache.minio.local:9000", "us-west-2", "virtual-hosted"),
            ("s3://kv-cache.s3.us-west-2.amazonaws.com", "eu-west-1", "region"),
            ("s3://kv-cache.s3.amazonaws.com", "us-west-2", "region"),
        ],
    )
    def test_per_salt_rejects_ambiguous_endpoints(self, endpoint, region, match):
        with pytest.raises(ValueError, match=match):
            S3L2AdapterConfig(
                s3_endpoint=endpoint,
                s3_region=region,
                s3_bucket_mode="per_cache_salt",
            )

    def test_single_mode_endpoints_are_unchanged(self):
        """The stricter endpoint rules apply only to per_cache_salt mode."""
        for ep in ("s3://my.kv.s3.us-west-2.amazonaws.com", "minio.local:9000"):
            S3L2AdapterConfig(s3_endpoint=ep, s3_region="us-west-2")

    def test_help_lists_new_fields(self):
        text = S3L2AdapterConfig.help()
        for field in (
            "s3_bucket_mode",
            "s3_bucket_template",
            "s3_bucket_auto_create",
            "s3_bucket_ttl_days",
            "s3_expected_bucket_owner",
        ):
            assert field in text

    def test_single_mode_never_calls_bucket_apis(self, adapters, monkeypatch):
        cfg = S3L2AdapterConfig(
            s3_endpoint=f"s3://{endpoint()}",
            s3_region="us-east-1",
            s3_prefer_http2=False,
            s3_num_io_threads=1,
        )
        a = S3L2Adapter(cfg)
        try:
            assert store(a, [key(1, "tenant:a")]) is True
            assert set(WORLD.objects[BASE]) == {name(key(1, "tenant:a"))}
            assert "client" not in MGMT
        finally:
            a.close()
