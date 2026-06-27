# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the S3L2Adapter startup size-seed (feature a) and the
periodic LRU-index sidecar checkpoint/restore (feature b).

These tests exercise the *pure* parts of the seeding path without any
live AWS / awscrt dependency:

- ``LastModified`` parsing + ListObjectsV2 XML parsing with mtime,
- oldest-first ordering from a parsed listing,
- size-summing from a parsed listing,
- sidecar serialize/deserialize roundtrip,
- corrupt / partial / unknown-version sidecar -> treated as absent,
- policy population in LRU order via a fake eviction-policy listener.

The heavy native dependencies (``awscrt``, ``torch``,
``lmcache.native_storage_ops``) are stubbed in ``sys.modules`` so the
real module-level seeding helpers import and run.
"""

# Standard
from types import ModuleType
import sys

# ---------------------------------------------------------------------------
# Stub heavy deps so the adapter module imports without awscrt / torch.
# Only the attributes touched at import time need to exist.
# ---------------------------------------------------------------------------


def _install_stubs() -> None:
    if "torch" not in sys.modules:
        # A permissive torch stub: any ``torch.X`` attribute resolves to a
        # generic sentinel object. This satisfies the many import-time
        # default-arg lookups (``torch.dtype``, ``torch.float8_e4m3fn``,
        # ``torch.Size``, ...) deep in the import chain without a real
        # torch. None of the seeding logic under test touches a tensor.
        class _Sentinel:
            """Resolves any attribute to itself; callable; usable as a type."""

            def is_available(self):
                return False

            def __getattr__(self, name):
                return self

            def __call__(self, *a, **k):
                return self

        _SENT = _Sentinel()

        class _TorchStub(ModuleType):
            def __getattr__(self, name):
                # ``torch.dtype`` / ``torch.Size`` etc. are only used in
                # annotations (not evaluated under ``from __future__``),
                # so a permissive sentinel suffices everywhere.
                if name in ("musa", "xpu", "hpu"):
                    return _SENT  # has is_available() -> False
                return _SENT

        torch = _TorchStub("torch")
        cuda = ModuleType("torch.cuda")
        cuda.is_available = lambda: False  # type: ignore[attr-defined]
        torch.cuda = cuda  # type: ignore[attr-defined]
        sys.modules["torch.cuda"] = cuda
        sys.modules["torch"] = torch

    if "awscrt" not in sys.modules:
        awscrt = ModuleType("awscrt")
        for sub in ("auth", "io", "s3"):
            mod = ModuleType(f"awscrt.{sub}")
            setattr(awscrt, sub, mod)
            sys.modules[f"awscrt.{sub}"] = mod
        http = ModuleType("awscrt.http")
        http.HttpHeaders = object  # type: ignore[attr-defined]
        http.HttpRequest = object  # type: ignore[attr-defined]
        sys.modules["awscrt.http"] = http
        io_mod = sys.modules["awscrt.io"]
        for name in (
            "ClientTlsContext",
            "TlsConnectionOptions",
            "TlsContextOptions",
        ):
            setattr(io_mod, name, object)
        sys.modules["awscrt"] = awscrt

    if "requests" not in sys.modules:
        sys.modules["requests"] = ModuleType("requests")

    # Stub the heavy MemoryObj-bearing module so importing ``base`` (which
    # only needs the *name* ``MemoryObj`` for annotations) doesn't drag in
    # torch / vLLM. The seeding logic under test never instantiates one.
    if "lmcache.v1.memory_management" not in sys.modules:
        mm = ModuleType("lmcache.v1.memory_management")

        class _MemoryObj:  # placeholder for type annotations only
            pass

        mm.MemoryObj = _MemoryObj  # type: ignore[attr-defined]
        sys.modules["lmcache.v1.memory_management"] = mm

    if "lmcache.native_storage_ops" not in sys.modules:
        nso = ModuleType("lmcache.native_storage_ops")

        class _Bitmap:
            def __init__(self, n=0):
                self._bits = [0] * n

            def set(self, i):
                self._bits[i] = 1

            def test(self, i):
                return bool(self._bits[i])

        nso.Bitmap = _Bitmap  # type: ignore[attr-defined]
        sys.modules["lmcache.native_storage_ops"] = nso


_install_stubs()

# First Party  (imported after stubs are in place)
from lmcache.v1.distributed.api import ObjectKey  # noqa: E402
from lmcache.v1.distributed.l2_adapters import s3_l2_adapter as s3mod  # noqa: E402

_parse_s3_last_modified = s3mod._parse_s3_last_modified
_parse_list_response_xml_with_mtime = s3mod._parse_list_response_xml_with_mtime
_order_entries_by_last_modified = s3mod._order_entries_by_last_modified
_serialize_lru_index = s3mod._serialize_lru_index
_deserialize_lru_index = s3mod._deserialize_lru_index
_object_key_to_string = s3mod._object_key_to_string
S3L2Adapter = s3mod.S3L2Adapter


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _key(model: str, rank: int = 0, group: int = 0, salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=bytes([rank & 0xFF, group & 0xFF]),
        model_name=model,
        kv_rank=rank,
        object_group_id=group,
        cache_salt=salt,
    )


def _listing_xml(rows: list[tuple[str, int, str | None]], next_token=None) -> bytes:
    """Build a ListObjectsV2 XML body. Each row is (key, size, last_modified)."""
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">',
    ]
    for key, size, lm in rows:
        parts.append("<Contents>")
        parts.append(f"<Key>{key}</Key>")
        parts.append(f"<Size>{size}</Size>")
        if lm is not None:
            parts.append(f"<LastModified>{lm}</LastModified>")
        parts.append("</Contents>")
    if next_token:
        parts.append(f"<NextContinuationToken>{next_token}</NextContinuationToken>")
    parts.append("</ListBucketResult>")
    return "".join(parts).encode("utf-8")


class _FakePolicy:
    """Minimal stand-in for an LRU EvictionPolicy.

    Records keys in insertion order (like ``LRUEvictionPolicy._order``)
    so a seed in LRU order can be asserted. ``on_keys_created`` reverses
    within a batch, matching the real policy, so we can verify the
    adapter's one-key-at-a-time notify preserves global order.
    """

    def __init__(self):
        self.order: list[ObjectKey] = []

    def on_keys_created(self, keys):
        for key in reversed(keys):
            if key in self.order:
                self.order.remove(key)
            self.order.append(key)

    def on_keys_touched(self, keys):
        for key in reversed(keys):
            if key in self.order:
                self.order.remove(key)
                self.order.append(key)

    def on_keys_removed(self, keys):
        for key in keys:
            if key in self.order:
                self.order.remove(key)


class _FakeListener:
    """Bridges adapter ``_notify_*`` to a ``_FakePolicy`` (like L2EvictionPolicy)."""

    def __init__(self, policy: _FakePolicy):
        self._policy = policy

    def on_l2_keys_stored(self, keys, sizes):
        self._policy.on_keys_created(keys)

    def on_l2_keys_accessed(self, keys):
        self._policy.on_keys_touched(keys)

    def on_l2_keys_deleted(self, keys):
        self._policy.on_keys_removed(keys)


def _bare_adapter(track_access_order: bool = False) -> S3L2Adapter:
    """Construct an S3L2Adapter without running ``__init__`` (no awscrt).

    Only the attributes the seeding code paths touch are populated.
    """
    import threading

    adapter = object.__new__(S3L2Adapter)
    adapter._listeners = []
    adapter._max_capacity_bytes = 0
    adapter._total_bytes_used = 0
    adapter._bytes_by_cache_salt = {}
    adapter._usage_lock = threading.Lock()
    adapter._lock = threading.Lock()
    adapter._key_sizes = {}
    adapter._object_size_cache = {}
    adapter._track_access_order = track_access_order
    adapter._access_ticks = {}
    adapter._access_tick_counter = 0
    return adapter


# ---------------------------------------------------------------------------
# LastModified parsing
# ---------------------------------------------------------------------------


def test_parse_last_modified_z_suffix():
    epoch = _parse_s3_last_modified("2024-01-02T03:04:05.000Z")
    assert epoch is not None
    # 2024-01-02T03:04:05 UTC
    assert abs(epoch - 1704164645.0) < 1.0


def test_parse_last_modified_ordering_is_monotonic():
    a = _parse_s3_last_modified("2024-01-01T00:00:00Z")
    b = _parse_s3_last_modified("2024-06-01T00:00:00Z")
    assert a is not None and b is not None
    assert a < b


def test_parse_last_modified_bad_value_is_none():
    assert _parse_s3_last_modified("not-a-date") is None
    assert _parse_s3_last_modified(None) is None
    assert _parse_s3_last_modified("") is None


# ---------------------------------------------------------------------------
# Listing parse with mtime + size summing
# ---------------------------------------------------------------------------


def test_parse_listing_with_mtime_extracts_size_and_mtime():
    k1 = _object_key_to_string(_key("modelA", 1, 1))
    k2 = _object_key_to_string(_key("modelB", 2, 2))
    body = _listing_xml(
        [
            (k1, 100, "2024-01-01T00:00:00Z"),
            (k2, 250, "2024-02-01T00:00:00Z"),
        ]
    )
    entries, token = _parse_list_response_xml_with_mtime(body)
    assert token is None
    assert len(entries) == 2
    total = sum(sz for _k, sz, _m in entries)
    assert total == 350
    # mtimes parsed and ordered
    assert entries[0][2] < entries[1][2]


def test_parse_listing_skips_foreign_and_sidecar_objects():
    good = _object_key_to_string(_key("modelA", 1, 1))
    body = _listing_xml(
        [
            (good, 10, "2024-01-01T00:00:00Z"),
            ("_lmcache_lru_index.json", 999, "2024-01-01T00:00:00Z"),
            ("some-foreign-object-no-ats", 42, "2024-01-01T00:00:00Z"),
        ]
    )
    entries, _token = _parse_list_response_xml_with_mtime(body)
    # only the parseable cache key survives; foreign + sidecar skipped
    assert len(entries) == 1
    assert sum(sz for _k, sz, _m in entries) == 10


def test_parse_listing_next_token_roundtrip():
    k1 = _object_key_to_string(_key("m", 1))
    body = _listing_xml([(k1, 5, "2024-01-01T00:00:00Z")], next_token="CURSOR123")
    _entries, token = _parse_list_response_xml_with_mtime(body)
    assert token == "CURSOR123"


# ---------------------------------------------------------------------------
# Ordering oldest-first
# ---------------------------------------------------------------------------


def test_order_by_last_modified_oldest_first():
    newest = (_key("m", 3), 30, _parse_s3_last_modified("2024-03-01T00:00:00Z"))
    oldest = (_key("m", 1), 10, _parse_s3_last_modified("2024-01-01T00:00:00Z"))
    middle = (_key("m", 2), 20, _parse_s3_last_modified("2024-02-01T00:00:00Z"))
    ordered = _order_entries_by_last_modified([newest, oldest, middle])
    ranks = [k.kv_rank for k, _sz in ordered]
    assert ranks == [1, 2, 3]  # oldest LastModified first


def test_order_treats_missing_mtime_as_oldest():
    with_mtime = (_key("m", 2), 20, _parse_s3_last_modified("2024-01-01T00:00:00Z"))
    no_mtime = (_key("m", 1), 10, None)
    ordered = _order_entries_by_last_modified([with_mtime, no_mtime])
    # None mtime sorts first (== coldest, evicted first)
    assert ordered[0][0].kv_rank == 1


# ---------------------------------------------------------------------------
# Sidecar serialize/deserialize roundtrip + robustness
# ---------------------------------------------------------------------------


def test_sidecar_roundtrip():
    entries = [
        ("modelA@00000001@1@0101", 100, 1),
        ("modelB@00000002@2@0202", 250, 2),
        ("modelC@00000003@3@0303", 300, 5),
    ]
    body = _serialize_lru_index(entries)
    restored = _deserialize_lru_index(body)
    assert restored == entries


def test_sidecar_corrupt_json_is_none():
    assert _deserialize_lru_index(b"{not valid json") is None
    assert _deserialize_lru_index(b"\xff\xfe\x00") is None


def test_sidecar_unknown_version_is_none():
    import json

    body = json.dumps({"version": 999, "entries": []}).encode()
    assert _deserialize_lru_index(body) is None


def test_sidecar_wrong_top_level_shape_is_none():
    import json

    assert _deserialize_lru_index(json.dumps([1, 2, 3]).encode()) is None
    assert _deserialize_lru_index(json.dumps("hello").encode()) is None


def test_sidecar_partial_rows_skipped_not_fatal():
    import json

    payload = {
        "version": 1,
        "entries": [
            ["good@00000001@1@0101", 10, 1],
            ["too", "few"],  # malformed: len 2
            ["bad_size@00000002@2@0202", "notint", 2],  # bad size
            ["good2@00000003@3@0303", 20, 3],
        ],
    }
    restored = _deserialize_lru_index(json.dumps(payload).encode())
    # the two well-formed rows survive; the two malformed rows are dropped
    assert restored is not None
    assert len(restored) == 2
    assert restored[0][0] == "good@00000001@1@0101"
    assert restored[1][0] == "good2@00000003@3@0303"


# ---------------------------------------------------------------------------
# Policy population in LRU order (the load-bearing path for feature a)
# ---------------------------------------------------------------------------


def test_seed_policy_in_order_populates_policy_lru_order():
    adapter = _bare_adapter()
    policy = _FakePolicy()
    adapter.register_listener(_FakeListener(policy))

    # ordered oldest-first: rank 1 oldest, rank 3 newest
    ordered = [
        (_key("m", 1), 10),
        (_key("m", 2), 20),
        (_key("m", 3), 30),
    ]
    seeded = adapter._seed_policy_in_order(ordered)
    assert seeded == 3

    # usage counter seeded with summed sizes
    assert adapter.get_usage().total_bytes_used == 60

    # policy LRU order: oldest (rank 1) at front == first eviction victim
    assert [k.kv_rank for k in policy.order] == [1, 2, 3]

    # _key_sizes seeded so a future delete balances accounting
    assert adapter._key_sizes[_key("m", 1)] == 10


def test_seed_policy_skips_already_known_keys():
    adapter = _bare_adapter()
    policy = _FakePolicy()
    adapter.register_listener(_FakeListener(policy))

    # pretend rank 2 was already stored this process generation
    adapter._key_sizes[_key("m", 2)] = 20
    adapter._total_bytes_used = 20
    adapter._bytes_by_cache_salt = {"": 20}

    ordered = [(_key("m", 1), 10), (_key("m", 2), 20), (_key("m", 3), 30)]
    seeded = adapter._seed_policy_in_order(ordered)
    # rank 2 skipped -> only 2 new keys seeded, no double count
    assert seeded == 2
    assert adapter.get_usage().total_bytes_used == 60
    assert [k.kv_rank for k in policy.order] == [1, 3]


def test_snapshot_lru_index_sorted_by_tick():
    adapter = _bare_adapter(track_access_order=True)
    adapter._access_ticks = {
        "c": (30, 9),
        "a": (10, 1),
        "b": (20, 5),
    }
    snap = adapter._snapshot_lru_index()
    # ascending tick == oldest access first
    assert [name for name, _sz, _tick in snap] == ["a", "b", "c"]


def test_notify_stored_records_access_ticks_when_enabled():
    adapter = _bare_adapter(track_access_order=True)
    adapter.register_listener(_FakeListener(_FakePolicy()))
    adapter._notify_keys_stored([_key("m", 1), _key("m", 2)], [10, 20])
    snap = adapter._snapshot_lru_index()
    assert len(snap) == 2
    # both recorded with positive ticks, ascending
    assert snap[0][2] < snap[1][2]


def test_notify_stored_no_tracking_when_disabled():
    adapter = _bare_adapter(track_access_order=False)
    adapter.register_listener(_FakeListener(_FakePolicy()))
    adapter._notify_keys_stored([_key("m", 1)], [10])
    assert adapter._snapshot_lru_index() == []


# ---------------------------------------------------------------------------
# Standalone runner (pytest not required in this environment)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                import traceback

                print(f"FAIL {name}: {exc}")
                traceback.print_exc()
    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILED'}")
    sys.exit(1 if failures else 0)
