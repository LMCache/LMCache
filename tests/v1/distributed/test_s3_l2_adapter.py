# SPDX-License-Identifier: Apache-2.0
# Portions contributed by Metrum AI.
"""
Unit tests for S3L2Adapter.

The real ``awscrt.s3.S3Request`` is replaced with an in-memory fake that
routes PUT/GET/HEAD/DELETE against a shared dict.  No network required.
"""

# Standard
from concurrent.futures import Future as ConcurrentFuture
import gc
import io
import logging
import select
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener
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

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])

# =============================================================================
# Fake awscrt.s3.S3Request
# =============================================================================


class _FakeBackend:
    """In-memory backing store shared by all fake S3Requests in a test."""

    def __init__(self):
        self._data: dict[str, bytes] = {}
        self._lock = threading.Lock()
        # When non-None, every request raises with this message
        # (used to simulate network failures for the circuit-breaker test).
        self._inject_error: str | None = None
        # When set, GET bodies are delivered in pieces of this many bytes
        # at increasing offsets, as a ranged GET would.
        self._get_chunk_size: int | None = None
        self._put_count = 0
        self._get_count = 0
        self._delete_count = 0
        self._head_count = 0

    def reset(self):
        with self._lock:
            self._data.clear()
            self._inject_error = None
            self._get_chunk_size = None
            self._put_count = self._get_count = 0
            self._delete_count = self._head_count = 0

    def get(self, key: str) -> bytes | None:
        with self._lock:
            return self._data.get(key)

    def put(self, key: str, data: bytes):
        with self._lock:
            self._data[key] = data

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._data.pop(key, None) is not None

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def set_error(self, msg: str | None):
        with self._lock:
            self._inject_error = msg

    def set_get_chunk_size(self, n: int | None):
        with self._lock:
            self._get_chunk_size = n

    def get_chunk_size(self) -> int | None:
        with self._lock:
            return self._get_chunk_size

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {
                "put": self._put_count,
                "get": self._get_count,
                "delete": self._delete_count,
                "head": self._head_count,
            }


_BACKEND = _FakeBackend()


def _path_to_key(path: str) -> str:
    # Adapter uses url-encoded key in path; "@" is %-encoded. Reverse by
    # unquoting.  Simpler: path always starts with "/", strip it, unquote.
    # Standard
    from urllib.parse import unquote

    return unquote(path.lstrip("/"))


def _build_list_objects_v2_response(
    backend: "_FakeBackend",
    prefix: str | None,
    max_keys: int,
    continuation_token: str | None,
) -> bytes:
    """Render a minimal ListObjectsV2 XML response from the fake backend.

    The adapter's parser only consumes ``<Contents>/<Key>`` +
    ``<Contents>/<Size>`` and ``<NextContinuationToken>``, so other
    fields can be omitted. Continuation tokens here are just the
    "start-after-this-key" cursor — opaque to the adapter, simple
    enough for tests.
    """
    with backend._lock:
        all_keys = sorted(backend._data.keys())
        sizes = {k: len(backend._data[k]) for k in all_keys}

    if prefix:
        all_keys = [k for k in all_keys if k.startswith(prefix)]
    if continuation_token is not None:
        # In our fake, the token IS the last key returned on the
        # previous page. Start strictly after it.
        all_keys = [k for k in all_keys if k > continuation_token]

    page = all_keys[:max_keys]
    truncated = len(all_keys) > max_keys
    next_token = page[-1] if (truncated and page) else None

    body = ["<?xml version='1.0' encoding='UTF-8'?>"]
    body.append("<ListBucketResult xmlns='http://s3.amazonaws.com/doc/2006-03-01/'>")
    body.append(f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>")
    for k in page:
        body.append(f"<Contents><Key>{k}</Key><Size>{sizes[k]}</Size></Contents>")
    if next_token is not None:
        body.append(f"<NextContinuationToken>{next_token}</NextContinuationToken>")
    body.append("</ListBucketResult>")
    return "".join(body).encode("utf-8")


def _parse_list_query(path: str) -> tuple[str | None, int, str | None]:
    """Pull ``prefix`` / ``max-keys`` / ``continuation-token`` out of the
    ListObjectsV2 query string."""
    # Standard
    from urllib.parse import parse_qs, urlsplit

    qs = parse_qs(urlsplit(path).query)
    prefix = qs.get("prefix", [None])[0]
    max_keys = int(qs.get("max-keys", ["1000"])[0])
    cont = qs.get("continuation-token", [None])[0]
    return prefix, max_keys, cont


def _call_on_done(on_done, **kwargs) -> None:
    """Invoke ``on_done`` the way the CRT does: after ``finished_future``
    is settled, with any exception it raises dropped, never propagated.
    A check that lives in ``on_done`` therefore cannot fail a request."""
    if on_done is None:
        return
    try:
        on_done(**kwargs)
    except Exception:
        pass


class _FakeS3Request:
    """In-process substitute for awscrt.s3.S3Request.

    Dispatches by HTTP method + operation_name against ``_BACKEND`` and
    fires the on_headers / on_body callbacks that the real CRT would
    invoke, then settles ``finished_future`` and only then calls
    ``on_done``, in the CRT's order.
    """

    def __init__(
        self,
        *,
        client,
        type,  # noqa: A002
        request,
        credential_provider,
        region,
        on_body=None,
        on_done=None,
        on_headers=None,
        operation_name=None,
        **kwargs,
    ):
        self.finished_future: ConcurrentFuture = ConcurrentFuture()

        # Extract method + path from the HttpRequest
        method = request.method
        path = request.path

        # ListObjectsV2 is dispatched as GET on the bucket root with a
        # ``?list-type=2&...`` query string — intercept before the
        # per-key GET branch below so it doesn't try to look up an
        # object named "".
        if (
            method == "GET"
            and operation_name == "ListObjectsV2"
            and path.startswith("/?")
        ):
            prefix, max_keys, cont = _parse_list_query(path)
            xml = _build_list_objects_v2_response(_BACKEND, prefix, max_keys, cont)
            try:
                if on_headers is not None:
                    on_headers(200, [("content-type", "application/xml")])
                if on_body is not None:
                    on_body(xml, 0)
                self.finished_future.set_result(None)
                _call_on_done(on_done, error=None, status_code=200)
            except Exception as e:
                if not self.finished_future.done():
                    self.finished_future.set_exception(e)
            return
        key_str = _path_to_key(path)

        err_inj = _BACKEND._inject_error

        try:
            if err_inj:
                raise RuntimeError(err_inj)

            if method == "PUT":
                data = b""
                body = request.body_stream
                # body is a MemoryViewStream — read it fully
                body.seek(0)
                data = bytes(body.read())
                _BACKEND.put(key_str, data)
                with _BACKEND._lock:
                    _BACKEND._put_count += 1
                if on_headers is not None:
                    on_headers(200, [])
                self.finished_future.set_result(None)
                _call_on_done(on_done, error=None, status_code=200)

            elif method == "GET":
                with _BACKEND._lock:
                    _BACKEND._get_count += 1
                data = _BACKEND.get(key_str)
                if data is None:
                    self.finished_future.set_exception(
                        RuntimeError(f"S3 GET 404 for {key_str}")
                    )
                    _call_on_done(on_done, error=None, status_code=404)
                    return
                if on_body is not None:
                    step = _BACKEND.get_chunk_size() or max(len(data), 1)
                    for off in range(0, len(data), step):
                        on_body(data[off : off + step], off)
                # The CRT completes ``finished_future`` before it invokes
                # ``on_done``: an exception raised in ``on_done`` cannot
                # fail a request that has already been reported done.
                self.finished_future.set_result(None)
                _call_on_done(on_done, error=None, status_code=200)

            elif method == "HEAD":
                with _BACKEND._lock:
                    _BACKEND._head_count += 1
                data = _BACKEND.get(key_str)
                if data is None:
                    if on_headers is not None:
                        on_headers(404, [])
                    self.finished_future.set_exception(
                        RuntimeError(f"S3 HEAD 404 for {key_str}")
                    )
                    return
                if on_headers is not None:
                    on_headers(200, [("content-length", str(len(data)))])
                self.finished_future.set_result(None)

            elif method == "DELETE":
                with _BACKEND._lock:
                    _BACKEND._delete_count += 1
                _BACKEND.delete(key_str)
                if on_headers is not None:
                    on_headers(204, [])
                self.finished_future.set_result(None)
                _call_on_done(on_done, error=None, status_code=204)

            else:
                self.finished_future.set_exception(
                    RuntimeError(f"unexpected method {method}")
                )

        except Exception as e:
            if not self.finished_future.done():
                self.finished_future.set_exception(e)
            _call_on_done(on_done, error=str(e), status_code=None)


class _FakeHttpRequest:
    """Plain-Python HttpRequest that preserves body_stream as-is."""

    def __init__(self, method, path, headers, body_stream=None):
        self.method = method
        self.path = path
        self.headers = headers
        self.body_stream = body_stream


@pytest.fixture(autouse=True)
def patch_s3_request(monkeypatch):
    """Replace awscrt.s3.S3Request in the adapter with the in-memory fake.

    Also patch HttpRequest so body_stream is preserved as a plain Python
    object (real HttpRequest wraps it in an opaque awscrt InputStream).

    Stubs the credentials provider so tests don't depend on boto3 or any
    AWS credentials being present in the environment; mocked requests
    never reach a real signer.
    """
    _BACKEND.reset()
    monkeypatch.setattr(s3mod.s3, "S3Request", _FakeS3Request)
    monkeypatch.setattr(s3mod, "HttpRequest", _FakeHttpRequest)
    monkeypatch.setattr(
        s3mod,
        "_make_credentials_provider",
        lambda _config: s3mod.auth.AwsCredentialsProvider.new_static(
            "test-key", "test-secret"
        ),
    )
    yield


# =============================================================================
# Helpers
# =============================================================================


def create_object_key(chunk_id: int, model_name: str = "test_model") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
    )


def create_memory_obj(size: int = 16, fill_value: float = 1.0) -> TensorMemoryObj:
    raw_data = torch.empty(size, dtype=torch.float32)
    raw_data.fill_(fill_value)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.float32,
        address=0,
        phy_size=size * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def create_guarded_memory_obj(
    size: int = 16, guard_value: float = -1.0
) -> tuple[TensorMemoryObj, torch.Tensor]:
    """A MemoryObj over the first ``size`` float32 elements of a tensor
    twice that long. The second half is a sentinel: a copy that runs past
    the logical buffer lands there, where the test can see it, instead of
    in unrelated memory. Returns the object and the sentinel view."""
    backing = torch.full((size * 2,), guard_value, dtype=torch.float32)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.float32,
        address=0,
        phy_size=size * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    obj = TensorMemoryObj(backing[:size], metadata, parent_allocator=None)
    return obj, backing[size:]


def wait_for_event_fd(event_fd: int, timeout: float = 5.0) -> bool:
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
    if events:
        try:
            consume_fd(event_fd)
        except BlockingIOError:
            pass
        return True
    return False


@pytest.fixture
def adapter():
    config = S3L2AdapterConfig(
        s3_endpoint="s3://test-bucket",
        s3_region="us-east-1",
        s3_prefer_http2=False,  # skip TLS ALPN setup for faster init
        # Keep awscrt EventLoopGroup tiny so CI runners with tight FD
        # ulimits (each epoll event loop costs a few fds) don't exhaust
        # file descriptors across ~15 adapter fixtures in this module.
        # Production callers should leave this at the default 64.
        s3_num_io_threads=1,
        max_capacity_gb=0.001,  # 1 MB
    )
    a = S3L2Adapter(config)
    yield a
    a.close()


class _RecordingListener(L2AdapterListener):
    def __init__(self):
        self.stored: list[list[ObjectKey]] = []
        self.accessed: list[list[ObjectKey]] = []
        self.deleted: list[list[ObjectKey]] = []

    def on_l2_keys_stored(self, keys, sizes):
        self.stored.append(list(keys))

    def on_l2_keys_accessed(self, keys):
        self.accessed.append(list(keys))

    def on_l2_keys_deleted(self, keys):
        self.deleted.append(list(keys))


# =============================================================================
# Key serialization
# =============================================================================


class TestObjectKeySerialization:
    def test_format(self):
        key = ObjectKey(
            chunk_hash=b"\x00\x01\x02\x03",
            model_name="llama",
            kv_rank=255,
        )
        assert _object_key_to_string(key) == "llama@000000ff@0@00010203"

    def test_object_group_id_embedded(self):
        key = ObjectKey(
            chunk_hash=b"\x00\x01\x02\x03",
            model_name="llama",
            kv_rank=255,
            object_group_id=5,
        )
        assert _object_key_to_string(key) == "llama@000000ff@5@00010203"

    def test_cache_salt_appended(self):
        """A non-empty cache_salt must be included in the S3 object name so
        two users with the same model/rank/chunk map to distinct objects.
        """
        base_key = ObjectKey(
            chunk_hash=b"\x00\x01\x02\x03",
            model_name="llama",
            kv_rank=255,
        )
        salted = ObjectKey(
            chunk_hash=b"\x00\x01\x02\x03",
            model_name="llama",
            kv_rank=255,
            cache_salt="user-42",
        )
        assert _object_key_to_string(base_key) == "llama@000000ff@0@00010203"
        assert _object_key_to_string(salted) == "llama@000000ff@0@00010203@user-42"
        assert _object_key_to_string(base_key) != _object_key_to_string(salted)


# =============================================================================
# Event fd interface
# =============================================================================


class TestEventFdInterface:
    def test_three_distinct_fds(self, adapter):
        a = adapter.get_store_event_fd()
        b = adapter.get_lookup_and_lock_event_fd()
        c = adapter.get_load_event_fd()
        assert a >= 0 and b >= 0 and c >= 0
        assert len({a, b, c}) == 3


# =============================================================================
# Round-trip
# =============================================================================


class TestStoreLookupLoad:
    def test_roundtrip_single_key(self, adapter):
        key = create_object_key(1)
        obj = create_memory_obj(fill_value=3.14)

        # Store
        tid = adapter.submit_store_task([key], [obj])
        assert wait_for_event_fd(adapter.get_store_event_fd())
        completed = adapter.pop_completed_store_tasks()
        assert completed[tid].is_successful()

        # Lookup
        tid = adapter.submit_lookup_and_lock_task([key], {0: _EMPTY_LAYOUT})
        assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        bm = adapter.query_lookup_and_lock_result(tid)
        assert bm is not None and bm.test(0) is True

        # Load into a fresh buffer and verify the bytes match.
        dst = create_memory_obj(fill_value=0.0)
        tid = adapter.submit_load_task([key], [dst])
        assert wait_for_event_fd(adapter.get_load_event_fd())
        bm = adapter.query_load_result(tid)
        assert bm is not None and bm.test(0) is True
        assert torch.allclose(dst.tensor, torch.full((16,), 3.14))

    def test_partial_hits(self, adapter):
        # Store keys 0, 2
        stored = [create_object_key(0), create_object_key(2)]
        objs = [create_memory_obj(fill_value=float(i)) for i in range(2)]
        adapter.submit_store_task(stored, objs)
        wait_for_event_fd(adapter.get_store_event_fd())
        adapter.pop_completed_store_tasks()

        # Lookup 0, 1, 2, 3 — expect bitmap 1010
        keys = [create_object_key(i) for i in range(4)]
        tid = adapter.submit_lookup_and_lock_task(keys, {0: _EMPTY_LAYOUT})
        wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        bm = adapter.query_lookup_and_lock_result(tid)
        assert bm is not None
        assert bm.test(0) is True
        assert bm.test(1) is False
        assert bm.test(2) is True
        assert bm.test(3) is False

    def test_load_miss_returns_zero_bit(self, adapter):
        key = create_object_key(99)
        dst = create_memory_obj()
        tid = adapter.submit_load_task([key], [dst])
        wait_for_event_fd(adapter.get_load_event_fd())
        bm = adapter.query_load_result(tid)
        assert bm is not None
        assert bm.test(0) is False

    def test_query_lookup_returns_none_after_pop(self, adapter):
        key = create_object_key(1)
        tid = adapter.submit_lookup_and_lock_task([key], {0: _EMPTY_LAYOUT})
        wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        assert adapter.query_lookup_and_lock_result(tid) is not None
        assert adapter.query_lookup_and_lock_result(tid) is None


# =============================================================================
# Size validation on load
# =============================================================================


class TestLoadSizeValidation:
    """The store is not trusted to hold objects of the expected size.

    A GET must never write outside the destination buffer, and a load
    is a hit only when exactly the destination's byte count arrived.
    """

    @staticmethod
    def _store(adapter, key, obj) -> None:
        tid = adapter.submit_store_task([key], [obj])
        assert wait_for_event_fd(adapter.get_store_event_fd())
        assert adapter.pop_completed_store_tasks()[tid].is_successful()

    @staticmethod
    def _load(adapter, keys, dsts):
        tid = adapter.submit_load_task(keys, dsts)
        assert wait_for_event_fd(adapter.get_load_event_fd())
        bm = adapter.query_load_result(tid)
        assert bm is not None
        return bm

    @staticmethod
    def _replace_object(key, data: bytes) -> str:
        key_str = s3mod._object_key_to_string(key)
        _BACKEND.put(key_str, data)
        return key_str

    def test_short_object_is_a_miss(self, adapter):
        # Stored at the right size, then replaced on the store by a
        # shorter object under the same key. The cached size still says
        # the object is complete, so the GET is issued and delivers half.
        key = create_object_key(1)
        self._store(adapter, key, create_memory_obj(fill_value=3.14))
        data = _BACKEND.get(s3mod._object_key_to_string(key))
        assert data is not None
        key_str = self._replace_object(key, data[: len(data) // 2])

        bm = self._load(adapter, [key], [create_memory_obj(fill_value=0.0)])
        assert bm.test(0) is False
        # The size cache no longer vouches for the key, so the next
        # lookup asks S3 again instead of repeating the failed load.
        assert key_str not in adapter._object_size_cache

    def test_mismatching_cached_size_skips_the_get(self, adapter):
        # A fresh adapter learns the size from HEAD during lookup; the
        # load then sees it disagrees with the destination and does not
        # transfer anything.
        key = create_object_key(2)
        dst = create_memory_obj(fill_value=0.0)
        self._replace_object(key, b"\x55" * (dst.get_size() // 2))

        tid = adapter.submit_lookup_and_lock_task([key], {0: _EMPTY_LAYOUT})
        assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        lookup_bm = adapter.query_lookup_and_lock_result(tid)
        assert lookup_bm is not None and lookup_bm.test(0) is True

        gets_before = _BACKEND.counts()["get"]
        bm = self._load(adapter, [key], [dst])
        assert bm.test(0) is False
        assert _BACKEND.counts()["get"] == gets_before
        assert torch.all(dst.tensor == 0.0)

    def test_oversized_object_is_rejected_before_any_copy(self, adapter):
        # No cached size, so the GET is issued; the body is twice the
        # destination and arrives in one callback. It must be refused
        # before the first memmove: neither the buffer nor the guard
        # region after it may change.
        key = create_object_key(3)
        dst, guard = create_guarded_memory_obj()
        self._replace_object(key, b"\x7f" * (dst.get_size() * 2))

        bm = self._load(adapter, [key], [dst])
        assert bm.test(0) is False
        assert torch.all(dst.tensor == -1.0)
        assert torch.all(guard == -1.0)

    def test_oversized_object_in_a_later_callback(self, adapter):
        # The object grew after this adapter cached its size, so the
        # size check before the GET passes and the body arrives in three
        # pieces; the third would cross the end of the destination.
        key = create_object_key(4)
        dst, guard = create_guarded_memory_obj()
        capacity = dst.get_size()
        self._store(adapter, key, create_memory_obj(fill_value=2.0))
        key_str = self._replace_object(key, b"\x7f" * (capacity * 3 // 2))
        _BACKEND.set_get_chunk_size(capacity // 2)

        bm = self._load(adapter, [key], [dst])
        assert bm.test(0) is False
        assert torch.all(guard == -1.0)
        assert key_str not in adapter._object_size_cache

    def test_fragmented_delivery_of_a_complete_object_is_a_hit(self, adapter):
        key = create_object_key(5)
        self._store(adapter, key, create_memory_obj(fill_value=1.5))
        dst = create_memory_obj(fill_value=0.0)
        _BACKEND.set_get_chunk_size(24)  # 64 bytes -> offsets 0, 24, 48

        bm = self._load(adapter, [key], [dst])
        assert bm.test(0) is True
        assert torch.all(dst.tensor == 1.5)

    def test_only_the_wrong_sized_key_misses(self, adapter):
        keys = [create_object_key(i) for i in (10, 11, 12)]
        for key in keys:
            self._store(adapter, key, create_memory_obj(fill_value=1.0))
        data = _BACKEND.get(s3mod._object_key_to_string(keys[1]))
        assert data is not None
        self._replace_object(keys[1], data[:8])

        dsts = [create_memory_obj(fill_value=0.0) for _ in keys]
        bm = self._load(adapter, keys, dsts)
        assert (bm.test(0), bm.test(1), bm.test(2)) == (True, False, True)
        assert torch.all(dsts[0].tensor == 1.0)
        assert torch.all(dsts[2].tensor == 1.0)


# =============================================================================
# Eviction (delete + locking)
# =============================================================================


class TestEviction:
    def _store(self, adapter, key, obj):
        adapter.submit_store_task([key], [obj])
        wait_for_event_fd(adapter.get_store_event_fd())
        adapter.pop_completed_store_tasks()

    def _lookup(self, adapter, key):
        tid = adapter.submit_lookup_and_lock_task([key], {0: _EMPTY_LAYOUT})
        wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        return adapter.query_lookup_and_lock_result(tid)

    def test_delete_removes_key(self, adapter):
        key = create_object_key(1)
        self._store(adapter, key, create_memory_obj())
        assert _BACKEND.contains(_object_key_to_string(key))
        adapter.delete([key])
        assert not _BACKEND.contains(_object_key_to_string(key))

    def test_lock_blocks_delete(self, adapter):
        key = create_object_key(1)
        self._store(adapter, key, create_memory_obj())
        bm = self._lookup(adapter, key)  # bumps refcount
        assert bm.test(0) is True

        deletes_before = _BACKEND.counts()["delete"]
        adapter.delete([key])
        assert _BACKEND.counts()["delete"] == deletes_before
        # Still there.
        assert _BACKEND.contains(_object_key_to_string(key))

        adapter.submit_unlock([key])
        adapter.delete([key])
        assert not _BACKEND.contains(_object_key_to_string(key))

    def test_refcount_unlock(self, adapter):
        key = create_object_key(1)
        self._store(adapter, key, create_memory_obj())
        self._lookup(adapter, key)
        self._lookup(adapter, key)  # refcount now 2

        adapter.submit_unlock([key])  # refcount 1, still locked
        adapter.delete([key])
        assert _BACKEND.contains(_object_key_to_string(key))

        adapter.submit_unlock([key])  # refcount 0
        adapter.delete([key])
        assert not _BACKEND.contains(_object_key_to_string(key))

    def test_delete_on_unknown_key(self, adapter):
        # Should not raise.
        adapter.delete([create_object_key(42)])


# =============================================================================
# get_usage
# =============================================================================


class TestGetUsage:
    def test_disabled_returns_minus_one(self):
        cfg = S3L2AdapterConfig(
            s3_endpoint="s3://b",
            s3_region="r",
            s3_prefer_http2=False,
            s3_num_io_threads=1,
            max_capacity_gb=0.0,
        )
        a = S3L2Adapter(cfg)
        try:
            usage = a.get_usage()
            assert usage.usage_fraction == -1.0
            assert usage.total_bytes_used == 0
            assert usage.total_capacity_bytes == 0
        finally:
            a.close()

    def test_usage_grows_on_store_and_shrinks_on_delete(self, adapter):
        # adapter max_capacity_gb = 0.001 = 1 MB
        # each obj is 16 floats = 64 bytes
        keys = [create_object_key(i) for i in range(4)]
        objs = [create_memory_obj() for _ in range(4)]

        adapter.submit_store_task(keys, objs)
        wait_for_event_fd(adapter.get_store_event_fd())
        adapter.pop_completed_store_tasks()

        total = 4 * 64
        capacity = int(0.001 * 1024**3)
        usage = adapter.get_usage()
        assert usage.total_bytes_used == total
        assert usage.total_capacity_bytes == capacity
        assert usage.usage_fraction == pytest.approx(total / capacity)

        adapter.delete(keys)
        usage = adapter.get_usage()
        assert usage.total_bytes_used == 0
        assert usage.usage_fraction == 0.0


# =============================================================================
# Circuit breaker
# =============================================================================


class TestCircuitBreaker:
    def test_trips_after_three_connection_errors(self, adapter):
        _BACKEND.set_error("CONNECTION_REFUSED: mock")

        keys = [create_object_key(i) for i in range(3)]
        for k in keys:
            obj = create_memory_obj()
            adapter.submit_store_task([k], [obj])
            wait_for_event_fd(adapter.get_store_event_fd(), timeout=2.0)
            adapter.pop_completed_store_tasks()

        status = adapter.report_status()
        assert status["connection_disabled"] is True
        assert status["is_healthy"] is False

        # Subsequent submit short-circuits.
        _BACKEND.set_error(None)  # even if we un-inject, the breaker is tripped
        put_before = _BACKEND.counts()["put"]
        disabled_tid = adapter.submit_store_task(
            [create_object_key(42)], [create_memory_obj()]
        )
        wait_for_event_fd(adapter.get_store_event_fd(), timeout=2.0)
        completed = adapter.pop_completed_store_tasks()
        assert not completed[disabled_tid].is_successful()
        assert _BACKEND.counts()["put"] == put_before  # never reached the backend

        # Lookup and load also short-circuit to all-zero bitmaps.
        tid = adapter.submit_lookup_and_lock_task(
            [create_object_key(1)], {0: _EMPTY_LAYOUT}
        )
        wait_for_event_fd(adapter.get_lookup_and_lock_event_fd(), timeout=2.0)
        bm = adapter.query_lookup_and_lock_result(tid)
        assert bm is not None and bm.test(0) is False

        tid = adapter.submit_load_task([create_object_key(1)], [create_memory_obj()])
        wait_for_event_fd(adapter.get_load_event_fd(), timeout=2.0)
        bm = adapter.query_load_result(tid)
        assert bm is not None and bm.test(0) is False


# =============================================================================
# Listener notifications
# =============================================================================


class TestListener:
    def test_stored_and_deleted_fire(self, adapter):
        listener = _RecordingListener()
        adapter.register_listener(listener)

        key = create_object_key(1)
        adapter.submit_store_task([key], [create_memory_obj()])
        wait_for_event_fd(adapter.get_store_event_fd())
        adapter.pop_completed_store_tasks()
        # Give the coroutine a moment to fire notify; it fires inline but
        # on the loop thread, so yield briefly.
        time.sleep(0.05)
        assert any(key in batch for batch in listener.stored)

        adapter.delete([key])
        assert any(key in batch for batch in listener.deleted)

    def test_accessed_fires_on_hit(self, adapter):
        listener = _RecordingListener()
        adapter.register_listener(listener)

        key = create_object_key(1)
        adapter.submit_store_task([key], [create_memory_obj()])
        wait_for_event_fd(adapter.get_store_event_fd())
        adapter.pop_completed_store_tasks()

        tid = adapter.submit_lookup_and_lock_task([key], {0: _EMPTY_LAYOUT})
        wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        adapter.query_lookup_and_lock_result(tid)
        time.sleep(0.05)
        assert any(key in batch for batch in listener.accessed)


# =============================================================================
# Config
# =============================================================================


class TestConfig:
    def test_from_dict_requires_endpoint_and_region(self):
        with pytest.raises(ValueError):
            S3L2AdapterConfig.from_dict({"s3_region": "us-east-1"})
        with pytest.raises(ValueError):
            S3L2AdapterConfig.from_dict({"s3_endpoint": "s3://b"})

    def test_from_dict_parses_all_fields(self):
        cfg = S3L2AdapterConfig.from_dict(
            {
                "type": "s3",
                "s3_endpoint": "s3://my-bucket",
                "s3_region": "us-west-2",
                "s3_num_io_threads": 32,
                "s3_prefer_http2": False,
                "s3_enable_s3express": True,
                "disable_tls": True,
                "aws_access_key_id": "id",
                "aws_secret_access_key": "secret",
                "max_capacity_gb": 2.5,
            }
        )
        assert cfg.s3_endpoint == "s3://my-bucket"
        assert cfg.s3_region == "us-west-2"
        assert cfg.s3_num_io_threads == 32
        assert cfg.s3_prefer_http2 is False
        assert cfg.s3_enable_s3express is True
        assert cfg.disable_tls is True
        assert cfg.aws_access_key_id == "id"
        assert cfg.aws_secret_access_key == "secret"
        assert cfg.max_capacity_gb == 2.5

    def test_help_nonempty(self):
        assert isinstance(S3L2AdapterConfig.help(), str)
        assert "s3_endpoint" in S3L2AdapterConfig.help()


# =============================================================================
# Factory registration
# =============================================================================


class TestFactoryRegistration:
    def test_create_l2_adapter_registers_s3(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters import create_l2_adapter

        cfg = S3L2AdapterConfig.from_dict(
            {
                "type": "s3",
                "s3_endpoint": "s3://fac-test",
                "s3_region": "us-east-1",
                "s3_prefer_http2": False,
                "s3_num_io_threads": 1,
            }
        )
        adp = create_l2_adapter(cfg)
        try:
            assert isinstance(adp, S3L2Adapter)
        finally:
            adp.close()


# =============================================================================
# list_l2_keys
# =============================================================================
#
# These tests populate the fake S3 backend directly via ``_BACKEND.put``,
# bypassing the adapter's store pipeline. The listing path is what's
# under test: ListObjectsV2 dispatch, XML parse, prefix filter, and
# cursor pagination.


def _put_object(key: ObjectKey, size: int = 0) -> None:
    """Place a key in the fake S3 backend with deterministic content.

    Mirrors the production PUT pipeline: ``_object_key_to_string`` →
    ``_format_safe_path`` (URL-encoding only — no path-flattening) →
    URL-unquote in the fake → backend storage. So the fake's stored
    name is byte-identical to the literal S3 object name a real PUT
    would land.
    """
    # First Party
    from lmcache.v1.distributed.l2_adapters.s3_l2_adapter import (
        _object_key_to_string,
    )

    _BACKEND.put(_object_key_to_string(key), b"\x00" * size)


class TestS3L2AdapterListKeys:
    def test_empty_bucket_returns_empty_page(self, adapter):
        page = adapter.list_l2_keys()
        assert page.entries == ()
        assert page.next_page_token is None

    def test_lists_all_keys_when_unfiltered(self, adapter):
        for i in range(3):
            _put_object(create_object_key(i, model_name="llama"), size=100 + i)

        page = adapter.list_l2_keys(page_size=10)

        assert len(page.entries) == 3
        assert {e.size_bytes for e in page.entries} == {100, 101, 102}
        assert page.next_page_token is None

    def test_model_name_pushed_down_as_prefix(self, adapter):
        llama = ObjectKey(chunk_hash=b"\x00" * 4, model_name="llama", kv_rank=0)
        mistral = ObjectKey(chunk_hash=b"\x00" * 4, model_name="mistral", kv_rank=0)
        _put_object(llama, size=1)
        _put_object(mistral, size=1)

        page = adapter.list_l2_keys(model_name="llama", page_size=10)
        assert [e.key for e in page.entries] == [llama.to_encoded_object_key()]

    def test_model_name_with_slash_round_trips(self, adapter):
        # Real HF model ids contain ``/``. The adapter stores the
        # literal object name on S3 (no path-flattening) and the
        # listing decodes it back unchanged.
        original = ObjectKey(
            chunk_hash=b"\xde\xad\xbe\xef",
            model_name="meta-llama/Llama-3.1-8B",
            kv_rank=7,
        )
        encoded = original.to_encoded_object_key()
        _put_object(original, size=128)

        # The listing returns the original model_name (slash intact).
        page = adapter.list_l2_keys(page_size=10)
        assert len(page.entries) == 1
        assert page.entries[0].key == encoded
        assert page.entries[0].key.model_name == "meta-llama/Llama-3.1-8B"

        # And the model_name filter accepts the caller-supplied form
        # verbatim — no encoding magic to remember.
        page = adapter.list_l2_keys(model_name="meta-llama/Llama-3.1-8B", page_size=10)
        assert [e.key for e in page.entries] == [encoded]

    def test_pagination_cursor_walks_full_set(self, adapter):
        # 5 keys, page_size=2 → expect 3 calls: (0,1), (2,3), (4,).
        for i in range(5):
            _put_object(create_object_key(i, model_name="llama"), size=1)

        collected = []
        cursor = None
        while True:
            page = adapter.list_l2_keys(page_size=2, cursor=cursor)
            collected.extend(page.entries)
            cursor = page.next_page_token
            if cursor is None:
                break
        assert len(collected) == 5

    def test_page_size_must_be_positive(self, adapter):
        with pytest.raises(ValueError):
            adapter.list_l2_keys(page_size=0)

    def test_page_size_clamped_to_s3_max(self, adapter):
        # Ask for far more than S3's MaxKeys ceiling. The adapter
        # clamps internally and the fake honors whatever it receives,
        # so 5 keys come back without error.
        for i in range(5):
            _put_object(create_object_key(i, model_name="m"), size=1)
        page = adapter.list_l2_keys(page_size=999_999)
        assert len(page.entries) == 5
        assert page.next_page_token is None

    def test_circuit_breaker_blocks_listing(self, adapter):
        # When the connection is disabled, listing must surface a
        # clear error instead of issuing a doomed request.
        with adapter._lock:
            adapter._connection_disabled = True
        with pytest.raises(RuntimeError) as exc:
            adapter.list_l2_keys()
        assert "disabled" in str(exc.value)

    def test_unparsable_objects_silently_skipped(self, adapter):
        # An object with a name that isn't this adapter's format
        # (e.g. left behind by another tool) must not poison the
        # response — it's just dropped from the listing.
        _put_object(create_object_key(0, model_name="m"), size=1)
        _BACKEND.put("not-our-format-key", b"junk")

        page = adapter.list_l2_keys(page_size=10)

        assert len(page.entries) == 1
        assert (
            page.entries[0].key
            == create_object_key(0, model_name="m").to_encoded_object_key()
        )


# =============================================================================
# CRT client sizing
# =============================================================================

_MIB = 1024 * 1024

# The exact kwargs the adapter passed to ``S3Client`` before the sizing keys
# existed. A config without the new keys must still produce exactly these.
_BASE_CLIENT_KWARGS = {
    "bootstrap",
    "region",
    "enable_s3express",
    "tls_connection_options",
    "tls_mode",
    "signing_config",
}


class _RecordingS3Client:
    """Stand-in for ``awscrt.s3.S3Client`` that records its kwargs."""

    def __init__(self, calls: list[dict], **kwargs):
        calls.append(kwargs)


@pytest.fixture
def s3_client_calls(monkeypatch) -> list[dict]:
    """Replace ``S3Client`` and return the list of kwargs it was built with."""
    calls: list[dict] = []
    monkeypatch.setattr(
        s3mod.s3,
        "S3Client",
        lambda **kwargs: _RecordingS3Client(calls, **kwargs),
    )
    return calls


@pytest.fixture
def adapter_warnings():
    """Collect WARNING records from the adapter module's logger, which does
    not propagate to pytest's caplog."""
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture(level=logging.WARNING)
    lg = logging.getLogger(s3mod.__name__)
    lg.addHandler(handler)
    try:
        yield records
    finally:
        lg.removeHandler(handler)


def _sized_config(**overrides) -> S3L2AdapterConfig:
    d = {
        "type": "s3",
        "s3_endpoint": "s3://sizing-test",
        "s3_region": "us-east-1",
        "s3_prefer_http2": False,
        "s3_num_io_threads": 1,
    }
    d.update(overrides)
    return S3L2AdapterConfig.from_dict(d)


def _build_and_close(cfg: S3L2AdapterConfig) -> None:
    S3L2Adapter(cfg).close()


class TestCrtSizingConfig:
    def test_keys_default_to_none(self):
        cfg = _sized_config()
        assert cfg.s3_throughput_target_gbps is None
        assert cfg.s3_part_size_mb is None
        assert cfg.s3_multipart_threshold_mb is None

    def test_good_values_parse(self):
        cfg = _sized_config(
            s3_throughput_target_gbps=100,
            s3_part_size_mb=16,
            s3_multipart_threshold_mb=256,
        )
        assert cfg.s3_throughput_target_gbps == 100.0
        assert isinstance(cfg.s3_throughput_target_gbps, float)
        assert cfg.s3_part_size_mb == 16
        assert cfg.s3_multipart_threshold_mb == 256

    @pytest.mark.parametrize(
        "key, max_value",
        [
            ("s3_throughput_target_gbps", 4000),
            ("s3_part_size_mb", 5 * 1024),
            ("s3_multipart_threshold_mb", 5 * 1024),
        ],
    )
    def test_upper_bound_is_accepted(self, key, max_value):
        assert getattr(_sized_config(**{key: max_value}), key) == max_value

    @pytest.mark.parametrize(
        "bad",
        [
            0,
            True,
            "100",
            float("nan"),
            float("inf"),
            4000.5,
            # Too large for a float: must be a ValueError, not OverflowError.
            10**400,
        ],
    )
    def test_bad_throughput_target_rejected(self, bad):
        with pytest.raises(ValueError, match="s3_throughput_target_gbps"):
            _sized_config(s3_throughput_target_gbps=bad)

    @pytest.mark.parametrize("key", ["s3_part_size_mb", "s3_multipart_threshold_mb"])
    @pytest.mark.parametrize(
        "bad",
        # 2**44 MiB is 2**64 bytes, which awscrt would wrap to 0.
        [0, True, "16", 16.0, 5 * 1024 + 1, 2**44],
    )
    def test_bad_integer_keys_rejected(self, key, bad):
        with pytest.raises(ValueError, match=key):
            _sized_config(**{key: bad})


class TestCrtClientKwargs:
    def test_defaults_pass_exactly_the_base_kwargs(self, s3_client_calls):
        """Contract test: it passes on the code before the sizing keys too.
        It pins that a config without them builds the same single client."""
        _build_and_close(_sized_config())
        assert len(s3_client_calls) == 1
        assert set(s3_client_calls[0]) == _BASE_CLIENT_KWARGS

    def test_sizing_keys_are_forwarded(self, s3_client_calls):
        _build_and_close(
            _sized_config(
                s3_throughput_target_gbps=100,
                s3_part_size_mb=16,
                s3_multipart_threshold_mb=256,
            )
        )
        (kwargs,) = s3_client_calls
        assert set(kwargs) == _BASE_CLIENT_KWARGS | {
            "throughput_target_gbps",
            "part_size",
            "multipart_upload_threshold",
        }
        assert kwargs["throughput_target_gbps"] == 100.0
        assert isinstance(kwargs["throughput_target_gbps"], float)
        assert kwargs["part_size"] == 16 * _MIB
        assert kwargs["multipart_upload_threshold"] == 256 * _MIB

    @pytest.mark.parametrize("part_mb", [32, 5 * 1024])
    def test_part_size_at_or_above_32_mib_warns_and_is_kept(
        self, part_mb, s3_client_calls, adapter_warnings
    ):
        _build_and_close(_sized_config(s3_part_size_mb=part_mb))
        (kwargs,) = s3_client_calls
        assert kwargs["part_size"] == part_mb * _MIB
        assert any("s3_part_size_mb" in r.getMessage() for r in adapter_warnings)

    @pytest.mark.parametrize("part_mb", [1, 31])
    def test_part_size_below_32_mib_is_kept(
        self, part_mb, s3_client_calls, adapter_warnings
    ):
        _build_and_close(_sized_config(s3_part_size_mb=part_mb))
        (kwargs,) = s3_client_calls
        assert kwargs["part_size"] == part_mb * _MIB
        assert adapter_warnings == []

    def test_target_without_part_size_warns_and_keeps_crt_part_size(
        self, s3_client_calls, adapter_warnings
    ):
        _build_and_close(_sized_config(s3_throughput_target_gbps=10))
        (kwargs,) = s3_client_calls
        assert "part_size" not in kwargs
        (record,) = adapter_warnings
        assert "s3_part_size_mb" in record.getMessage()

    def test_target_with_part_size_does_not_warn(
        self, s3_client_calls, adapter_warnings
    ):
        _build_and_close(
            _sized_config(s3_throughput_target_gbps=10, s3_part_size_mb=16)
        )
        (kwargs,) = s3_client_calls
        assert kwargs["part_size"] == 16 * _MIB
        assert adapter_warnings == []

    def test_real_awscrt_takes_the_forwarded_kwargs_at_their_bounds(self, monkeypatch):
        # The real S3Client is built through a spy, so this fails if a kwarg
        # is not forwarded or if the installed awscrt rejects it.
        calls: list[dict] = []
        real_client = s3mod.s3.S3Client

        def _spy(**kwargs):
            calls.append(kwargs)
            return real_client(**kwargs)

        monkeypatch.setattr(s3mod.s3, "S3Client", _spy)
        _build_and_close(
            _sized_config(
                s3_throughput_target_gbps=4000,
                s3_part_size_mb=31,
                s3_multipart_threshold_mb=5 * 1024,
            )
        )
        (kwargs,) = calls
        assert kwargs["throughput_target_gbps"] == 4000.0
        assert kwargs["part_size"] == 31 * _MIB
        assert kwargs["multipart_upload_threshold"] == 5 * 1024 * _MIB


# =============================================================================
# MemoryViewStream.readinto
# =============================================================================

_STREAM_SRC = bytes(range(256)) * 4 + b"tail"  # 1028 bytes, no repeats per 256


def _writable_src() -> memoryview:
    return memoryview(bytearray(_STREAM_SRC))


class TestMemoryViewStreamReadinto:
    @pytest.mark.parametrize("make_src", [_writable_src, lambda: _STREAM_SRC])
    @pytest.mark.parametrize(
        "dst_size", [0, 7, len(_STREAM_SRC), len(_STREAM_SRC) + 100]
    )
    def test_first_read_copies_min_of_buffer_and_stream(self, make_src, dst_size):
        stream = s3mod.MemoryViewStream(make_src())
        sentinel = 0xAB
        dst = bytearray([sentinel]) * (dst_size + 8)
        n = stream.readinto(memoryview(dst)[:dst_size])
        expected = min(dst_size, len(_STREAM_SRC))
        assert n == expected
        assert stream.tell() == expected
        assert bytes(dst[:n]) == _STREAM_SRC[:n]
        # Nothing written past the copied bytes.
        assert dst[n:] == bytearray([sentinel]) * (len(dst) - n)

    @pytest.mark.parametrize("make_src", [_writable_src, lambda: _STREAM_SRC])
    def test_read_larger_than_remaining_stops_at_end(self, make_src):
        stream = s3mod.MemoryViewStream(make_src())
        stream.seek(1000)
        dst = bytearray(64)
        assert stream.readinto(dst) == len(_STREAM_SRC) - 1000
        assert bytes(dst[:28]) == _STREAM_SRC[1000:]
        assert dst[28:] == bytearray(36)
        assert stream.readinto(dst) == 0
        assert stream.tell() == len(_STREAM_SRC)

    @pytest.mark.parametrize("make_src", [_writable_src, lambda: _STREAM_SRC])
    def test_chunked_reads_reassemble_byte_identical(self, make_src):
        stream = s3mod.MemoryViewStream(make_src())
        out = bytearray()
        sizes = [1, 3, 255, 256, 0, 2, 1000]
        for size in sizes:
            buf = bytearray(size)
            n = stream.readinto(buf)
            out += buf[:n]
        assert bytes(out) == _STREAM_SRC

    def test_seek_back_rereads_same_bytes(self):
        stream = s3mod.MemoryViewStream(_writable_src())
        first = bytearray(100)
        stream.readinto(first)
        stream.seek(0)
        again = bytearray(100)
        assert stream.readinto(again) == 100
        assert again == first

    def test_matches_read(self):
        a = s3mod.MemoryViewStream(_writable_src())
        b = s3mod.MemoryViewStream(_writable_src())
        buf = bytearray(300)
        while True:
            n = a.readinto(buf)
            chunk = b.read(300)
            assert bytes(buf[:n]) == bytes(chunk)
            if n == 0:
                break

    def test_source_bytes_are_not_modified(self):
        src = bytearray(_STREAM_SRC)
        stream = s3mod.MemoryViewStream(src)
        stream.readinto(bytearray(len(src)))
        assert bytes(src) == _STREAM_SRC

    def test_memory_obj_buffer_is_copied_exactly(self):
        # PUT bodies are streams over ``MemoryObj.byte_array`` (float32 here).
        obj = create_memory_obj(size=64, fill_value=3.5)
        stream = s3mod.MemoryViewStream(obj.byte_array)
        buf = bytearray(len(stream))
        assert stream.readinto(buf) == len(buf) == 64 * 4
        assert bytes(buf) == bytes(memoryview(obj.byte_array).cast("B"))

    @pytest.mark.parametrize("offset, whence", [(-1, 0), (-5, 1), (-2000, 2)])
    def test_negative_seek_is_rejected(self, offset, whence):
        stream = s3mod.MemoryViewStream(_STREAM_SRC)
        stream.seek(4)
        with pytest.raises(ValueError, match="negative seek"):
            stream.seek(offset, whence)
        assert stream.tell() == 4

    def test_adapter_put_body_goes_through_readinto(self, adapter, monkeypatch):
        """Drain each PUT body with ``readinto`` the way awscrt's
        ``InputStream`` does when the stream has it; ``read`` must not run."""
        fake_request = s3mod.s3.S3Request

        def _read_not_expected(self, size=None):
            raise AssertionError("PUT body read through read(), not readinto()")

        def _request(**kwargs):
            req = kwargs["request"]
            if req.method == "PUT":
                body, out, chunk = req.body_stream, bytearray(), bytearray(100)
                while n := body.readinto(chunk):
                    out += chunk[:n]
                req.body_stream = io.BytesIO(bytes(out))
            return fake_request(**kwargs)

        monkeypatch.setattr(s3mod.MemoryViewStream, "read", _read_not_expected)
        monkeypatch.setattr(s3mod.s3, "S3Request", _request)
        key, obj = create_object_key(7), create_memory_obj(size=64, fill_value=2.5)
        adapter.submit_store_task([key], [obj])
        assert wait_for_event_fd(adapter.get_store_event_fd())
        assert _BACKEND.get(_object_key_to_string(key)) == bytes(
            memoryview(obj.byte_array).cast("B")
        )


# =============================================================================
# Separate metadata client
# =============================================================================


@pytest.fixture
def request_clients(monkeypatch) -> list[tuple[str, object]]:
    """Record ``(operation, client)`` for every S3Request the adapter makes.

    ``operation`` is the HTTP method, or ``LIST`` for ListObjectsV2.
    """
    seen: list[tuple[str, object]] = []

    def _recording_request(**kwargs):
        op = kwargs["request"].method
        if kwargs.get("operation_name") == "ListObjectsV2":
            op = "LIST"
        seen.append((op, kwargs["client"]))
        return _FakeS3Request(**kwargs)

    monkeypatch.setattr(s3mod.s3, "S3Request", _recording_request)
    return seen


def _exercise_every_request_type(adp: S3L2Adapter) -> None:
    """Drive one PUT, GET, HEAD, DELETE and LIST through ``adp``."""
    stored = create_object_key(1)
    adp.submit_store_task([stored], [create_memory_obj()])
    assert wait_for_event_fd(adp.get_store_event_fd())
    adp.pop_completed_store_tasks()

    adp.submit_load_task([stored], [create_memory_obj(fill_value=0.0)])
    assert wait_for_event_fd(adp.get_load_event_fd())

    # Written behind the adapter's back, so its size is not cached and
    # the lookup must HEAD it.
    foreign = create_object_key(2)
    _put_object(foreign, size=64)
    task = adp.submit_lookup_and_lock_task([foreign], {})
    assert wait_for_event_fd(adp.get_lookup_and_lock_event_fd())
    adp.query_lookup_and_lock_result(task)
    adp.submit_unlock([foreign])

    adp.delete([stored])
    adp.list_l2_keys()


# The objects both clients must share, compared by identity, and the
# plain values they must agree on.
_SHARED_CLIENT_OBJECTS = ("bootstrap", "tls_connection_options", "signing_config")
_SHARED_CLIENT_VALUES = ("region", "enable_s3express", "tls_mode")


class TestSeparateMetaClient:
    def test_flag_defaults_to_false(self):
        assert _sized_config().s3_separate_meta_client is False

    @pytest.mark.parametrize("bad", [1, "true", None])
    def test_non_bool_rejected(self, bad):
        with pytest.raises(ValueError, match="s3_separate_meta_client"):
            _sized_config(s3_separate_meta_client=bad)

    def test_disabled_routes_everything_to_one_client(
        self, s3_client_calls, request_clients
    ):
        """Contract test: it passes on the code before the meta client too.
        It pins that the flag off keeps one client for every request."""
        adp = S3L2Adapter(_sized_config())
        try:
            _exercise_every_request_type(adp)
        finally:
            adp.close()
        assert len(s3_client_calls) == 1
        ops = {op for op, _ in request_clients}
        assert ops == {"PUT", "GET", "HEAD", "DELETE", "LIST"}
        assert len({id(client) for _, client in request_clients}) == 1

    def test_enabled_routes_metadata_to_the_meta_client(
        self, s3_client_calls, request_clients
    ):
        adp = S3L2Adapter(_sized_config(s3_separate_meta_client=True))
        try:
            _exercise_every_request_type(adp)
        finally:
            adp.close()
        assert len(s3_client_calls) == 2
        bulk_kwargs, meta_kwargs = s3_client_calls
        assert set(bulk_kwargs) == _BASE_CLIENT_KWARGS
        assert set(meta_kwargs) == _BASE_CLIENT_KWARGS | {"throughput_target_gbps"}
        assert meta_kwargs["throughput_target_gbps"] == 4.0

        clients_by_op: dict[str, set[int]] = {}
        for op, client in request_clients:
            clients_by_op.setdefault(op, set()).add(id(client))
        assert set(clients_by_op) == {"PUT", "GET", "HEAD", "DELETE", "LIST"}
        bulk = clients_by_op["PUT"] | clients_by_op["GET"]
        meta = clients_by_op["HEAD"] | clients_by_op["DELETE"] | clients_by_op["LIST"]
        assert len(bulk) == 1 and len(meta) == 1
        assert bulk != meta

    @pytest.mark.parametrize("tls, s3express", [(True, True), (False, False)])
    def test_meta_client_shares_bootstrap_tls_and_signing(
        self, s3_client_calls, tls, s3express
    ):
        _build_and_close(
            _sized_config(
                s3_separate_meta_client=True,
                disable_tls=not tls,
                s3_prefer_http2=tls,
                s3_enable_s3express=s3express,
            )
        )
        bulk_kwargs, meta_kwargs = s3_client_calls
        for key in _SHARED_CLIENT_OBJECTS:
            assert meta_kwargs[key] is bulk_kwargs[key], key
        for key in _SHARED_CLIENT_VALUES:
            assert meta_kwargs[key] == bulk_kwargs[key], key
        expected_mode = (
            s3mod.s3.S3RequestTlsMode.ENABLED
            if tls
            else s3mod.s3.S3RequestTlsMode.DISABLED
        )
        assert meta_kwargs["tls_mode"] == expected_mode
        assert (meta_kwargs["tls_connection_options"] is not None) == tls
        assert (meta_kwargs["signing_config"] is not None) == s3express

    def test_meta_client_ignores_bulk_sizing(self, s3_client_calls):
        _build_and_close(
            _sized_config(
                s3_separate_meta_client=True,
                s3_throughput_target_gbps=100,
                s3_part_size_mb=16,
                s3_multipart_threshold_mb=256,
            )
        )
        bulk_kwargs, meta_kwargs = s3_client_calls
        assert bulk_kwargs["throughput_target_gbps"] == 100.0
        assert set(meta_kwargs) == _BASE_CLIENT_KWARGS | {"throughput_target_gbps"}
        assert meta_kwargs["throughput_target_gbps"] == 4.0

    @pytest.mark.parametrize("tls", [True, False])
    def test_real_awscrt_builds_and_releases_both_clients(self, tls):
        # No S3Client mock: two real CRT clients, with TLS on or off.
        adp = S3L2Adapter(
            _sized_config(
                s3_separate_meta_client=True,
                disable_tls=not tls,
                s3_prefer_http2=tls,
            )
        )
        try:
            assert adp._meta_client is not adp._s3_client
            shutdown_events = [
                adp._s3_client.shutdown_event,
                adp._meta_client.shutdown_event,
            ]
            _exercise_every_request_type(adp)
        finally:
            adp.close()
        gc.collect()
        assert all(event.wait(5) for event in shutdown_events)
