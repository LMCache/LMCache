# SPDX-License-Identifier: Apache-2.0
"""
Tests verifying the S3 L2 adapter works correctly with MinIO-style
configuration (``disable_tls=true``, static credentials, non-AWS
endpoint).

Uses the same in-memory fake S3 backend as ``test_s3_l2_adapter.py`` —
no network or MinIO server required.
"""

# Standard
from concurrent.futures import Future as ConcurrentFuture
import select
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
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
pytestmark = pytest.mark.no_shared_allocator

# =============================================================================
# Fake S3 backend (shared with test_s3_l2_adapter.py pattern)
# =============================================================================


class _FakeBackend:
    """In-memory backing store shared by all fake S3Requests in a test."""

    def __init__(self):
        self._data: dict[str, bytes] = {}
        self._lock = threading.Lock()
        self._inject_error: str | None = None

    def reset(self):
        with self._lock:
            self._data.clear()
            self._inject_error = None

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


_BACKEND = _FakeBackend()


def _path_to_key(path: str) -> str:
    # Standard
    from urllib.parse import unquote

    return unquote(path.lstrip("/"))


class _FakeS3Request:
    """In-process substitute for awscrt.s3.S3Request."""

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
        method = request.method
        path = request.path
        key_str = _path_to_key(path)

        err_inj = _BACKEND._inject_error

        try:
            if err_inj:
                raise RuntimeError(err_inj)

            if method == "PUT":
                body = request.body_stream
                body.seek(0)
                data = bytes(body.read())
                _BACKEND.put(key_str, data)
                if on_headers is not None:
                    on_headers(200, [])
                if on_done is not None:
                    on_done(error=None, status_code=200)
                self.finished_future.set_result(None)

            elif method == "GET":
                data = _BACKEND.get(key_str)
                if data is None:
                    if on_done is not None:
                        try:
                            on_done(error=None, status_code=404)
                        except Exception:
                            pass
                    self.finished_future.set_exception(
                        RuntimeError(f"S3 GET 404 for {key_str}")
                    )
                    return
                if on_body is not None:
                    on_body(data, 0)
                if on_done is not None:
                    on_done(error=None, status_code=200)
                self.finished_future.set_result(None)

            elif method == "HEAD":
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
                _BACKEND.delete(key_str)
                if on_headers is not None:
                    on_headers(204, [])
                if on_done is not None:
                    on_done(error=None, status_code=204)
                self.finished_future.set_result(None)

            else:
                self.finished_future.set_exception(
                    RuntimeError(f"unexpected method {method}")
                )

        except Exception as e:
            try:
                if on_done is not None:
                    on_done(error=str(e), status_code=None)
            except Exception:
                pass
            if not self.finished_future.done():
                self.finished_future.set_exception(e)


class _FakeHttpRequest:
    """Plain-Python HttpRequest that preserves body_stream as-is."""

    def __init__(self, method, path, headers, body_stream=None):
        self.method = method
        self.path = path
        self.headers = headers
        self.body_stream = body_stream


@pytest.fixture(autouse=True)
def patch_s3_request(monkeypatch):
    """Replace awscrt S3Request with in-memory fake; stub credentials."""
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
    """Create an ObjectKey with a deterministic chunk hash."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
    )


def create_memory_obj(size: int = 16, fill_value: float = 1.0) -> TensorMemoryObj:
    """Create a TensorMemoryObj with known content for roundtrip tests."""
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


def wait_for_event_fd(event_fd: int, timeout: float = 5.0) -> bool:
    """Wait for an event fd to become readable."""
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


def _minio_config(
    s3_endpoint: str = "lmcache-kv.localhost:9000",
    s3_region: str = "us-east-1",
    s3_num_io_threads: int = 1,
    s3_prefer_http2: bool = False,
    s3_enable_s3express: bool = False,
    disable_tls: bool = True,
    aws_access_key_id: str | None = "minioadmin",
    aws_secret_access_key: str | None = "minioadmin",
    max_capacity_gb: float = 0.001,
) -> S3L2AdapterConfig:
    """Build a MinIO-style S3L2AdapterConfig with sensible defaults."""
    return S3L2AdapterConfig(
        s3_endpoint=s3_endpoint,
        s3_region=s3_region,
        s3_num_io_threads=s3_num_io_threads,
        s3_prefer_http2=s3_prefer_http2,
        s3_enable_s3express=s3_enable_s3express,
        disable_tls=disable_tls,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        max_capacity_gb=max_capacity_gb,
    )


@pytest.fixture
def minio_adapter():
    """Create an S3L2Adapter configured for MinIO."""
    a = S3L2Adapter(_minio_config())
    yield a
    a.close()


# =============================================================================
# Config parsing tests (MinIO-specific patterns)
# =============================================================================


class TestMinIOConfig:
    def test_from_dict_minio_style(self):
        """Verify from_dict parses a typical MinIO config correctly."""
        cfg = S3L2AdapterConfig.from_dict(
            {
                "type": "s3",
                "s3_endpoint": "lmcache-kv.localhost:9000",
                "s3_region": "us-east-1",
                "disable_tls": True,
                "aws_access_key_id": "minioadmin",
                "aws_secret_access_key": "minioadmin",
            }
        )
        assert cfg.s3_endpoint == "lmcache-kv.localhost:9000"
        assert cfg.s3_region == "us-east-1"
        assert cfg.disable_tls is True
        assert cfg.aws_access_key_id == "minioadmin"
        assert cfg.aws_secret_access_key == "minioadmin"
        assert cfg.s3_enable_s3express is False

    def test_from_dict_rejects_missing_endpoint(self):
        """MinIO configs still require s3_endpoint."""
        with pytest.raises(ValueError, match="s3_endpoint"):
            S3L2AdapterConfig.from_dict(
                {
                    "type": "s3",
                    "s3_region": "us-east-1",
                    "disable_tls": True,
                }
            )

    def test_from_dict_rejects_missing_region(self):
        """MinIO configs still require s3_region for SigV4 signing."""
        with pytest.raises(ValueError, match="s3_region"):
            S3L2AdapterConfig.from_dict(
                {
                    "type": "s3",
                    "s3_endpoint": "lmcache-kv.localhost:9000",
                    "disable_tls": True,
                }
            )

    def test_disable_tls_defaults_to_false(self):
        """When disable_tls is omitted, it defaults to False (TLS enabled)."""
        cfg = S3L2AdapterConfig.from_dict(
            {
                "type": "s3",
                "s3_endpoint": "bucket.s3.us-west-2.amazonaws.com",
                "s3_region": "us-west-2",
            }
        )
        assert cfg.disable_tls is False

    def test_non_aws_endpoint_without_s3_prefix(self):
        """MinIO endpoints don't need the s3:// prefix."""
        cfg = S3L2AdapterConfig(
            s3_endpoint="minio.local:9000",
            s3_region="us-east-1",
            disable_tls=True,
            aws_access_key_id="minio",
            aws_secret_access_key="minio123",
        )
        assert cfg.s3_endpoint == "minio.local:9000"
        assert cfg.disable_tls is True

    def test_minio_config_with_eviction(self):
        """Verify eviction config is parsed correctly alongside MinIO settings."""
        cfg = S3L2AdapterConfig.from_dict(
            {
                "type": "s3",
                "s3_endpoint": "lmcache-kv.localhost:9000",
                "s3_region": "us-east-1",
                "disable_tls": True,
                "aws_access_key_id": "minioadmin",
                "aws_secret_access_key": "minioadmin",
                "max_capacity_gb": 50.0,
            }
        )
        assert cfg.max_capacity_gb == 50.0


# =============================================================================
# Adapter construction tests
# =============================================================================


class TestMinIOAdapterConstruction:
    def test_adapter_initializes_with_minio_config(self):
        """The adapter should initialize successfully with MinIO-style config."""
        cfg = _minio_config()
        adapter = S3L2Adapter(cfg)
        try:
            status = adapter.report_status()
            assert status["is_healthy"] is True
            assert status["type"] == "S3L2Adapter"
            assert status["connection_disabled"] is False
        finally:
            adapter.close()

    def test_adapter_reports_endpoint(self):
        """report_status should reflect the MinIO endpoint."""
        cfg = _minio_config()
        adapter = S3L2Adapter(cfg)
        try:
            status = adapter.report_status()
            assert "localhost" in status["endpoint"]
        finally:
            adapter.close()

    def test_event_fds_are_valid(self, minio_adapter):
        """All three event fds must be distinct and non-negative."""
        a = minio_adapter.get_store_event_fd()
        b = minio_adapter.get_lookup_and_lock_event_fd()
        c = minio_adapter.get_load_event_fd()
        assert a >= 0 and b >= 0 and c >= 0
        assert len({a, b, c}) == 3


# =============================================================================
# Roundtrip tests with MinIO-style adapter
# =============================================================================


class TestMinIORoundtrip:
    def test_store_lookup_load(self, minio_adapter):
        """Full roundtrip: store → lookup → load with MinIO config."""
        key = create_object_key(1)
        obj = create_memory_obj(fill_value=2.718)

        # Store
        tid = minio_adapter.submit_store_task([key], [obj])
        assert wait_for_event_fd(minio_adapter.get_store_event_fd())
        completed = minio_adapter.pop_completed_store_tasks()
        assert completed[tid].is_successful()

        # Lookup
        tid = minio_adapter.submit_lookup_and_lock_task([key], {0: _EMPTY_LAYOUT})
        assert wait_for_event_fd(minio_adapter.get_lookup_and_lock_event_fd())
        bm = minio_adapter.query_lookup_and_lock_result(tid)
        assert bm is not None and bm.test(0) is True

        # Load and verify
        dst = create_memory_obj(fill_value=0.0)
        tid = minio_adapter.submit_load_task([key], [dst])
        assert wait_for_event_fd(minio_adapter.get_load_event_fd())
        bm = minio_adapter.query_load_result(tid)
        assert bm is not None and bm.test(0) is True
        assert torch.allclose(dst.tensor, torch.full((16,), 2.718))

    def test_multiple_keys(self, minio_adapter):
        """Store and retrieve multiple keys."""
        keys = [create_object_key(i) for i in range(5)]
        objs = [create_memory_obj(fill_value=float(i)) for i in range(5)]

        # Store all
        tid = minio_adapter.submit_store_task(keys, objs)
        assert wait_for_event_fd(minio_adapter.get_store_event_fd())
        completed = minio_adapter.pop_completed_store_tasks()
        assert completed[tid].is_successful()

        # Lookup all — expect all hits
        tid = minio_adapter.submit_lookup_and_lock_task(keys, {0: _EMPTY_LAYOUT})
        assert wait_for_event_fd(minio_adapter.get_lookup_and_lock_event_fd())
        bm = minio_adapter.query_lookup_and_lock_result(tid)
        assert bm is not None
        for i in range(5):
            assert bm.test(i) is True

    def test_lookup_miss(self, minio_adapter):
        """Lookup on a key that was never stored returns miss."""
        key = create_object_key(99)
        tid = minio_adapter.submit_lookup_and_lock_task([key], {0: _EMPTY_LAYOUT})
        assert wait_for_event_fd(minio_adapter.get_lookup_and_lock_event_fd())
        bm = minio_adapter.query_lookup_and_lock_result(tid)
        assert bm is not None and bm.test(0) is False


# =============================================================================
# Eviction with MinIO config
# =============================================================================


class TestMinIOEviction:
    def _store(self, adapter, key, obj):
        adapter.submit_store_task([key], [obj])
        wait_for_event_fd(adapter.get_store_event_fd())
        adapter.pop_completed_store_tasks()

    def test_delete_removes_key(self, minio_adapter):
        """Delete should remove the object from the backend."""
        key = create_object_key(1)
        self._store(minio_adapter, key, create_memory_obj())
        assert _BACKEND.contains(_object_key_to_string(key))
        minio_adapter.delete([key])
        assert not _BACKEND.contains(_object_key_to_string(key))

    def test_usage_tracking_with_minio(self, minio_adapter):
        """Capacity tracking works with MinIO-style config."""
        keys = [create_object_key(i) for i in range(3)]
        objs = [create_memory_obj() for _ in range(3)]

        minio_adapter.submit_store_task(keys, objs)
        wait_for_event_fd(minio_adapter.get_store_event_fd())
        minio_adapter.pop_completed_store_tasks()

        usage = minio_adapter.get_usage()
        assert usage.total_bytes_used == 3 * 64  # 16 floats * 4 bytes * 3 keys
        assert usage.total_bytes_used > 0

        minio_adapter.delete(keys)
        usage = minio_adapter.get_usage()
        assert usage.total_bytes_used == 0


# =============================================================================
# Circuit breaker with MinIO config
# =============================================================================


class TestMinIOCircuitBreaker:
    def test_connection_errors_trip_breaker(self, minio_adapter):
        """Circuit breaker triggers after repeated connection failures."""
        _BACKEND.set_error("CONNECTION_REFUSED: mock minio down")

        for i in range(3):
            obj = create_memory_obj()
            minio_adapter.submit_store_task([create_object_key(i)], [obj])
            wait_for_event_fd(minio_adapter.get_store_event_fd(), timeout=2.0)
            minio_adapter.pop_completed_store_tasks()

        status = minio_adapter.report_status()
        assert status["connection_disabled"] is True
        assert status["is_healthy"] is False


# =============================================================================
# Factory registration
# =============================================================================


class TestMinIOFactoryRegistration:
    def test_create_via_factory_with_minio_config(self):
        """The 's3' factory registration works with MinIO-style config."""
        # First Party
        from lmcache.v1.distributed.l2_adapters import create_l2_adapter

        cfg = S3L2AdapterConfig.from_dict(
            {
                "type": "s3",
                "s3_endpoint": "lmcache-kv.localhost:9000",
                "s3_region": "us-east-1",
                "disable_tls": True,
                "aws_access_key_id": "minioadmin",
                "aws_secret_access_key": "minioadmin",
                "s3_prefer_http2": False,
                "s3_num_io_threads": 1,
            }
        )
        adapter = create_l2_adapter(cfg)
        try:
            assert isinstance(adapter, S3L2Adapter)
            status = adapter.report_status()
            assert status["is_healthy"] is True
        finally:
            adapter.close()
