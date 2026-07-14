# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for GCSL2Adapter.

``google.cloud.storage`` is replaced with an in-memory fake injected into
``sys.modules`` before the adapter is instantiated.  No network or GCS
credentials required.
"""

# Standard
import select
import sys
import threading
import time
import types

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.distributed.l2_adapters.gcs_l2_adapter import (
    GCSL2Adapter,
    GCSL2AdapterConfig,
    _object_key_to_string,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

# =============================================================================
# In-memory fake GCS backend
# =============================================================================


class _FakeBackend:
    """In-memory backing store shared by all fake GCS calls in a test."""

    def __init__(self):
        self._data: dict[str, bytes] = {}
        self._lock = threading.Lock()
        self._inject_error: str | None = None
        self._put_count = 0
        self._get_count = 0
        self._delete_count = 0
        self._head_count = 0

    def reset(self):
        with self._lock:
            self._data.clear()
            self._inject_error = None
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

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {
                "put": self._put_count,
                "get": self._get_count,
                "delete": self._delete_count,
                "head": self._head_count,
            }


_BACKEND = _FakeBackend()


# =============================================================================
# Fake GCS objects (Blob, Bucket, Client)
# =============================================================================


class _FakeBlob:
    """Minimal substitute for ``google.cloud.storage.Blob``."""

    def __init__(self, name: str, backend: _FakeBackend):
        self.name = name
        self._backend = backend
        self.size: int | None = None

    def upload_from_string(self, data, content_type=None):
        with self._backend._lock:
            err = self._backend._inject_error
            self._backend._put_count += 1
        if err:
            raise RuntimeError(err)
        raw = data if isinstance(data, bytes) else bytes(data)
        self._backend.put(self.name, raw)

    def download_as_bytes(self) -> bytes:
        with self._backend._lock:
            err = self._backend._inject_error
            self._backend._get_count += 1
        if err:
            raise RuntimeError(err)
        data = self._backend.get(self.name)
        if data is None:
            raise RuntimeError(f"404 GET {self.name}: Not Found")
        return data

    def delete(self):
        with self._backend._lock:
            self._backend._delete_count += 1
        self._backend.delete(self.name)


class _FakeBucket:
    """Minimal substitute for ``google.cloud.storage.Bucket``."""

    def __init__(self, backend: _FakeBackend):
        self._backend = backend

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(name, self._backend)

    def get_blob(self, name: str) -> _FakeBlob | None:
        with self._backend._lock:
            err = self._backend._inject_error
            self._backend._head_count += 1
        if err:
            raise RuntimeError(err)
        data = self._backend.get(name)
        if data is None:
            return None
        b = _FakeBlob(name, self._backend)
        b.size = len(data)
        return b


class _FakeGCSClient:
    """Minimal substitute for ``google.cloud.storage.Client``."""

    def __init__(self, backend: _FakeBackend, project=None, credentials=None):
        self._backend = backend

    def bucket(self, name: str) -> _FakeBucket:
        return _FakeBucket(self._backend)

    def close(self):
        pass


# =============================================================================
# sys.modules injection fixture
# =============================================================================


@pytest.fixture(autouse=True)
def patch_gcs(monkeypatch):
    """Replace google.cloud.storage in sys.modules with an in-memory fake.

    This lets GCSL2Adapter.__init__'s lazy ``from google.cloud import
    storage`` succeed without the real google-cloud-storage package.
    """
    _BACKEND.reset()

    # Build the fake module hierarchy.
    fake_storage = types.ModuleType("google.cloud.storage")
    fake_storage.Client = lambda *args, **kwargs: _FakeGCSClient(
        _BACKEND, *args, **kwargs
    )

    fake_sa = types.ModuleType("google.oauth2.service_account")

    class _FakeCreds:
        @staticmethod
        def from_service_account_file(path):
            return _FakeCreds()

    fake_sa.Credentials = _FakeCreds

    # Inject into sys.modules, creating parent packages if absent.
    if "google" not in sys.modules:
        monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    if "google.cloud" not in sys.modules:
        gc_mod = types.ModuleType("google.cloud")
        monkeypatch.setitem(sys.modules, "google.cloud", gc_mod)
    # Ensure `from google.cloud import storage` resolves to our fake.
    # raising=False: google.cloud may be a real namespace pkg with no
    # storage attr when google-cloud-storage is not installed.
    monkeypatch.setattr(
        sys.modules["google.cloud"], "storage", fake_storage, raising=False
    )
    monkeypatch.setitem(sys.modules, "google.cloud.storage", fake_storage)

    if "google.oauth2" not in sys.modules:
        monkeypatch.setitem(
            sys.modules, "google.oauth2", types.ModuleType("google.oauth2")
        )
    monkeypatch.setattr(
        sys.modules["google.oauth2"], "service_account", fake_sa, raising=False
    )
    monkeypatch.setitem(sys.modules, "google.oauth2.service_account", fake_sa)

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
    config = GCSL2AdapterConfig(
        gcs_bucket="test-bucket",
        gcs_num_workers=2,
        max_capacity_gb=0.001,  # 1 MB
    )
    a = GCSL2Adapter(config)
    yield a
    a.close()


class _RecordingListener(L2AdapterListener):
    def __init__(self):
        self.stored: list[list[ObjectKey]] = []
        self.accessed: list[list[ObjectKey]] = []
        self.deleted: list[list[ObjectKey]] = []

    def on_l2_keys_stored(self, keys):
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
        assert _object_key_to_string(key) == "llama@000000ff@00010203"

    def test_cache_salt_appended(self):
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
        assert _object_key_to_string(base_key) == "llama@000000ff@00010203"
        assert _object_key_to_string(salted) == "llama@000000ff@00010203@user-42"
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
        tid = adapter.submit_lookup_and_lock_task([key])
        assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        bm = adapter.query_lookup_and_lock_result(tid)
        assert bm is not None and bm.test(0) is True

        # Load into a fresh buffer and verify the data matches.
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
        tid = adapter.submit_lookup_and_lock_task(keys)
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
        tid = adapter.submit_lookup_and_lock_task([key])
        wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        assert adapter.query_lookup_and_lock_result(tid) is not None
        assert adapter.query_lookup_and_lock_result(tid) is None


# =============================================================================
# Eviction (delete + locking)
# =============================================================================


class TestEviction:
    def _store(self, adapter, key, obj):
        adapter.submit_store_task([key], [obj])
        wait_for_event_fd(adapter.get_store_event_fd())
        adapter.pop_completed_store_tasks()

    def _lookup(self, adapter, key):
        tid = adapter.submit_lookup_and_lock_task([key])
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
        adapter.delete([create_object_key(42)])  # must not raise


# =============================================================================
# get_usage
# =============================================================================


class TestGetUsage:
    def test_disabled_returns_minus_one(self):
        cfg = GCSL2AdapterConfig(gcs_bucket="b", gcs_num_workers=1, max_capacity_gb=0.0)
        a = GCSL2Adapter(cfg)
        try:
            usage = a.get_usage()
            assert usage.usage_fraction == -1.0
            assert usage.total_bytes_used == 0
            assert usage.total_capacity_bytes == 0
        finally:
            a.close()

    def test_usage_grows_on_store_and_shrinks_on_delete(self, adapter):
        # adapter max_capacity_gb = 0.001 = 1 MB; each obj = 16 floats = 64 B
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

        for i in range(3):
            obj = create_memory_obj()
            adapter.submit_store_task([create_object_key(i)], [obj])
            wait_for_event_fd(adapter.get_store_event_fd(), timeout=2.0)
            adapter.pop_completed_store_tasks()

        status = adapter.report_status()
        assert status["connection_disabled"] is True
        assert status["is_healthy"] is False

        # Subsequent submits short-circuit without touching the backend.
        _BACKEND.set_error(None)
        puts_before = _BACKEND.counts()["put"]
        disabled_tid = adapter.submit_store_task(
            [create_object_key(42)], [create_memory_obj()]
        )
        wait_for_event_fd(adapter.get_store_event_fd(), timeout=2.0)
        completed = adapter.pop_completed_store_tasks()
        assert not completed[disabled_tid].is_successful()
        assert _BACKEND.counts()["put"] == puts_before

        # Lookup and load also short-circuit to all-zero bitmaps.
        tid = adapter.submit_lookup_and_lock_task([create_object_key(1)])
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

        tid = adapter.submit_lookup_and_lock_task([key])
        wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        adapter.query_lookup_and_lock_result(tid)
        time.sleep(0.05)
        assert any(key in batch for batch in listener.accessed)


# =============================================================================
# Config
# =============================================================================


class TestConfig:
    def test_from_dict_requires_bucket(self):
        with pytest.raises(ValueError):
            GCSL2AdapterConfig.from_dict({})
        with pytest.raises(ValueError):
            GCSL2AdapterConfig.from_dict({"gcs_bucket": ""})

    def test_from_dict_parses_all_fields(self):
        cfg = GCSL2AdapterConfig.from_dict(
            {
                "type": "gcs",
                "gcs_bucket": "my-bucket",
                "gcs_credentials_file": "/path/to/creds.json",
                "gcs_project": "my-project",
                "gcs_num_workers": 32,
                "max_capacity_gb": 2.5,
            }
        )
        assert cfg.gcs_bucket == "my-bucket"
        assert cfg.gcs_credentials_file == "/path/to/creds.json"
        assert cfg.gcs_project == "my-project"
        assert cfg.gcs_num_workers == 32
        assert cfg.max_capacity_gb == 2.5

    def test_from_dict_defaults(self):
        cfg = GCSL2AdapterConfig.from_dict({"gcs_bucket": "b"})
        assert cfg.gcs_credentials_file is None
        assert cfg.gcs_project is None
        assert cfg.gcs_num_workers == 64
        assert cfg.max_capacity_gb == 0.0

    def test_invalid_num_workers(self):
        with pytest.raises(ValueError):
            GCSL2AdapterConfig.from_dict({"gcs_bucket": "b", "gcs_num_workers": 0})

    def test_help_nonempty(self):
        h = GCSL2AdapterConfig.help()
        assert isinstance(h, str)
        assert "gcs_bucket" in h


# =============================================================================
# Factory registration
# =============================================================================


class TestFactoryRegistration:
    def test_create_l2_adapter_registers_gcs(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters import create_l2_adapter

        cfg = GCSL2AdapterConfig.from_dict(
            {
                "type": "gcs",
                "gcs_bucket": "fac-test",
                "gcs_num_workers": 1,
            }
        )
        adp = create_l2_adapter(cfg)
        try:
            assert isinstance(adp, GCSL2Adapter)
        finally:
            adp.close()
