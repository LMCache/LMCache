# SPDX-License-Identifier: Apache-2.0
"""
Tests for the autumn-rs L2 adapter.

The unit tests replace the ``autumn`` client module with an in-memory fake, so
they need neither a cluster nor the client. ``TestAutumnL2AdapterIntegration``
runs against a real cluster when ``AUTUMN_MANAGER`` (``ip:port``) is set, with
``AUTUMN_CREDENTIAL_FILE`` for a protected namespace.
"""

# Standard
from types import ModuleType
import asyncio
import ctypes
import os
import select
import sys
import threading
import uuid

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.key_codec import encode_storage_key
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.autumn_l2_adapter import (
    AutumnL2AdapterConfig,
    read_credential_file,
)
from lmcache.v1.distributed.l2_adapters.config import (
    get_registered_l2_adapter_types,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])


# =============================================================================
# Fake autumn client module
# =============================================================================


# The formats PyO3's ``PyBuffer<u8>`` accepts on a little-endian host.
_PYO3_U8_FORMATS = ("B", "@B", "=B", ">B")


def _check_buffers(bufs) -> None:
    for b in bufs:
        if memoryview(b).format not in _PYO3_U8_FORMATS:
            raise BufferError("buffer contents are not compatible with u8")


class _FakeStore:
    def __init__(self) -> None:
        self.data: dict[bytes, bytes] = {}
        self.fail_puts = False
        self.raise_gets = False
        self.connects = 0
        self.ttls: list[int] = []
        self.scopes: list[str] = []
        self.auth: list[tuple] = []
        self.regions: list = []
        self.registered: list = []


class _FakeBatchClient:
    def __init__(
        self,
        store,
        manager,
        per_worker_cap=16,
        direct=False,
        *,
        scope,
        principal=None,
        credential=None,
    ):
        if (principal is None) != (credential is None):
            raise ValueError("principal and credential are both-or-neither")
        self._store = store
        self._scope = scope
        store.scopes.append(scope)
        store.auth.append((principal, credential))

    def _full(self, key: bytes) -> bytes:
        return self._scope.encode() + b"/" + key

    def put_from(self, keys, bufs, ttl_secs=0, region=None):
        _check_buffers(bufs)
        self._store.ttls.append(ttl_secs)
        self._store.regions.append(region)
        if self._store.fail_puts:
            return [False] * len(keys)
        for k, b in zip(keys, bufs, strict=True):
            self._store.data[self._full(k)] = bytes(b)
        return [True] * len(keys)

    def get_into(self, keys, bufs, region=None):
        _check_buffers(bufs)
        self._store.regions.append(region)
        if self._store.raise_gets:
            raise RuntimeError("cluster unreachable")
        out = []
        for k, b in zip(keys, bufs, strict=True):
            v = self._store.data.get(self._full(k))
            if v is None or len(v) != b.nbytes:
                out.append(False)
                continue
            ctypes.memmove(ctypes.addressof(ctypes.c_char.from_buffer(b)), v, len(v))
            out.append(True)
        return out

    def exists(self, keys):
        return [self._full(k) in self._store.data for k in keys]


class _FakeRegion:
    def __init__(self, buf):
        _check_buffers([buf])
        mv = memoryview(buf)
        self.nbytes = mv.nbytes
        self.ptr = ctypes.addressof(ctypes.c_char.from_buffer(mv))
        self.unregistered = False
        _FAKE_STORE.registered.append(self)

    def registered(self):
        return True

    def unregister(self):
        self.unregistered = True


class _FakeClient:
    def __init__(self, store, scope):
        self._store = store
        self._scope = scope

    @classmethod
    async def connect(cls, manager, scope, principal=None, credential=None):
        _FAKE_STORE.connects += 1
        await asyncio.sleep(0.05)
        return cls(_FAKE_STORE, scope)

    async def delete(self, key):
        self._store.data.pop(self._scope.encode() + b"/" + key, None)

    async def close(self):
        pass


_FAKE_STORE = _FakeStore()


@pytest.fixture
def fake_autumn(monkeypatch):
    global _FAKE_STORE
    _FAKE_STORE = _FakeStore()
    mod = ModuleType("autumn")
    mod.BatchClient = lambda *a, **kw: _FakeBatchClient(_FAKE_STORE, *a, **kw)
    mod.Client = _FakeClient
    mod.transports = []
    mod.set_transport = mod.transports.append
    mod.register = _FakeRegion
    monkeypatch.setitem(sys.modules, "autumn", mod)
    return _FAKE_STORE, mod


# =============================================================================
# Helpers
# =============================================================================


def create_object_key(chunk_id: int, model: str = "org/model") -> ObjectKey:
    # Spread the ids over the hash space, as real chunk hashes are.
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes((chunk_id * 0x9E3779B1) & 0xFFFFFFFF),
        model_name=model,
        kv_rank=0,
    )


def create_memory_obj(size: int = 256, fill_value: float = 1.0) -> TensorMemoryObj:
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


def wait_for_event_fd(event_fd: int, timeout: float = 10.0) -> bool:
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    if poll.poll(timeout * 1000):
        try:
            consume_fd(event_fd)
        except BlockingIOError:
            pass
        return True
    return False


def store(adapter, keys, objs) -> bool:
    tid = adapter.submit_store_task(keys, objs)
    assert wait_for_event_fd(adapter.get_store_event_fd())
    return adapter.pop_completed_store_tasks()[tid].is_successful()


def lookup(adapter, keys) -> list[bool]:
    tid = adapter.submit_lookup_and_lock_task(keys, {0: _EMPTY_LAYOUT})
    assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
    bitmap = adapter.query_lookup_and_lock_result(tid)
    assert bitmap is not None
    return [bitmap.test(i) for i in range(len(keys))]


def load(adapter, keys, objs) -> list[bool]:
    tid = adapter.submit_load_task(keys, objs)
    assert wait_for_event_fd(adapter.get_load_event_fd())
    bitmap = adapter.query_load_result(tid)
    assert bitmap is not None
    return [bitmap.test(i) for i in range(len(keys))]


def make_adapter(l1_memory_desc=None, **fields):
    cfg = AutumnL2AdapterConfig.from_dict({"manager": "127.0.0.1:9001", **fields})
    return create_l2_adapter(cfg, l1_memory_desc)


def l1_desc(arena: torch.Tensor) -> L1MemoryDesc:
    return L1MemoryDesc(ptr=arena.data_ptr(), size=arena.nbytes, align_bytes=64)


# =============================================================================
# Config
# =============================================================================


class TestAutumnL2AdapterConfig:
    def test_registered(self):
        assert "autumn" in get_registered_l2_adapter_types()

    def test_defaults(self):
        cfg = AutumnL2AdapterConfig.from_dict({"manager": "10.0.0.1:9001"})
        assert cfg.namespace == "kvc"
        assert cfg.credential_file == ""
        assert cfg.num_workers == 4
        assert cfg.transport == "tcp"
        assert cfg.direct is False
        assert cfg.ttl_secs == 0
        assert cfg.max_capacity_gb == 0

    @pytest.mark.parametrize(
        "fields",
        [
            {},
            {"manager": ""},
            {"manager": "m:1", "namespace": "a/b"},
            {"manager": "m:1", "namespace": ""},
            {"manager": "m:1", "num_workers": 0},
            {"manager": "m:1", "transport": "rdma"},
            {"manager": "m:1", "direct": "yes"},
            {"manager": "m:1", "ttl_secs": -1},
            {"manager": "m:1", "max_capacity_gb": -1},
        ],
    )
    def test_rejects_invalid(self, fields):
        with pytest.raises(ValueError):
            AutumnL2AdapterConfig.from_dict(fields)

    def test_help_names_every_field(self):
        text = AutumnL2AdapterConfig.help()
        for name in (
            "manager",
            "namespace",
            "credential_file",
            "num_workers",
            "transport",
            "direct",
            "ttl_secs",
            "max_capacity_gb",
        ):
            assert name in text


class TestCredentialFile:
    def test_labeled(self, tmp_path):
        p = tmp_path / "kvc.cred"
        p.write_text("principal: kvc\ncredential: 00ff10\n")
        assert read_credential_file(str(p)) == ("kvc", b"\x00\xff\x10")

    def test_two_bare_lines(self, tmp_path):
        p = tmp_path / "kvc.cred"
        p.write_text("kvc\nabcd\n")
        assert read_credential_file(str(p)) == ("kvc", b"\xab\xcd")

    def test_no_principal_is_rejected(self, tmp_path):
        p = tmp_path / "kvc.cred"
        p.write_text("credential: abcd\n")
        with pytest.raises(ValueError):
            read_credential_file(str(p))


# =============================================================================
# Keys
# =============================================================================


def test_keys_start_with_the_hash_digit(fake_autumn):
    store_, _ = fake_autumn
    keys = [create_object_key(i) for i in range(64)]
    adapter = make_adapter()
    try:
        assert store(adapter, keys, [create_memory_obj(4) for _ in keys])
    finally:
        adapter.close()
    expected = {
        ("kvc/%s/lmcache/%s" % (k.chunk_hash.hex()[0], encode_storage_key(k))).encode()
        for k in keys
    }
    assert set(store_.data) == expected
    assert len({k[4:5] for k in expected}) > 1


# =============================================================================
# Adapter over the fake client
# =============================================================================


class TestAutumnL2Adapter:
    def test_store_lookup_load_roundtrip(self, fake_autumn):
        adapter = make_adapter(ttl_secs=600)
        try:
            keys = [create_object_key(i) for i in range(8)]
            objs = [create_memory_obj(64, float(i)) for i in range(8)]
            assert store(adapter, keys, objs)
            assert lookup(adapter, keys) == [True] * 8
            out = [create_memory_obj(64, 0.0) for _ in range(8)]
            assert load(adapter, keys, out) == [True] * 8
            for a, b in zip(objs, out, strict=True):
                assert torch.equal(a.tensor, b.tensor)
            adapter.submit_unlock(keys)
        finally:
            adapter.close()
        store_, _ = fake_autumn
        assert store_.ttls == [600]
        assert store_.scopes == ["kvc"]
        assert all(k.startswith(b"kvc/") and b"/lmcache/" in k for k in store_.data)

    def test_missing_and_wrong_size_values_are_misses(self, fake_autumn):
        adapter = make_adapter()
        try:
            stored = create_object_key(1)
            assert store(adapter, [stored], [create_memory_obj(64)])
            missing = create_object_key(2)
            assert lookup(adapter, [stored, missing]) == [True, False]
            # A buffer of another size must not take the value.
            out = [create_memory_obj(32), create_memory_obj(64)]
            assert load(adapter, [stored, missing], out) == [False, False]
        finally:
            adapter.close()

    def test_failed_put_fails_the_store(self, fake_autumn):
        store_, _ = fake_autumn
        store_.fail_puts = True
        adapter = make_adapter()
        try:
            assert not store(adapter, [create_object_key(1)], [create_memory_obj()])
        finally:
            adapter.close()

    def test_delete_removes_the_keys_and_their_usage(self, fake_autumn):
        adapter = make_adapter(max_capacity_gb=1)
        try:
            keys = [create_object_key(i) for i in range(4)]
            assert store(adapter, keys, [create_memory_obj(64) for _ in keys])
            used_before = adapter.get_usage().total_bytes_used
            assert used_before > 0
            adapter.delete(keys[:2])
            assert lookup(adapter, keys) == [False, False, True, True]
            assert adapter.get_usage().total_bytes_used < used_before
        finally:
            adapter.close()

    def test_credential_file_binds_the_client(self, fake_autumn, tmp_path):
        p = tmp_path / "kvc.cred"
        p.write_text("principal: kvc\ncredential: abcd\n")
        adapter = make_adapter(credential_file=str(p))
        adapter.close()
        store_, _ = fake_autumn
        assert store_.auth == [("kvc", b"\xab\xcd")]

    def test_transport_is_set_before_connecting(self, fake_autumn):
        _, mod = fake_autumn
        arena = torch.empty(4096, dtype=torch.uint8)
        make_adapter(l1_desc(arena), transport="ucx").close()
        assert mod.transports == ["ucx"]
        make_adapter().close()
        assert mod.transports == ["ucx"]

    def test_ucx_moves_values_through_the_registered_l1_arena(self, fake_autumn):
        store_, _ = fake_autumn
        arena = torch.empty(1 << 20, dtype=torch.uint8)
        adapter = make_adapter(l1_desc(arena), transport="ucx")
        try:
            key = create_object_key(1)
            assert store(adapter, [key], [create_memory_obj(64)])
            assert load(adapter, [key], [create_memory_obj(64)]) == [True]
        finally:
            adapter.close()
        region = store_.regions[0]
        assert (region.ptr, region.nbytes) == (arena.data_ptr(), arena.nbytes)
        assert store_.regions == [region, region]
        assert region.unregistered

    def test_ucx_without_an_l1_arena_is_refused(self, fake_autumn):
        _, mod = fake_autumn
        with pytest.raises(ValueError):
            make_adapter(transport="ucx")
        assert mod.transports == []

    def test_tcp_registers_nothing(self, fake_autumn):
        store_, _ = fake_autumn
        adapter = make_adapter()
        try:
            key = create_object_key(1)
            assert store(adapter, [key], [create_memory_obj(64)])
            assert load(adapter, [key], [create_memory_obj(64)]) == [True]
        finally:
            adapter.close()
        assert store_.regions == [None, None]
        assert store_.registered == []

    def test_status_names_the_backend(self, fake_autumn):
        adapter = make_adapter(namespace="lmc")
        try:
            status = adapter.report_status()
            assert status["type"] == "autumn"
            assert status["namespace"] == "lmc"
            assert status["is_healthy"] is True
        finally:
            adapter.close()

    def test_a_failing_batch_is_a_miss(self, fake_autumn):
        store_, _ = fake_autumn
        adapter = make_adapter()
        try:
            key = create_object_key(1)
            assert store(adapter, [key], [create_memory_obj(64)])
            store_.raise_gets = True
            assert load(adapter, [key], [create_memory_obj(64)]) == [False]
            store_.raise_gets = False
            assert load(adapter, [key], [create_memory_obj(64)]) == [True]
        finally:
            adapter.close()

    def test_concurrent_deletes_share_one_connection(self, fake_autumn):
        store_, _ = fake_autumn
        adapter = make_adapter(max_capacity_gb=1)
        try:
            keys = [create_object_key(i) for i in range(8)]
            assert store(adapter, keys, [create_memory_obj(16) for _ in keys])
            threads = [
                threading.Thread(target=adapter.delete, args=([k],)) for k in keys
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert lookup(adapter, keys) == [False] * 8
        finally:
            adapter.close()
        assert store_.connects == 1

    def test_close_is_idempotent_and_refuses_new_work(self, fake_autumn):
        # First Party
        from lmcache.v1.distributed.l2_adapters.autumn_l2_adapter import (
            AutumnConnector,
        )

        _, mod = fake_autumn
        conn = AutumnConnector(mod, "127.0.0.1:9001", "kvc")
        conn.submit_batch_exists(["m@0@0@ab"])
        conn.close()
        conn.close()
        with pytest.raises(RuntimeError):
            conn.submit_batch_exists(["m@0@0@ab"])


# =============================================================================
# Real cluster
# =============================================================================


@pytest.mark.skipif(
    not os.environ.get("AUTUMN_MANAGER"), reason="AUTUMN_MANAGER is not set"
)
class TestAutumnL2AdapterIntegration:
    @pytest.fixture(autouse=True)
    def setup_adapter(self):
        pytest.importorskip("autumn")
        self.adapter = make_adapter(
            manager=os.environ["AUTUMN_MANAGER"],
            namespace=os.environ.get("AUTUMN_NAMESPACE", "kvc"),
            credential_file=os.environ.get("AUTUMN_CREDENTIAL_FILE", ""),
            ttl_secs=600,
            max_capacity_gb=1,
        )
        # A fresh model name per run, so earlier runs cannot satisfy a lookup.
        self.model = "lmcache-test/%s" % uuid.uuid4().hex
        yield
        self.adapter.close()

    def test_store_lookup_load_delete(self):
        n = 16
        keys = [create_object_key(i, self.model) for i in range(n)]
        objs = [create_memory_obj(64 * 1024, float(i)) for i in range(n)]
        assert lookup(self.adapter, keys) == [False] * n
        assert store(self.adapter, keys, objs)
        assert lookup(self.adapter, keys) == [True] * n
        out = [create_memory_obj(64 * 1024, -1.0) for _ in range(n)]
        assert load(self.adapter, keys, out) == [True] * n
        for a, b in zip(objs, out, strict=True):
            assert torch.equal(a.tensor, b.tensor)
        self.adapter.submit_unlock(keys)
        self.adapter.delete(keys)
        assert lookup(self.adapter, keys) == [False] * n
