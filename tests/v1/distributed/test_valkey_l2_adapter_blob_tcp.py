# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the ``valkey`` adapter's ``blob-tcp`` transport.

These exercise the adapter against an in-process fake of ``glide_sync`` that
answers the valkey-large-object module's ``BLOB.*`` commands, so neither
``valkey-glide-sync`` nor a server with the module is needed. Error texts are
the ones real glide produces for the module's replies.
"""

# Standard
from typing import Any, Callable
import select
import sys
import threading
import time
import types

# Third Party
import pytest
import torch

# ---------------------------------------------------------------------------
# In-process fake of glide_sync
# ---------------------------------------------------------------------------
#
# Installed into ``sys.modules`` per test by the ``_fake_glide`` fixture, as
# the other valkey adapter tests do, so the real package is never shadowed
# for the integration tests collected after this file.

UNKNOWN_COMMAND = (
    "An error was signalled by the server: - ResponseError: unknown command "
    "'{name}', with args beginning with: "
)
DISABLED = "DISABLED: largeobj module is disabled"
WRONGTYPE = "WRONGTYPE: Operation against a key holding the wrong kind of value"


class RequestError(Exception):
    pass


class _BlobServer:
    """Shared state behind every fake client: the values, and what was sent."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        # key -> (type, value); type is "module" or "string".
        self.store: dict[bytes, tuple[str, bytes]] = {}
        self.ttls: dict[bytes, int] = {}
        # wire key -> exception raised by any BLOB command on that key.
        self.faults: dict[bytes, Exception] = {}
        # Every custom command, as a tuple of byte strings.
        self.calls: list[tuple[bytes, ...]] = []
        # "loaded", "absent", "disabled", or an exception BLOB.INFO raises.
        self.state: Any = "loaded"
        # Every client created, in order.
        self.clients: list["_FakeClient"] = []
        # Raised by every client's close(), if set.
        self.close_error: Exception | None = None

    def reset(self) -> None:
        with self.lock:
            self.store.clear()
            self.ttls.clear()
            self.faults.clear()
            self.calls.clear()
            self.clients.clear()
        self.state = "loaded"
        self.close_error = None

    def commands(self, name: bytes) -> list[tuple[bytes, ...]]:
        return [c for c in self.calls if c[0] == name]


_SERVER = _BlobServer()


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False

    @classmethod
    def create(cls, config: Any) -> "_FakeClient":
        client = cls()
        with _SERVER.lock:
            _SERVER.clients.append(client)
        return client

    def custom_command(self, args: list, route: Any = None) -> Any:
        name = bytes(args[0]).upper()
        with _SERVER.lock:
            _SERVER.calls.append(tuple(bytes(a) for a in args))
        if not name.startswith(b"BLOB."):
            raise RequestError(UNKNOWN_COMMAND.format(name=name.decode()))
        if _SERVER.state == "absent":
            raise RequestError(UNKNOWN_COMMAND.format(name=name.decode()))
        if _SERVER.state == "disabled":
            raise RequestError(DISABLED)
        if name == b"BLOB.INFO" and isinstance(_SERVER.state, Exception):
            raise _SERVER.state
        key = bytes(args[1])
        if key in _SERVER.faults:
            raise _SERVER.faults[key]
        with _SERVER.lock:
            entry = _SERVER.store.get(key)
            if entry is not None and entry[0] != "module":
                raise RequestError(WRONGTYPE)
            if name == b"BLOB.TCP_SET":
                # bytes() copies, so the fake never aliases the caller's buffer.
                _SERVER.store[key] = ("module", bytes(args[2]))
                return "OK"
            if name == b"BLOB.TCP_GET":
                return None if entry is None else entry[1]
            if name == b"BLOB.INFO":
                return None if entry is None else [b"len", len(entry[1])]
        raise RequestError(UNKNOWN_COMMAND.format(name=name.decode()))

    # The base pool inspects ``get`` for the buffer capability during warm-up;
    # the blob-tcp pool never calls it.
    def get(self, key: bytes, buffer: Any = None) -> Any:
        raise AssertionError("blob-tcp must not use plain GET")

    def expire(self, key: bytes, seconds: int, option: Any = None) -> bool:
        with _SERVER.lock:
            if bytes(key) not in _SERVER.store:
                return False
            _SERVER.ttls[bytes(key)] = seconds
        return True

    def exists(self, keys: list[bytes]) -> int:
        with _SERVER.lock:
            return sum(1 for k in keys if bytes(k) in _SERVER.store)

    def delete(self, keys: list[bytes]) -> int:
        n = 0
        with _SERVER.lock:
            for k in keys:
                if _SERVER.store.pop(bytes(k), None) is not None:
                    n += 1
        return n

    def close(self) -> None:
        self.closed = True
        if _SERVER.close_error is not None:
            raise _SERVER.close_error


class _FakeClusterClient(_FakeClient):
    def info(self, sections: Any = None, route: Any = None) -> dict:
        return {}


def _build_fake_glide_modules() -> dict[str, types.ModuleType]:
    fake = types.ModuleType("glide_sync")

    def _record(name: str) -> Callable[..., tuple[str, dict]]:
        def _ctor(**kw: Any) -> tuple[str, dict]:
            return (name, kw)

        return _ctor

    fake.ServerCredentials = lambda u, p: ("creds", u, p)  # type: ignore[attr-defined]
    fake.NodeAddress = lambda h, p: ("addr", h, p)  # type: ignore[attr-defined]
    fake.AdvancedGlideClientConfiguration = _record("adv_std")  # type: ignore[attr-defined]
    fake.AdvancedGlideClusterClientConfiguration = _record(  # type: ignore[attr-defined]
        "adv_cluster"
    )
    fake.GlideClientConfiguration = _record("cfg_std")  # type: ignore[attr-defined]
    fake.GlideClusterClientConfiguration = _record("cfg_cluster")  # type: ignore[attr-defined]
    fake.GlideClient = _FakeClient  # type: ignore[attr-defined]
    fake.GlideClusterClient = _FakeClusterClient  # type: ignore[attr-defined]
    fake.RequestError = RequestError  # type: ignore[attr-defined]

    routes_mod = types.ModuleType("glide_shared.routes")
    routes_mod.AllNodes = lambda: ("route", "all_nodes")  # type: ignore[attr-defined]
    core_opts_mod = types.ModuleType("glide_shared.commands.core_options")

    class _InfoSection:
        MEMORY = "memory"

    core_opts_mod.InfoSection = _InfoSection  # type: ignore[attr-defined]
    return {
        "glide_sync": fake,
        "glide_shared": types.ModuleType("glide_shared"),
        "glide_shared.commands": types.ModuleType("glide_shared.commands"),
        "glide_shared.commands.core_options": core_opts_mod,
        "glide_shared.routes": routes_mod,
    }


# First Party
from lmcache.v1.common.object_key import ObjectKey  # noqa: E402
from lmcache.v1.distributed.api import MemoryLayoutDesc  # noqa: E402
from lmcache.v1.distributed.l2_adapters.factory import (  # noqa: E402
    create_l2_adapter_from_registry,
)
from lmcache.v1.distributed.l2_adapters.valkey.blob_tcp_worker_pool import (  # noqa: E402
    ValkeyBlobTcpWorkerPool,
)
from lmcache.v1.distributed.l2_adapters.valkey_l2_adapter import (  # noqa: E402
    ValkeyL2Adapter,
    ValkeyL2AdapterConfig,
)
from lmcache.v1.memory_management import (  # noqa: E402
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd  # noqa: E402

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _obj(nbytes: int = 256, fill: float = 1.0) -> TensorMemoryObj:
    """An object backed by its own tensor; blob-tcp needs no arena."""
    raw = torch.empty(nbytes // 4, dtype=torch.float32)
    raw.fill_(fill)
    meta = MemoryObjMetadata(
        shape=torch.Size([nbytes // 4]),
        dtype=torch.float32,
        address=0,
        phy_size=nbytes,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw, meta, parent_allocator=None)


def _bytes_of(obj: TensorMemoryObj) -> bytes:
    return bytes(obj.byte_array)


def _key(chunk_id: int, model_name: str = "test_model") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
        cache_salt="",
    )


def _wire(adapter: ValkeyL2Adapter, key: ObjectKey) -> bytes:
    return adapter._wire_key(key).encode()  # noqa: SLF001


def _wait(fd: int, ready: Callable[[], Any], timeout: float = 5.0) -> Any:
    """Poll ``fd`` until ``ready()`` returns something other than None."""
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not poll.poll(remaining * 1000):
            break
        try:
            consume_fd(fd)
        except BlockingIOError:
            pass
        result = ready()
        if result is not None:
            return result
    raise AssertionError(f"no completion within {timeout}s")


def _store(adapter: ValkeyL2Adapter, keys: list, objs: list) -> Any:
    task_id = adapter.submit_store_task(keys, objs)
    return _wait(
        adapter.get_store_event_fd(),
        lambda: adapter.pop_completed_store_tasks().get(task_id),
    )


def _load(adapter: ValkeyL2Adapter, keys: list, objs: list) -> Any:
    task_id = adapter.submit_load_task(keys, objs)
    return _wait(
        adapter.get_load_event_fd(), lambda: adapter.query_load_result(task_id)
    )


def _lookup(adapter: ValkeyL2Adapter, keys: list) -> Any:
    task_id = adapter.submit_lookup_and_lock_task(keys, {0: _EMPTY_LAYOUT})
    return _wait(
        adapter.get_lookup_and_lock_event_fd(),
        lambda: adapter.query_lookup_and_lock_result(task_id),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_glide(monkeypatch: pytest.MonkeyPatch):
    for name, module in _build_fake_glide_modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    yield


@pytest.fixture(autouse=True)
def _reset_state():
    _SERVER.reset()
    yield
    _SERVER.reset()


def _make_config(**overrides: Any) -> ValkeyL2AdapterConfig:
    base: dict[str, Any] = {
        "startup_nodes": [("localhost", 6379)],
        "num_workers": 2,
        "connection_timeout": 2.0,
        "request_timeout": 2.0,
        "transport": "blob-tcp",
    }
    base.update(overrides)
    return ValkeyL2AdapterConfig(**base)


@pytest.fixture
def adapter():
    a = ValkeyL2Adapter(_make_config())
    yield a
    a.close()


# ===========================================================================
# Config and construction
# ===========================================================================


class TestConfig:
    def test_from_dict_accepts_blob_tcp(self):
        config = ValkeyL2AdapterConfig.from_dict(
            {
                "type": "valkey",
                "transport": "blob-tcp",
                "startup_nodes": "localhost:6379",
            }
        )
        assert config.transport == "blob-tcp"
        assert not config.uses_rdma

    def test_help_mentions_blob_tcp(self):
        assert "blob-tcp" in ValkeyL2AdapterConfig.help()

    def test_builds_without_an_l1_memory_descriptor(self):
        adapter = create_l2_adapter_from_registry(_make_config(), None)
        try:
            assert isinstance(adapter, ValkeyL2Adapter)
        finally:
            adapter.close()


class TestStartupCheck:
    def test_a_new_client_checks_the_module_and_later_requests_do_not(self):
        adapter = ValkeyL2Adapter(_make_config(num_workers=1))
        try:
            assert _SERVER.commands(b"BLOB.INFO") == [
                (b"BLOB.INFO", b"__lmcache_blob_tcp_probe__")
            ]

            assert _store(adapter, [_key(1)], [_obj()]).is_successful()
            assert _load(adapter, [_key(1)], [_obj()]).test(0)

            assert len(_SERVER.commands(b"BLOB.INFO")) == 1
        finally:
            adapter.close()

    def test_a_missing_module_fails_construction(self):
        _SERVER.state = "absent"
        with pytest.raises(RuntimeError, match="--loadmodule"):
            ValkeyL2Adapter(_make_config())

    def test_a_client_that_fails_the_check_is_closed(self):
        _SERVER.state = "absent"
        with pytest.raises(RuntimeError):
            ValkeyL2Adapter(_make_config(num_workers=1))
        assert len(_SERVER.clients) == 1
        assert _SERVER.clients[0].closed

    def test_a_failing_close_does_not_hide_the_module_error(self):
        _SERVER.state = "absent"
        _SERVER.close_error = RequestError("connection already broken")
        with pytest.raises(RuntimeError, match="--loadmodule"):
            ValkeyL2Adapter(_make_config(num_workers=1))

    def test_cluster_mode_checks_and_round_trips(self):
        adapter = ValkeyL2Adapter(_make_config(cluster_mode=True, num_workers=1))
        try:
            assert isinstance(_SERVER.clients[0], _FakeClusterClient)
            assert len(_SERVER.commands(b"BLOB.INFO")) == 1
            src = _obj(256, fill=4.0)
            assert _store(adapter, [_key(1)], [src]).is_successful()
            dst = _obj(256, fill=0.0)
            assert _load(adapter, [_key(1)], [dst]).test(0)
            assert _bytes_of(dst) == _bytes_of(src)
        finally:
            adapter.close()

    def test_a_disabled_module_fails_construction(self):
        _SERVER.state = "disabled"
        with pytest.raises(RuntimeError, match="largeobj.enabled yes"):
            ValkeyL2Adapter(_make_config())

    def test_another_error_propagates_unchanged(self):
        _SERVER.state = RequestError("NOPERM this user has no permissions")
        with pytest.raises(RequestError, match="NOPERM"):
            ValkeyL2Adapter(_make_config())

    def test_a_string_at_the_probe_key_propagates_wrongtype(self):
        _SERVER.store[b"__lmcache_blob_tcp_probe__"] = ("string", b"x")
        with pytest.raises(RequestError, match="WRONGTYPE"):
            ValkeyL2Adapter(_make_config())


# ===========================================================================
# Store
# ===========================================================================


class TestStore:
    def test_chunks_are_stored_as_module_objects(self, adapter: ValkeyL2Adapter):
        objs = [_obj(256, fill=1.0), _obj(512, fill=2.0)]
        keys = [_key(1), _key(2)]

        result = _store(adapter, keys, objs)

        assert result.is_successful()
        assert result.bytes_transferred() == 256 + 512
        for key, obj in zip(keys, objs, strict=True):
            assert _SERVER.store[_wire(adapter, key)] == ("module", _bytes_of(obj))

    def test_ttl_is_applied_after_the_store(self):
        adapter = ValkeyL2Adapter(_make_config(ttl_seconds=30))
        try:
            key = _key(1)
            assert _store(adapter, [key], [_obj()]).is_successful()
            assert _SERVER.ttls[_wire(adapter, key)] == 30
        finally:
            adapter.close()

    def test_no_ttl_by_default(self, adapter: ValkeyL2Adapter):
        assert _store(adapter, [_key(1)], [_obj()]).is_successful()
        assert _SERVER.ttls == {}

    def test_a_server_error_fails_only_its_own_key(self, adapter: ValkeyL2Adapter):
        keys = [_key(1), _key(2)]
        _SERVER.faults[_wire(adapter, keys[1])] = RequestError(
            "ERR max object size exceeded"
        )

        result = _store(adapter, keys, [_obj(), _obj()])

        assert not result.is_successful()
        assert _wire(adapter, keys[0]) in _SERVER.store
        assert _wire(adapter, keys[1]) not in _SERVER.store

    def test_a_string_key_fails_the_store(self, adapter: ValkeyL2Adapter):
        key = _key(1)
        _SERVER.store[_wire(adapter, key)] = ("string", b"plain")

        assert not _store(adapter, [key], [_obj()]).is_successful()
        assert _SERVER.store[_wire(adapter, key)] == ("string", b"plain")

    def test_an_empty_value_is_refused_before_sending(self):
        pool = ValkeyBlobTcpWorkerPool(addresses=[("localhost", 6379)], num_workers=1)
        try:
            with pytest.raises(ValueError, match="empty"):
                pool.submit_set("k", memoryview(b"")).result(timeout=5)
            assert _SERVER.commands(b"BLOB.TCP_SET") == []
        finally:
            pool.close()


# ===========================================================================
# Load
# ===========================================================================


class TestLoad:
    def test_a_stored_chunk_loads_back(self, adapter: ValkeyL2Adapter):
        src = _obj(256, fill=3.0)
        assert _store(adapter, [_key(1)], [src]).is_successful()
        dst = _obj(256, fill=0.0)

        loaded = _load(adapter, [_key(1)], [dst])

        assert loaded.test(0)
        assert _bytes_of(dst) == _bytes_of(src)

    def test_a_missing_key_is_not_loaded(self, adapter: ValkeyL2Adapter):
        dst = _obj(256, fill=7.0)
        before = _bytes_of(dst)

        assert not _load(adapter, [_key(1)], [dst]).test(0)
        assert _bytes_of(dst) == before

    @pytest.mark.parametrize("stored_bytes", [512, 128], ids=["larger", "smaller"])
    def test_a_value_of_another_size_is_not_loaded(
        self, adapter: ValkeyL2Adapter, stored_bytes: int
    ):
        key = _key(1)
        _SERVER.store[_wire(adapter, key)] = ("module", b"\x01" * stored_bytes)
        dst = _obj(256, fill=7.0)
        before = _bytes_of(dst)

        assert not _load(adapter, [key], [dst]).test(0)
        assert _bytes_of(dst) == before

    def test_a_string_key_is_not_loaded(self, adapter: ValkeyL2Adapter):
        key = _key(1)
        _SERVER.store[_wire(adapter, key)] = ("string", b"\x01" * 256)
        dst = _obj(256, fill=7.0)
        before = _bytes_of(dst)

        assert not _load(adapter, [key], [dst]).test(0)
        assert _bytes_of(dst) == before

    def test_lookup_then_load_round_trip(self, adapter: ValkeyL2Adapter):
        keys = [_key(1), _key(2)]
        assert _store(adapter, keys, [_obj(), _obj()]).is_successful()

        found = _lookup(adapter, keys + [_key(3)])

        assert found.test(0) and found.test(1) and not found.test(2)
        adapter.submit_unlock(keys)


# ===========================================================================
# Delete and status
# ===========================================================================


class TestDeleteAndStatus:
    def test_delete_removes_stored_keys(self, adapter: ValkeyL2Adapter):
        keys = [_key(1), _key(2)]
        assert _store(adapter, keys, [_obj(), _obj()]).is_successful()

        adapter.delete(keys)

        assert _SERVER.store == {}

    def test_status_names_the_transport_without_rdma_settings(
        self, adapter: ValkeyL2Adapter
    ):
        status = adapter.report_status()
        assert status["is_healthy"]
        assert status["transport"] == "blob-tcp"
        assert "rdma_provider" not in status
        assert "l1_arena_bytes" not in status
