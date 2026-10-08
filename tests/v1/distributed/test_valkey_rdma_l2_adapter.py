# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ValkeyRdmaL2Adapter.

These exercise the adapter against an in-process fake of the ``glide_sync``
RDMA API, so neither ``valkey-glide-sync`` nor a server with the
valkey-large-object module is needed.
"""

# Standard
from dataclasses import dataclass
from typing import Any, Callable, Optional
import ctypes
import select
import sys
import threading
import time
import types

# Third Party
import pytest
import torch

# ---------------------------------------------------------------------------
# In-process fake of glide_sync's RDMA surface
# ---------------------------------------------------------------------------
#
# Installed into ``sys.modules`` per test by the ``_fake_glide`` fixture, as
# ``test_valkey_l2_adapter.py`` does, so the real package is never shadowed
# for the integration tests collected after this file.


class RdmaError(Exception):
    """Stands in for ``glide_sync.RdmaError``."""


@dataclass(frozen=True)
class RdmaReadReceipt:
    bytes_written: int
    checksum: Optional[int] = None


class _RdmaServer:
    """Shared state behind every fake client: the values, and what happened."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.store: dict[bytes, bytes] = {}
        self.ttls: dict[bytes, int] = {}
        # Every region any client registered, in order.
        self.registrations: list[_FakeRegion] = []
        # Every transfer as (op, key, offset, length).
        self.transfers: list[tuple[str, bytes, int, int]] = []
        # op -> exception to raise, or callable(key) -> Optional[exception].
        self.faults: dict[str, Any] = {}
        # What the fake client answers for the two support checks.
        self.available = True
        self.usable = True
        # Client configurations, in creation order.
        self.configs: list[Any] = []

    def reset(self) -> None:
        with self.lock:
            self.store.clear()
            self.ttls.clear()
        self.registrations.clear()
        self.transfers.clear()
        self.faults.clear()
        self.configs.clear()
        self.available = True
        self.usable = True

    def maybe_fault(self, op: str, key: bytes) -> None:
        fault = self.faults.get(op)
        if fault is None:
            return
        if callable(fault):
            result = fault(key)
            if result is not None:
                raise result
        else:
            raise fault


_SERVER = _RdmaServer()


@dataclass(frozen=True)
class _FakeWindow:
    region: "_FakeRegion"
    offset: int
    length: int

    def memoryview(self) -> memoryview:
        return self.region.memoryview()[self.offset : self.offset + self.length]


class _FakeRegion:
    def __init__(self, client: "_FakeRdmaClient", memory: Any) -> None:
        self.client = client
        self._memory = memory
        self.capacity = memoryview(memory).cast("B").nbytes
        self.closed = False
        # Set by a failed transfer, as glide does; the next transfer through
        # the region is refused.
        self.revoked = False

    def window(self, offset: int = 0, length: Optional[int] = None) -> _FakeWindow:
        if self.closed:
            raise RdmaError("this RDMA region is closed")
        if length is None:
            length = self.capacity - offset
        if offset < 0 or length < 0 or offset + length > self.capacity:
            raise RdmaError(
                f"window [{offset}, {offset + length}) runs past the end of a "
                f"region of {self.capacity} bytes"
            )
        return _FakeWindow(self, offset, length)

    def memoryview(self) -> memoryview:
        if self.closed:
            raise RdmaError("this RDMA region is closed")
        return memoryview(self._memory).cast("B")

    def close(self) -> None:
        self.closed = True


class _FakeRdmaClient:
    """Enough of ``glide_sync.GlideClient`` for the RDMA pool."""

    def __init__(self) -> None:
        self.closed = False
        self.config: Any = None

    @classmethod
    def create(cls, config: Any) -> "_FakeRdmaClient":
        inst = cls()
        inst.config = config
        _SERVER.configs.append(config)
        return inst

    @staticmethod
    def rdma_available() -> bool:
        return _SERVER.available

    @staticmethod
    def rdma_usable() -> bool:
        return _SERVER.usable

    def register_rdma_region(self, memory: Any) -> _FakeRegion:
        if memoryview(memory).readonly:
            raise TypeError("memory must be writable")
        region = _FakeRegion(self, memory)
        _SERVER.registrations.append(region)
        return region

    def _check(self, window: _FakeWindow) -> None:
        if window.region.client is not self:
            raise RdmaError("this region is registered with a different client")
        if window.region.closed:
            raise RdmaError("this RDMA region is closed")
        if window.region.revoked:
            raise RdmaError(
                "RDMA transfer cancelled - the region was revoked, so nothing "
                "can transfer through it"
            )

    def rdma_set(self, key: bytes, window: _FakeWindow) -> None:
        self._check(window)
        key = bytes(key)
        _SERVER.transfers.append(("set", key, window.offset, window.length))
        try:
            _SERVER.maybe_fault("set", key)
        except Exception:
            window.region.revoked = True
            raise
        with _SERVER.lock:
            _SERVER.store[key] = bytes(window.memoryview())

    def rdma_get(self, key: bytes, window: _FakeWindow) -> Optional[RdmaReadReceipt]:
        self._check(window)
        key = bytes(key)
        _SERVER.transfers.append(("get", key, window.offset, window.length))
        try:
            _SERVER.maybe_fault("get", key)
        except Exception:
            window.region.revoked = True
            raise
        with _SERVER.lock:
            value = _SERVER.store.get(key)
        if value is None:
            return None
        # As the module does with the window length: refuse before writing.
        if len(value) > window.length:
            raise RdmaError("client address space smaller than object length")
        window.memoryview()[: len(value)] = value
        return RdmaReadReceipt(bytes_written=len(value))

    # Plain commands a real client also has; the pool probes ``get`` for the
    # buffer capability. The RDMA pool never uses them for a transfer.
    def set(self, key: bytes, value: Any, expiry: Any = None) -> None:
        with _SERVER.lock:
            _SERVER.store[bytes(key)] = bytes(value)

    def get(self, key: bytes, buffer: Any = None) -> Any:
        with _SERVER.lock:
            value = _SERVER.store.get(bytes(key))
        if value is None or buffer is None:
            return value
        n = min(len(value), len(buffer))
        buffer[:n] = value[:n]
        return n

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


class _FakeRdmaClusterClient(_FakeRdmaClient):
    def info(self, sections: Any = None, route: Any = None) -> dict:
        return {}


@dataclass(frozen=True)
class _EfaDirect:
    pass


@dataclass(frozen=True)
class _Tcp:
    bind: Optional[str] = None


class _RdmaProvider:
    EfaDirect = _EfaDirect
    Tcp = _Tcp


@dataclass
class _RdmaConfiguration:
    provider: Any
    interface: Optional[str] = None


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
    fake.GlideClient = _FakeRdmaClient  # type: ignore[attr-defined]
    fake.GlideClusterClient = _FakeRdmaClusterClient  # type: ignore[attr-defined]
    fake.RdmaConfiguration = _RdmaConfiguration  # type: ignore[attr-defined]
    fake.RdmaProvider = _RdmaProvider  # type: ignore[attr-defined]
    fake.RdmaError = RdmaError  # type: ignore[attr-defined]

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
from lmcache.v1.distributed.api import (  # noqa: E402
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.distributed.internal_api import L1MemoryDesc  # noqa: E402
from lmcache.v1.distributed.l2_adapters.config import (  # noqa: E402
    get_type_name_for_config,
)
from lmcache.v1.distributed.l2_adapters.factory import (  # noqa: E402
    create_l2_adapter_from_registry,
)
from lmcache.v1.distributed.l2_adapters.valkey_l2_adapter import (  # noqa: E402
    ValkeyL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.valkey_rdma_l2_adapter import (  # noqa: E402
    ValkeyRdmaL2Adapter,
    ValkeyRdmaL2AdapterConfig,
)
from lmcache.v1.memory_management import (  # noqa: E402
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd  # noqa: E402

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])

ARENA_BYTES = 1 << 16
ALIGN = 4096


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _Arena:
    """A stand-in for the L1 arena: one tensor, objects carved out of it at
    aligned offsets, so each has a known place for the window assertions."""

    def __init__(self, size: int = ARENA_BYTES, align: int = ALIGN) -> None:
        self.tensor = torch.zeros(size, dtype=torch.uint8)
        self.align = align
        self.desc = L1MemoryDesc(
            ptr=self.tensor.data_ptr(), size=size, align_bytes=align
        )
        self._next = 0

    def alloc(self, nbytes: int = 256, fill: float = 1.0) -> TensorMemoryObj:
        assert nbytes % 4 == 0
        offset = self._next
        phy_size = -(-nbytes // self.align) * self.align
        self._next += phy_size
        assert self._next <= self.desc.size, "arena exhausted"
        raw = self.tensor[offset : offset + nbytes].view(torch.float32)
        raw.fill_(fill)
        meta = MemoryObjMetadata(
            shape=torch.Size([nbytes // 4]),
            dtype=torch.float32,
            address=offset,
            phy_size=phy_size,
            fmt=MemoryFormat.KV_2LTD,
            ref_count=1,
        )
        return TensorMemoryObj(raw, meta, parent_allocator=None)

    def bytes_of(self, obj: TensorMemoryObj) -> bytes:
        start = obj.meta.address
        return bytes(self.tensor[start : start + obj.get_size()].tolist())


def _outside_arena_obj(nbytes: int = 256, fill: float = 1.0) -> TensorMemoryObj:
    """An object backed by its own tensor, so it lives outside any arena."""
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


def _key(chunk_id: int, model_name: str = "test_model") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
        cache_salt="",
    )


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


def _store(adapter: ValkeyRdmaL2Adapter, keys: list, objs: list) -> Any:
    task_id = adapter.submit_store_task(keys, objs)
    return _wait(
        adapter.get_store_event_fd(),
        lambda: adapter.pop_completed_store_tasks().get(task_id),
    )


def _load(adapter: ValkeyRdmaL2Adapter, keys: list, objs: list) -> Any:
    task_id = adapter.submit_load_task(keys, objs)
    return _wait(
        adapter.get_load_event_fd(), lambda: adapter.query_load_result(task_id)
    )


def _lookup(adapter: ValkeyRdmaL2Adapter, keys: list) -> Any:
    task_id = adapter.submit_lookup_and_lock_task(keys, {0: _EMPTY_LAYOUT})
    return _wait(
        adapter.get_lookup_and_lock_event_fd(),
        lambda: adapter.query_lookup_and_lock_result(task_id),
    )


def _transfers(op: str) -> list[tuple[bytes, int, int]]:
    return [(k, off, ln) for o, k, off, ln in _SERVER.transfers if o == op]


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


def _make_config(**overrides: Any) -> ValkeyRdmaL2AdapterConfig:
    base: dict[str, Any] = {
        "startup_nodes": [("localhost", 6379)],
        "num_workers": 2,
        "connection_timeout": 2.0,
        "request_timeout": 2.0,
    }
    base.update(overrides)
    return ValkeyRdmaL2AdapterConfig(**base)


@pytest.fixture
def arena() -> _Arena:
    return _Arena()


@pytest.fixture
def adapter(arena: _Arena):
    a = ValkeyRdmaL2Adapter(_make_config(), arena.desc)
    yield a
    a.close()


# ===========================================================================
# Config
# ===========================================================================


class TestConfig:
    def test_defaults_to_efa_direct_with_no_interface(self):
        cfg = _make_config()
        assert cfg.rdma_provider == "efa-direct"
        assert cfg.rdma_interface is None
        assert isinstance(cfg, ValkeyL2AdapterConfig)

    def test_rejects_unknown_provider(self):
        with pytest.raises(ValueError, match="rdma_provider"):
            _make_config(rdma_provider="verbs")

    def test_rejects_empty_interface(self):
        with pytest.raises(ValueError, match="rdma_interface"):
            _make_config(rdma_interface="")

    def test_inherited_validation_still_applies(self):
        with pytest.raises(ValueError, match="startup_nodes"):
            ValkeyRdmaL2AdapterConfig(startup_nodes=[])

    def test_from_dict_parses_rdma_and_inherited_fields(self):
        cfg = ValkeyRdmaL2AdapterConfig.from_dict(
            {
                "type": "valkey_rdma",
                "startup_nodes": "h1:6379,h2:6380",
                "num_workers": 3,
                "ttl_seconds": 60,
                "rdma_provider": "tcp",
                "rdma_interface": "efa0",
            }
        )
        assert cfg.startup_nodes == [("h1", 6379), ("h2", 6380)]
        assert cfg.num_workers == 3
        assert cfg.ttl_seconds == 60
        assert cfg.rdma_provider == "tcp"
        assert cfg.rdma_interface == "efa0"

    def test_from_dict_defaults_rdma_fields(self):
        cfg = ValkeyRdmaL2AdapterConfig.from_dict({"startup_nodes": "h:1"})
        assert cfg.rdma_provider == "efa-direct"
        assert cfg.rdma_interface is None

    @pytest.mark.parametrize(
        "field, value",
        [("rdma_provider", 7), ("rdma_interface", 7), ("rdma_provider", "verbs")],
    )
    def test_from_dict_rejects_bad_rdma_values(self, field: str, value: Any):
        with pytest.raises(ValueError, match=field):
            ValkeyRdmaL2AdapterConfig.from_dict({"startup_nodes": "h:1", field: value})

    def test_help_covers_inherited_and_rdma_fields(self):
        text = ValkeyRdmaL2AdapterConfig.help()
        assert "startup_nodes" in text
        assert "rdma_provider" in text
        assert "rdma_interface" in text

    def test_registered_under_its_own_type_name(self):
        assert get_type_name_for_config(_make_config()) == "valkey_rdma"
        # The parent config still resolves to the parent type.
        plain = ValkeyL2AdapterConfig(startup_nodes=[("h", 1)])
        assert get_type_name_for_config(plain) == "valkey"


# ===========================================================================
# Factory
# ===========================================================================


class TestFactory:
    def test_builds_the_rdma_adapter_from_the_registry(self, arena: _Arena):
        adapter = create_l2_adapter_from_registry(_make_config(), arena.desc)
        try:
            assert isinstance(adapter, ValkeyRdmaL2Adapter)
        finally:
            adapter.close()

    def test_requires_an_l1_memory_descriptor(self):
        with pytest.raises(ValueError, match="L1 memory descriptor"):
            create_l2_adapter_from_registry(_make_config(), None)

    @pytest.mark.parametrize(
        "desc",
        [
            L1MemoryDesc(ptr=0, size=65536, align_bytes=4096),
            L1MemoryDesc(ptr=0x1000, size=0, align_bytes=4096),
        ],
    )
    def test_rejects_an_invalid_descriptor(self, desc: L1MemoryDesc):
        with pytest.raises(ValueError, match="invalid L1 memory descriptor"):
            create_l2_adapter_from_registry(_make_config(), desc)


# ===========================================================================
# RDMA support checks
# ===========================================================================


class TestSupportChecks:
    def test_client_built_without_rdma_is_refused(self, arena: _Arena):
        _SERVER.available = False
        with pytest.raises(RuntimeError, match="without RDMA support"):
            ValkeyRdmaL2Adapter(_make_config(), arena.desc)

    def test_missing_libfabric_is_refused(self, arena: _Arena):
        _SERVER.usable = False
        with pytest.raises(RuntimeError, match="libfabric"):
            ValkeyRdmaL2Adapter(_make_config(), arena.desc)

    def test_client_without_the_rdma_api_is_refused(
        self, arena: _Arena, monkeypatch: pytest.MonkeyPatch
    ):
        # An older glide-sync: a client class with no rdma_available at all.
        monkeypatch.delattr(sys.modules["glide_sync"].GlideClient, "rdma_available")
        with pytest.raises(RuntimeError, match="no RDMA API"):
            ValkeyRdmaL2Adapter(_make_config(), arena.desc)

    def test_a_failed_registration_names_workers_and_arena(
        self, arena: _Arena, monkeypatch: pytest.MonkeyPatch
    ):
        # libfabric reports the device's registration cap only as ENOMEM; the
        # operator needs to hear what to shrink.
        def refuse(self, memory):
            raise RdmaError(
                "fabric error in fi_mr_reg: Cannot allocate memory (errno -12)"
            )

        monkeypatch.setattr(_FakeRdmaClient, "register_rdma_region", refuse)
        with pytest.raises(RuntimeError, match="num_workers") as raised:
            ValkeyRdmaL2Adapter(_make_config(num_workers=3), arena.desc)
        message = str(raised.value)
        assert "3 workers" in message
        assert "fi_mr_reg" in message
        assert isinstance(raised.value.__cause__, RdmaError)

    def test_nothing_is_registered_when_refused(self, arena: _Arena):
        _SERVER.usable = False
        with pytest.raises(RuntimeError):
            ValkeyRdmaL2Adapter(_make_config(), arena.desc)
        assert _SERVER.registrations == []


# ===========================================================================
# Registration
# ===========================================================================


class TestRegistration:
    def test_each_worker_registers_the_whole_arena(self, arena: _Arena):
        adapter = ValkeyRdmaL2Adapter(_make_config(num_workers=3), arena.desc)
        try:
            # Warm-up may build fewer than three clients with an instant fake;
            # every client that exists has registered the whole arena once.
            assert 1 <= len(_SERVER.registrations) <= 3
            assert {r.capacity for r in _SERVER.registrations} == {arena.desc.size}
            clients = {id(r.client) for r in _SERVER.registrations}
            assert len(clients) == len(_SERVER.registrations)
            assert len(clients) == len(_SERVER.configs)
        finally:
            adapter.close()

    def test_registration_views_the_arena_itself(self, arena: _Arena):
        # The registration is a view of the L1 memory, not a copy.
        adapter = ValkeyRdmaL2Adapter(_make_config(num_workers=1), arena.desc)
        try:
            _SERVER.registrations[0].memoryview()[100:104] = b"\x01\x02\x03\x04"
            assert arena.tensor[100:104].tolist() == [1, 2, 3, 4]
        finally:
            adapter.close()

    def test_clients_are_configured_for_the_chosen_provider(self, arena: _Arena):
        adapter = ValkeyRdmaL2Adapter(
            _make_config(num_workers=1, rdma_provider="tcp", rdma_interface="lo"),
            arena.desc,
        )
        try:
            (name, kwargs) = _SERVER.configs[0]
            assert name == "cfg_std"
            rdma = kwargs["rdma"]
            assert isinstance(rdma.provider, _Tcp)
            assert rdma.interface == "lo"
        finally:
            adapter.close()

    def test_default_provider_is_efa_direct(self, arena: _Arena):
        adapter = ValkeyRdmaL2Adapter(_make_config(num_workers=1), arena.desc)
        try:
            rdma = _SERVER.configs[0][1]["rdma"]
            assert isinstance(rdma.provider, _EfaDirect)
            assert rdma.interface is None
        finally:
            adapter.close()

    def test_cluster_clients_are_configured_too(self, arena: _Arena):
        adapter = ValkeyRdmaL2Adapter(
            _make_config(num_workers=1, cluster_mode=True), arena.desc
        )
        try:
            (name, kwargs) = _SERVER.configs[0]
            assert name == "cfg_cluster"
            assert "rdma" in kwargs
        finally:
            adapter.close()

    def test_close_releases_every_registration_and_client(self, arena: _Arena):
        adapter = ValkeyRdmaL2Adapter(_make_config(num_workers=2), arena.desc)
        adapter.close()
        assert all(r.closed for r in _SERVER.registrations)
        assert all(r.client.closed for r in _SERVER.registrations)
        # Idempotent.
        adapter.close()


# ===========================================================================
# Store
# ===========================================================================


class TestStore:
    def test_bytes_are_read_out_of_the_objects_own_window(
        self, adapter: ValkeyRdmaL2Adapter, arena: _Arena
    ):
        objs = [arena.alloc(256, fill=1.0), arena.alloc(512, fill=2.0)]
        keys = [_key(1), _key(2)]

        result = _store(adapter, keys, objs)

        assert result.is_successful()
        assert result.bytes_transferred() == 256 + 512
        for key, obj in zip(keys, objs, strict=True):
            wire = adapter._wire_key(key).encode()  # noqa: SLF001
            assert _SERVER.store[wire] == arena.bytes_of(obj)
        # Each window is the object's arena offset and logical size, unpadded.
        windows = {k: (off, ln) for k, off, ln in _transfers("set")}
        for key, obj in zip(keys, objs, strict=True):
            wire = adapter._wire_key(key).encode()  # noqa: SLF001
            assert windows[wire] == (obj.meta.address, obj.get_size())

    def test_ttl_is_applied_after_the_transfer(self, arena: _Arena):
        adapter = ValkeyRdmaL2Adapter(_make_config(ttl_seconds=30), arena.desc)
        try:
            key = _key(1)
            assert _store(adapter, [key], [arena.alloc()]).is_successful()
            wire = adapter._wire_key(key).encode()  # noqa: SLF001
            assert _SERVER.ttls[wire] == 30
        finally:
            adapter.close()

    def test_no_ttl_by_default(self, adapter: ValkeyRdmaL2Adapter, arena: _Arena):
        assert _store(adapter, [_key(1)], [arena.alloc()]).is_successful()
        assert _SERVER.ttls == {}

    def test_an_object_outside_the_arena_fails_its_key_only(
        self, adapter: ValkeyRdmaL2Adapter, arena: _Arena
    ):
        inside = arena.alloc()
        outside = _outside_arena_obj()
        keys = [_key(1), _key(2)]

        result = _store(adapter, keys, [inside, outside])

        assert not result.is_successful()
        wires = [adapter._wire_key(k).encode() for k in keys]  # noqa: SLF001
        assert wires[0] in _SERVER.store
        assert wires[1] not in _SERVER.store
        # Nothing was attempted for the object that cannot be addressed.
        assert [k for k, _, _ in _transfers("set")] == [wires[0]]

    def test_a_server_failure_is_a_failed_store(
        self, adapter: ValkeyRdmaL2Adapter, arena: _Arena
    ):
        _SERVER.faults["set"] = RdmaError("ERR DRAM buffer pool exhausted")
        result = _store(adapter, [_key(1)], [arena.alloc()])
        assert not result.is_successful()
        assert result.bytes_transferred() == 0

    def test_empty_batch_completes(self, adapter: ValkeyRdmaL2Adapter):
        assert _store(adapter, [], []).is_successful()


# ===========================================================================
# Load
# ===========================================================================


class TestLoad:
    def test_bytes_land_in_the_destination_objects_window(
        self, adapter: ValkeyRdmaL2Adapter, arena: _Arena
    ):
        sources = [arena.alloc(256, fill=3.0), arena.alloc(256, fill=4.0)]
        keys = [_key(1), _key(2)]
        assert _store(adapter, keys, sources).is_successful()
        _SERVER.transfers.clear()
        destinations = [arena.alloc(256, fill=0.0), arena.alloc(256, fill=0.0)]

        bitmap = _load(adapter, keys, destinations)

        assert bitmap.test(0) and bitmap.test(1)
        for src, dst in zip(sources, destinations, strict=True):
            assert arena.bytes_of(dst) == arena.bytes_of(src)
        windows = {k: (off, ln) for k, off, ln in _transfers("get")}
        for key, dst in zip(keys, destinations, strict=True):
            wire = adapter._wire_key(key).encode()  # noqa: SLF001
            assert windows[wire] == (dst.meta.address, dst.get_size())

    def test_a_missing_key_is_a_miss_and_leaves_the_object_alone(
        self, adapter: ValkeyRdmaL2Adapter, arena: _Arena
    ):
        dst = arena.alloc(256, fill=9.0)
        before = arena.bytes_of(dst)

        bitmap = _load(adapter, [_key(404)], [dst])

        assert not bitmap.test(0)
        assert arena.bytes_of(dst) == before

    def test_a_smaller_value_is_a_miss(
        self, adapter: ValkeyRdmaL2Adapter, arena: _Arena
    ):
        # Fixed-size chunks must round-trip exactly, as with the plain adapter.
        key = _key(1)
        assert _store(adapter, [key], [arena.alloc(128)]).is_successful()

        bitmap = _load(adapter, [key], [arena.alloc(256)])

        assert not bitmap.test(0)

    def test_a_larger_value_is_refused_before_anything_lands(
        self, adapter: ValkeyRdmaL2Adapter, arena: _Arena
    ):
        key = _key(1)
        assert _store(adapter, [key], [arena.alloc(512, fill=5.0)]).is_successful()
        dst = arena.alloc(256, fill=0.0)

        bitmap = _load(adapter, [key], [dst])

        assert not bitmap.test(0)
        assert arena.bytes_of(dst) == bytes(256)

    def test_an_object_outside_the_arena_is_a_miss_without_a_transfer(
        self, adapter: ValkeyRdmaL2Adapter, arena: _Arena
    ):
        key = _key(1)
        assert _store(adapter, [key], [arena.alloc()]).is_successful()
        _SERVER.transfers.clear()

        bitmap = _load(adapter, [key], [_outside_arena_obj()])

        assert not bitmap.test(0)
        assert _transfers("get") == []

    def test_lookup_then_load_round_trip(
        self, adapter: ValkeyRdmaL2Adapter, arena: _Arena
    ):
        keys = [_key(1), _key(2)]
        assert _store(adapter, keys, [arena.alloc(), arena.alloc()]).is_successful()

        found = _lookup(adapter, keys + [_key(3)])

        assert found.test(0) and found.test(1) and not found.test(2)
        adapter.submit_unlock(keys)


# ===========================================================================
# Revoked regions
# ===========================================================================


class TestRevokedRegion:
    def test_a_revoked_region_is_registered_again_and_the_transfer_retried(
        self, arena: _Arena
    ):
        adapter = ValkeyRdmaL2Adapter(_make_config(num_workers=1), arena.desc)
        try:
            # A failure that is not a server reply revokes the region.
            _SERVER.faults["set"] = ConnectionError("connection reset")
            assert not _store(adapter, [_key(1)], [arena.alloc()]).is_successful()
            assert _SERVER.registrations[0].revoked
            _SERVER.faults.clear()

            result = _store(adapter, [_key(2)], [arena.alloc()])

            assert result.is_successful()
            # The worker registered the arena once more and the retried
            # transfer went through the new registration.
            assert len(_SERVER.registrations) == 2
            assert _SERVER.registrations[0].closed
            assert not _SERVER.registrations[1].closed
        finally:
            adapter.close()

    def test_a_server_reply_does_not_cost_a_registration(self, arena: _Arena):
        adapter = ValkeyRdmaL2Adapter(_make_config(num_workers=1), arena.desc)
        try:
            key = _key(1)
            assert _store(adapter, [key], [arena.alloc(512)]).is_successful()
            # Refused with a reply, so the region stays usable as it is.
            assert not _load(adapter, [key], [arena.alloc(256)]).test(0)
            assert _store(adapter, [_key(2)], [arena.alloc()]).is_successful()
            assert len(_SERVER.registrations) == 1
        finally:
            adapter.close()


# ===========================================================================
# Delete and status
# ===========================================================================


class TestDeleteAndStatus:
    def test_delete_removes_stored_keys(
        self, adapter: ValkeyRdmaL2Adapter, arena: _Arena
    ):
        keys = [_key(1), _key(2)]
        assert _store(adapter, keys, [arena.alloc(), arena.alloc()]).is_successful()

        adapter.delete(keys)

        assert _SERVER.store == {}

    def test_status_names_the_type_and_rdma_settings(
        self, adapter: ValkeyRdmaL2Adapter, arena: _Arena
    ):
        status = adapter.report_status()
        assert status["is_healthy"]
        assert status["type"] == "valkey_rdma"
        assert status["rdma_provider"] == "efa-direct"
        assert status["rdma_interface"] is None
        assert status["l1_arena_bytes"] == arena.desc.size


# ===========================================================================
# Window derivation
# ===========================================================================


class TestWindowDerivation:
    def test_a_byte_view_of_arena_memory_is_addressed_by_its_offset(
        self, adapter: ValkeyRdmaL2Adapter, arena: _Arena
    ):
        # The pool addresses whatever buffer it is handed by where the bytes
        # are, so a view at an arbitrary offset in the arena is a window at
        # exactly that offset.
        raw = (ctypes.c_ubyte * 64).from_address(arena.desc.ptr + 4096)
        view = memoryview(raw).cast("B")
        view[:] = bytes(range(64))
        pool = adapter._pool  # noqa: SLF001
        pool.submit_set("direct", view).result(timeout=5)
        assert _transfers("set")[-1] == (b"direct", 4096, 64)
        assert _SERVER.store[b"direct"] == bytes(range(64))
